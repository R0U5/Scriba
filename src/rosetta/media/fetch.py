"""Turning a URL into a local media file.

Two strategies, chosen automatically:

* **Direct download** for plain media URLs (``https://host/talk.mp3``), streamed
  to disk with byte-accurate progress.
* **yt-dlp** for the ~1800 sites it supports (YouTube, podcasts, Vimeo,
  SoundCloud, news embeds…), which also gives us a real title and duration.
"""

from __future__ import annotations

import ipaddress
import mimetypes
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from ..config import Settings
from ..errors import DownloadError, InvalidURLError, JobCancelled
from ..logging_setup import get_logger
from ..models import MediaInfo
from ..util.text import slugify

log = get_logger(__name__)

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]

MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".m4b", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".ts", ".m4v", ".3gp", ".aiff", ".alac",
}
_CONTENT_DISPOSITION = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', re.IGNORECASE)


@dataclass(slots=True)
class FetchResult:
    path: Path
    info: MediaInfo
    from_cache: bool = False


# --------------------------------------------------------------------------- #
# URL validation
# --------------------------------------------------------------------------- #
def validate_url(url: str, *, allow_private: bool = False) -> str:
    """Normalise and sanity-check a user-supplied URL.

    Blocks non-HTTP schemes and (unless explicitly allowed) private/loopback
    addresses, so a locally hosted UI cannot be pointed at internal services.
    """
    url = (url or "").strip()
    if not url:
        raise InvalidURLError("Enter a URL to transcribe.")
    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError(
            f"Unsupported URL scheme '{parsed.scheme}'.",
            hint="Use an http:// or https:// link, or pass a local file path.",
        )
    if not parsed.hostname:
        raise InvalidURLError(f"'{url}' is not a valid URL.")

    if not allow_private and not _host_is_public(parsed.hostname):
        raise InvalidURLError(
            f"Refusing to fetch from private address '{parsed.hostname}'.",
            hint="Set allow_private_hosts = true in rosetta.toml to allow LAN/NAS sources.",
        )
    return url


def _host_is_public(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True  # Let the HTTP layer produce the real DNS error message.
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:  # pragma: no cover
            continue
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            return False
    return True


# --------------------------------------------------------------------------- #
# Strategy selection
# --------------------------------------------------------------------------- #
def _looks_like_direct_media(url: str) -> bool:
    path = urlparse(url).path
    return Path(unquote(path)).suffix.lower() in MEDIA_EXTENSIONS


def _head(url: str, settings: Settings) -> tuple[str, int | None, str | None]:
    """Return ``(content_type, content_length, filename)`` without downloading."""
    headers = {"User-Agent": settings.user_agent}
    try:
        with httpx.Client(follow_redirects=True, timeout=20.0, headers=headers) as client:
            response = client.head(url)
            if response.status_code >= 400 or "content-type" not in response.headers:
                # Some CDNs reject HEAD. Ask for one byte instead — streamed, so
                # a server that ignores Range does not dump a 2 GB file on us.
                with client.stream(
                    "GET", url, headers={**headers, "Range": "bytes=0-0"}
                ) as ranged:
                    response = ranged
    except httpx.HTTPError as exc:
        log.debug("HEAD %s failed: %s", url, exc)
        return "", None, None

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    length_header = response.headers.get("content-range") or response.headers.get("content-length")
    length: int | None = None
    if length_header:
        try:
            length = int(str(length_header).split("/")[-1])
        except ValueError:
            length = None
    filename = None
    disposition = response.headers.get("content-disposition")
    if disposition and (match := _CONTENT_DISPOSITION.search(disposition)):
        filename = unquote(match.group(1)).strip()
    return content_type, length, filename


def _is_media_content_type(content_type: str, url: str) -> bool:
    if content_type.startswith(("audio/", "video/")):
        return True
    if content_type in ("application/octet-stream", "binary/octet-stream", ""):
        return _looks_like_direct_media(url)
    return content_type in ("application/mp4", "application/ogg", "application/x-mpegurl")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def fetch_media(
    url: str,
    dest_dir: Path,
    settings: Settings,
    *,
    progress: ProgressCB | None = None,
    cancelled: CancelCB | None = None,
) -> FetchResult:
    """Download ``url`` into ``dest_dir`` and describe what was downloaded."""

    dest_dir.mkdir(parents=True, exist_ok=True)

    local = _as_local_path(url)
    if local is not None:
        info = MediaInfo(
            source_url=str(local),
            title=local.stem,
            filesize=local.stat().st_size,
            extractor="local file",
        )
        if progress:
            progress(1.0, f"Using local file {local.name}")
        return FetchResult(path=local, info=info, from_cache=True)

    url = validate_url(url, allow_private=settings.allow_private_hosts)
    if progress:
        progress(0.0, "Inspecting source…")

    content_type, length, filename = _head(url, settings)
    if length and length > settings.max_download_bytes:
        raise DownloadError(
            f"Source is {length / 1e9:.1f} GB, larger than the {settings.max_download_bytes / 1e9:.1f} GB limit.",
            hint="Raise max_download_bytes in rosetta.toml if this is expected.",
        )

    if _is_media_content_type(content_type, url):
        return _download_direct(
            url, dest_dir, settings,
            content_type=content_type, total=length, filename=filename,
            progress=progress, cancelled=cancelled,
        )

    try:
        return _download_with_ytdlp(
            url, dest_dir, settings, progress=progress, cancelled=cancelled
        )
    except JobCancelled:
        raise
    except DownloadError:
        if _looks_like_direct_media(url):
            log.info("yt-dlp failed; retrying %s as a direct download", url)
            return _download_direct(
                url, dest_dir, settings,
                content_type=content_type, total=length, filename=filename,
                progress=progress, cancelled=cancelled,
            )
        raise


def _as_local_path(url: str) -> Path | None:
    """Accept plain filesystem paths and ``file://`` URLs as sources.

    Windows drive-letter and backslash paths are treated as local even though
    they carry no ``://`` scheme — they must never fall through to URL
    validation, which would otherwise mangle ``C:\…`` into ``https://C:\…``.
    """
    if url.startswith("file://"):
        parsed = urlparse(url)
        candidate = Path(unquote(parsed.path))
        if parsed.netloc and not candidate.exists():  # Windows UNC-ish
            candidate = Path(unquote(f"//{parsed.netloc}{parsed.path}"))
        if candidate.is_dir():
            raise InvalidURLError(
                f"'{candidate}' is a folder, not a media file.",
                hint="Point RosettaAI at a single audio or video file inside it.",
            )
        if candidate.exists():
            return candidate.resolve()
        raise InvalidURLError(f"Local file not found: {candidate}")
    if "://" in url:
        return None

    # A drive letter or a backslash means a local (Windows) path, never a host.
    if url[:2].isascii() and len(url) > 2 and url[1] == ":" and url[2] in "\\/":
        candidate = Path(url.replace("\\", "/")).expanduser()
    elif "\\" in url:
        candidate = Path(url.replace("\\", "/")).expanduser()
    else:
        candidate = Path(url).expanduser()

    if candidate.is_dir():
        raise InvalidURLError(
            f"'{candidate}' is a folder, not a media file.",
            hint="Point RosettaAI at a single audio or video file inside it.",
        )
    if candidate.is_file():
        return candidate.resolve()
    # A Windows-style path that does not exist must not be re-interpreted as a
    # URL (it would become https://C:\…). Fail with a clear message instead.
    if url[:2].isascii() and len(url) > 2 and url[1] == ":" and url[2] in "\\/":
        raise InvalidURLError(
            f"Local file not found: {candidate}",
            hint="Check the path and the file name, then try again.",
        )
    return None


# --------------------------------------------------------------------------- #
# Direct HTTP download
# --------------------------------------------------------------------------- #
def _download_direct(
    url: str,
    dest_dir: Path,
    settings: Settings,
    *,
    content_type: str,
    total: int | None,
    filename: str | None,
    progress: ProgressCB | None,
    cancelled: CancelCB | None,
) -> FetchResult:
    name = filename or Path(unquote(urlparse(url).path)).name or "download"
    stem = slugify(Path(name).stem, fallback="audio")
    suffix = Path(name).suffix or mimetypes.guess_extension(content_type) or ".bin"
    target = dest_dir / f"{stem}{suffix}"

    headers = {"User-Agent": settings.user_agent}
    downloaded = 0
    try:
        with httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(30.0, read=settings.download_timeout),
            headers=headers,
        ) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            if total is None:
                header = response.headers.get("content-length")
                total = int(header) if header and header.isdigit() else None
            with target.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 18):
                    if cancelled is not None and cancelled():
                        raise JobCancelled()
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > settings.max_download_bytes:
                        raise DownloadError("Download exceeded the configured size limit.")
                    if progress:
                        fraction = downloaded / total if total else 0.0
                        progress(
                            min(fraction, 0.99),
                            f"Downloading · {_mb(downloaded)}"
                            + (f" / {_mb(total)}" if total else ""),
                        )
    except JobCancelled:
        target.unlink(missing_ok=True)
        raise
    except httpx.HTTPStatusError as exc:
        target.unlink(missing_ok=True)
        raise DownloadError(
            f"The server returned {exc.response.status_code} for this URL.",
            hint="The link may have expired or require a login.",
        ) from exc
    except httpx.HTTPError as exc:
        target.unlink(missing_ok=True)
        raise DownloadError(f"Download failed: {exc}") from exc

    if progress:
        progress(1.0, f"Downloaded {_mb(downloaded)}")
    info = MediaInfo(
        source_url=url,
        title=Path(name).stem or stem,
        filesize=downloaded,
        webpage_url=url,
        extractor="direct",
    )
    return FetchResult(path=target, info=info)


def _mb(value: float) -> str:
    mb = value / (1024 * 1024)
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"


# --------------------------------------------------------------------------- #
# yt-dlp download
# --------------------------------------------------------------------------- #
def _download_with_ytdlp(
    url: str,
    dest_dir: Path,
    settings: Settings,
    *,
    progress: ProgressCB | None,
    cancelled: CancelCB | None,
) -> FetchResult:
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError as YTDLPError
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise DownloadError(
            "yt-dlp is not installed, so site URLs cannot be resolved.",
            hint="Run 'pip install yt-dlp'.",
        ) from exc

    def hook(status: dict) -> None:
        if cancelled is not None and cancelled():
            raise JobCancelled()
        if not progress:
            return
        if status.get("status") == "downloading":
            done = status.get("downloaded_bytes") or 0
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            fraction = min(done / total, 0.99) if total else 0.0
            speed = status.get("speed")
            speed_text = f" · {_mb(speed)}/s" if speed else ""
            progress(fraction, f"Downloading · {_mb(done)}"
                     + (f" / {_mb(total)}" if total else "") + speed_text)
        elif status.get("status") == "finished":
            progress(1.0, "Download complete")

    options: dict = {
        "format": "bestaudio/bestaudio*/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "restrictfilenames": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "concurrent_fragment_downloads": 4,
        "progress_hooks": [hook],
        "max_filesize": settings.max_download_bytes,
        "http_headers": {"User-Agent": settings.user_agent},
        "logger": _YTDLPLogger(),
    }

    try:
        with YoutubeDL(options) as ydl:
            payload = ydl.extract_info(url, download=True)
    except JobCancelled:
        raise
    except YTDLPError as exc:
        raise DownloadError(
            _friendly_ytdlp_error(str(exc)),
            hint="Try a direct media link, or download the file and pass its local path.",
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected extractor failure
        raise DownloadError(f"Could not fetch media from this URL: {exc}") from exc

    if payload is None:
        raise DownloadError("The URL did not yield any media.")
    if payload.get("_type") == "playlist":
        entries = [e for e in payload.get("entries") or [] if e]
        if not entries:
            raise DownloadError("That playlist is empty.")
        payload = entries[0]

    path = _resolve_ytdlp_path(payload, dest_dir)
    info = MediaInfo(
        source_url=url,
        title=payload.get("title") or path.stem,
        duration=float(payload.get("duration") or 0.0),
        uploader=payload.get("uploader") or payload.get("channel"),
        webpage_url=payload.get("webpage_url") or url,
        extractor=payload.get("extractor_key") or payload.get("extractor"),
        filesize=path.stat().st_size if path.exists() else None,
        thumbnail=payload.get("thumbnail"),
    )
    return FetchResult(path=path, info=info)


def _resolve_ytdlp_path(payload: dict, dest_dir: Path) -> Path:
    for download in payload.get("requested_downloads") or []:
        for key in ("filepath", "_filename", "filename"):
            if (value := download.get(key)) and Path(value).exists():
                return Path(value)
    for key in ("filepath", "_filename"):
        if (value := payload.get(key)) and Path(value).exists():
            return Path(value)
    media_id = payload.get("id")
    if media_id:
        matches = sorted(dest_dir.glob(f"{media_id}.*"), key=lambda p: p.stat().st_mtime)
        if matches:
            return matches[-1]
    raise DownloadError("The download finished but the media file could not be located.")


def _friendly_ytdlp_error(message: str) -> str:
    lowered = message.lower()
    if "private" in lowered or "sign in" in lowered or "login" in lowered:
        return "This media is private or requires a sign-in."
    if "unsupported url" in lowered:
        return "No downloader supports this URL."
    if "video unavailable" in lowered or "removed" in lowered:
        return "The media is unavailable or has been removed."
    if "geo" in lowered and "block" in lowered:
        return "This media is geo-blocked in your region."
    cleaned = re.sub(r"^ERROR:\s*", "", message.strip())
    return cleaned.splitlines()[0] if cleaned else "Download failed."


class _YTDLPLogger:
    """Route yt-dlp chatter into our logger instead of stdout."""

    def debug(self, message: str) -> None:
        if not message.startswith("[debug]"):
            log.debug("yt-dlp: %s", message)

    def info(self, message: str) -> None:  # pragma: no cover
        log.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        log.debug("yt-dlp warning: %s", message)

    def error(self, message: str) -> None:
        log.debug("yt-dlp error: %s", message)
