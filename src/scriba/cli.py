"""Command line interface.

    scriba                       # launch the web UI
    scriba transcribe <url>      # one-shot transcription in the terminal
    scriba models                # what can run on this machine
    scriba doctor                # diagnose the setup
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import load_settings
from .errors import ScribaError
from .logging_setup import setup_logging
from .models import Job, JobStatus, OutputFormat, Task, TranscriptionOptions
from .util.text import format_duration

console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode="rich",
    help="Scriba — transcribe any audio or video URL locally with open ASR models.",
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"Scriba {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Run with no arguments to open the web interface."""
    if ctx.invoked_subcommand is None:
        # Call the implementation, not the Typer command: invoking a command
        # function directly hands it Typer's OptionInfo sentinels instead of
        # the defaults, and settings validation then fails on every field.
        _run_server()


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #
@app.command()
def serve(
    host: str = typer.Option(None, help="Interface to bind [default: 127.0.0.1]."),
    port: int = typer.Option(None, help="Port to listen on [default: 8420]."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Where transcripts are saved."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser window."),
    reload: bool = typer.Option(False, "--reload", hidden=True, help="Developer auto-reload."),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG, INFO, WARNING or ERROR."),
) -> None:
    """Launch the web interface (the default command)."""
    _run_server(
        host=host, port=port, output=output,
        no_browser=no_browser, reload=reload, log_level=log_level,
    )


def _run_server(
    host: str | None = None,
    port: int | None = None,
    output: Path | None = None,
    *,
    no_browser: bool = False,
    reload: bool = False,
    log_level: str = "INFO",
) -> None:
    """Start the web server. Shared by `scriba serve` and the bare `scriba`."""
    settings = load_settings(host=host, port=port, output_dir=output, log_level=log_level)
    if no_browser:
        settings.open_browser = False
    settings.ensure_dirs()
    setup_logging(settings.log_level)

    from .server.run import find_free_port

    requested = settings.port
    try:
        settings.port = find_free_port(settings.host, requested)
    except OSError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if settings.port != requested:
        console.print(
            f"[yellow]Port {requested} is already in use "
            f"(Scriba may already be running) - using {settings.port} instead.[/yellow]"
        )

    host_label = "localhost" if settings.host in ("0.0.0.0", "127.0.0.1") else settings.host
    url = f"http://{host_label}:{settings.port}"
    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold]Scriba[/bold] [dim]v{__version__}[/dim]\n\n"
                f"  Web UI      [link={url}]{url}[/link]\n"
                f"  Transcripts {settings.output_dir}\n"
                f"  Device      {_device_label()}\n\n"
                "[dim]Press Ctrl+C to stop.[/dim]"
            ),
            border_style="magenta",
            padding=(1, 3),
        )
    )
    # Import the module, not the name: a module attribute lookup at call time
    # keeps the indirection explicit and patchable in one canonical place.
    from .server import run as server_run

    try:
        server_run.run_server(settings, reload=reload)
    except KeyboardInterrupt:  # pragma: no cover
        console.print("\n[dim]Stopped.[/dim]")


# --------------------------------------------------------------------------- #
# transcribe
# --------------------------------------------------------------------------- #
@app.command()
def transcribe(
    url: str = typer.Argument(..., help="Media URL or local file path."),
    engine: str = typer.Option(None, "--engine", "-e", help="faster-whisper | parakeet | canary-qwen."),
    model: str = typer.Option(None, "--model", "-m", help="Checkpoint id (see 'scriba models')."),
    language: str = typer.Option(None, "--language", "-l", help="Language code, e.g. en. Omit to autodetect."),
    translate: bool = typer.Option(False, "--translate", help="Translate speech to English (Whisper only)."),
    formats: str = typer.Option("txt,srt,json", "--format", "-f", help="Comma-separated: txt,srt,vtt,json,md,tsv."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory."),
    device: str = typer.Option(None, "--device", help="cuda | cpu | auto."),
    compute_type: str = typer.Option(None, "--compute-type", help="float16 | int8_float16 | int8 | bfloat16."),
    beam_size: int = typer.Option(5, "--beam-size", help="Whisper beam width."),
    word_timestamps: bool = typer.Option(False, "--word-timestamps", help="Emit per-word timings."),
    no_vad: bool = typer.Option(False, "--no-vad", help="Disable the voice-activity filter."),
    prompt: str = typer.Option(None, "--prompt", help="Vocabulary hint (names, jargon, spellings)."),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Keep the decoded WAV next to the transcript."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the output paths."),
) -> None:
    """Transcribe one URL and write the transcript files."""
    setup_logging("ERROR" if quiet else "WARNING", quiet=quiet)

    try:
        chosen = [OutputFormat(f.strip().lower()) for f in formats.split(",") if f.strip()]
    except ValueError as exc:
        console.print(f"[red]Unknown output format:[/red] {exc}")
        raise typer.Exit(2) from exc

    settings = load_settings(output_dir=output, device=device, compute_type=compute_type)
    settings.keep_audio = keep_audio
    settings.ensure_dirs()

    options = TranscriptionOptions(
        engine=engine or settings.engine,
        model=model or settings.model,
        language=language or settings.language,
        task=Task.TRANSLATE if translate else Task.TRANSCRIBE,
        formats=chosen,
        output_dir=str(settings.output_dir),
        device=device or settings.device,
        compute_type=compute_type or settings.compute_type,
        beam_size=beam_size,
        vad_filter=not no_vad,
        word_timestamps=word_timestamps,
        initial_prompt=prompt,
    )

    from .pipeline import transcribe_url

    job: Job
    if quiet:
        job = transcribe_url(url, settings, options)
    else:
        with Progress(
            SpinnerColumn(style="magenta"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=32, complete_style="magenta", finished_style="green"),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Starting…", total=100)

            def emit(current: Job) -> None:
                progress.update(
                    task_id,
                    completed=round(current.progress * 100, 1),
                    description=current.message[:58],
                )

            job = transcribe_url(url, settings, options, emit=emit)

    if job.status is not JobStatus.COMPLETED:
        console.print(f"[bold red]✗[/bold red] {job.message}")
        if job.error and job.error.get("hint"):
            console.print(f"  [dim]{job.error['hint']}[/dim]")
        raise typer.Exit(1)

    if quiet:
        for path in job.outputs.values():
            print(path)
        return

    result = job.result
    assert result is not None
    console.print()
    console.print(f"[bold green]✓[/bold green] [bold]{job.media.title if job.media else url}[/bold]")
    console.print(
        f"  [dim]{format_duration(result.duration)} of audio · {result.word_count:,} words · "
        f"{result.model} on {result.device} · {result.speed_factor:.0f}× realtime[/dim]"
    )
    console.print()
    for fmt, path in job.outputs.items():
        console.print(f"  [magenta]{fmt:>4}[/magenta]  {path}")
    console.print()


# --------------------------------------------------------------------------- #
# models / doctor / history
# --------------------------------------------------------------------------- #
@app.command()
def models() -> None:
    """List the engines and checkpoints available on this machine."""
    from .engines import catalogue

    for engine in catalogue():
        status = "[green]installed[/green]" if engine["available"] else "[yellow]not installed[/yellow]"
        console.print(f"\n[bold]{engine['label']}[/bold] [dim]({engine['key']})[/dim]  {status}")
        console.print(f"  [dim]{engine['tagline']}[/dim]")
        if not engine["available"] and engine.get("hint"):
            console.print(f"  [cyan]{escape(engine['hint'])}[/cyan]")

        table = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
        table.add_column("model", style="magenta")
        table.add_column("WER", justify="right")
        table.add_column("VRAM", justify="right")
        table.add_column("languages")
        table.add_column("notes", style="dim", overflow="fold")
        for model in engine["models"]:
            table.add_row(
                model["id"] + (" *" if model["default"] else ""),
                f"{model['wer']}%" if model["wer"] else "—",
                f"{model['vram_gb']:g} GB",
                model["languages"],
                model["note"],
            )
        console.print(table)
    console.print("\n[dim]* default checkpoint · WER: Open ASR Leaderboard average, lower is better[/dim]\n")


@app.command()
def doctor() -> None:
    """Check that everything Scriba needs is in place."""
    from .engines import catalogue, detect_gpu
    from .media.ffmpeg import ffmpeg_available, ffmpeg_path, ffmpeg_version

    settings = load_settings()
    settings.ensure_dirs()
    table = Table(title="Scriba environment", box=None, header_style="dim", pad_edge=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", style="dim", overflow="fold")

    ok = "[green]✓[/green]"
    warn = "[yellow]![/yellow]"
    bad = "[red]✗[/red]"

    table.add_row("Python", ok, sys.version.split()[0])
    table.add_row("Scriba", ok, __version__)

    gpu = detect_gpu()
    table.add_row(
        "GPU",
        ok if gpu.available else warn,
        gpu.label if gpu.available else "No CUDA device found — everything will run on the CPU",
    )

    if ffmpeg_available():
        table.add_row("ffmpeg", ok, f"{ffmpeg_version()}  ({ffmpeg_path()})")
    else:
        table.add_row("ffmpeg", bad, "Not found. Install ffmpeg or 'pip install imageio-ffmpeg'.")

    for engine in catalogue():
        table.add_row(
            engine["label"],
            ok if engine["available"] else warn,
            escape(engine["reason"] or f"default model: {engine['default_model']}"),
        )

    table.add_row("Output folder", ok, str(settings.output_dir))
    table.add_row("History database", ok, str(settings.db_path))
    console.print()
    console.print(table)
    console.print()


@app.command()
def history(limit: int = typer.Option(15, "--limit", "-n", help="How many jobs to show.")) -> None:
    """Show recent transcriptions."""
    from .store import HistoryStore

    settings = load_settings()
    settings.ensure_dirs()
    jobs = HistoryStore(settings.db_path).recent(limit)
    if not jobs:
        console.print("[dim]No transcriptions yet.[/dim]")
        return

    table = Table(box=None, header_style="dim", pad_edge=False)
    table.add_column("when", style="dim")
    table.add_column("title", overflow="ellipsis", max_width=42)
    table.add_column("model", style="magenta")
    table.add_column("length", justify="right")
    table.add_column("status")
    table.add_column("file", style="dim", overflow="ellipsis", max_width=44)

    import datetime as _dt

    for job in jobs:
        colour = {"completed": "green", "failed": "red", "cancelled": "yellow"}.get(job.status.value, "cyan")
        table.add_row(
            _dt.datetime.fromtimestamp(job.created_at).strftime("%b %d %H:%M"),
            (job.media.title if job.media else job.url),
            job.options.model or job.options.engine,
            format_duration(job.result.duration) if job.result else "—",
            f"[{colour}]{job.status.value}[/{colour}]",
            job.outputs.get("txt", next(iter(job.outputs.values()), "")),
        )
    console.print()
    console.print(table)
    console.print()


def _device_label() -> str:
    from .engines import detect_gpu

    gpu = detect_gpu()
    return gpu.label if gpu.available else "CPU (no CUDA GPU detected)"


def main() -> None:
    try:
        app()
    except ScribaError as exc:  # pragma: no cover - top-level safety net
        console.print(f"[red]{exc.message}[/red]")
        if exc.hint:
            console.print(f"[dim]{exc.hint}[/dim]")
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
