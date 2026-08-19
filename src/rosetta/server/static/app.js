/* ==========================================================================
   RosettaAI — front-end
   Vanilla ES modules: no build step, no framework, no network beyond this app.
   ========================================================================== */

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const STORAGE_KEY = 'rosetta.prefs.v1';
const STAGES = [
  ['resolving',     'Resolve'],
  ['downloading',   'Download'],
  ['decoding',      'Decode'],
  ['loading_model', 'Load model'],
  ['transcribing',  'Transcribe'],
  ['writing',       'Write files'],
];
const FORMATS = [
  ['txt',  'Clean prose, grouped into paragraphs'],
  ['srt',  'Subtitles for video players and editors'],
  ['vtt',  'WebVTT captions for the web'],
  ['json', 'Segments, timings and metadata'],
  ['md',   'Markdown with timestamped sections'],
  ['tsv',  'Tab-separated rows for spreadsheets'],
];
const LANGUAGES = [
  ['', 'Detect language'], ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'],
  ['de', 'German'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'],
  ['pl', 'Polish'], ['ru', 'Russian'], ['uk', 'Ukrainian'], ['tr', 'Turkish'],
  ['ar', 'Arabic'], ['hi', 'Hindi'], ['zh', 'Chinese'], ['ja', 'Japanese'],
  ['ko', 'Korean'], ['sv', 'Swedish'], ['no', 'Norwegian'], ['da', 'Danish'],
  ['fi', 'Finnish'], ['cs', 'Czech'], ['el', 'Greek'], ['he', 'Hebrew'],
  ['id', 'Indonesian'], ['vi', 'Vietnamese'], ['ro', 'Romanian'], ['hu', 'Hungarian'],
];

/* ------------------------------------------------------------------ state */
const state = {
  jobs: new Map(),
  activeId: null,
  engines: [],
  health: null,
  view: 'segments',
  search: '',
  matchIndex: 0,
  wasLive: null,
  prefs: {
    engine: 'faster-whisper',
    model: null,
    language: '',
    task: 'transcribe',
    formats: ['txt', 'srt', 'json'],
    vad_filter: true,
    word_timestamps: false,
    beam_size: 5,
    initial_prompt: '',
    output_dir: '',
    theme: null,
  },
};

function loadPrefs() {
  try {
    Object.assign(state.prefs, JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'));
  } catch { /* corrupted prefs are not worth crashing over */ }
}
function savePrefs() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.prefs)); } catch {}
}

/* -------------------------------------------------------------------- api */
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const isJson = (response.headers.get('content-type') || '').includes('json');
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = payload?.detail ?? payload;
    throw Object.assign(new Error(detail?.message || detail || response.statusText), {
      hint: detail?.hint, status: response.status,
    });
  }
  return payload;
}

/* --------------------------------------------------------------- helpers */
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
const escapeHtml = (value = '') =>
  value.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function clock(seconds = 0) {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
           : `${m}:${String(s).padStart(2, '0')}`;
}
function humanDuration(seconds = 0) {
  if (!seconds) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}
function relativeTime(epoch) {
  const delta = (Date.now() - epoch * 1000) / 1000;
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 604800) return `${Math.floor(delta / 86400)}d ago`;
  return new Date(epoch * 1000).toLocaleDateString();
}
const jobTitle = (job) => job.media?.title || job.url.replace(/^https?:\/\/(www\.)?/, '').slice(0, 70);

/* ------------------------------------------------------------------ toast */
const ICONS = {
  success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m4 12.5 5 5L20 6.5"/></svg>',
  error:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.2v.1"/></svg>',
  info:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.8v.1"/></svg>',
};
function toast(title, body = '', kind = 'info', ms = 5200) {
  const el = document.createElement('div');
  el.className = `toast toast--${kind}`;
  el.innerHTML = `<span class="toast__icon">${ICONS[kind]}</span>
    <div class="toast__body"><strong>${escapeHtml(title)}</strong>${body ? `<p>${escapeHtml(body)}</p>` : ''}</div>`;
  $('#toasts').append(el);
  setTimeout(() => {
    el.classList.add('is-out');
    setTimeout(() => el.remove(), 220);
  }, ms);
}

/* ------------------------------------------------------------------ theme */
function applyTheme(theme) {
  const resolved = theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  document.documentElement.dataset.theme = resolved;
  state.prefs.theme = theme;
  savePrefs();
}

/* =========================================================================
   Rendering
   ========================================================================= */
function renderHistory() {
  const list = $('#history-list');
  const jobs = [...state.jobs.values()].sort((a, b) => b.created_at - a.created_at);
  $('#history-empty').hidden = jobs.length > 0;

  list.innerHTML = jobs.map((job) => {
    const running = job.status === 'running';
    const pct = Math.round((job.progress || 0) * 100);
    const meta = running ? job.message
      : job.status === 'completed' ? `${humanDuration(job.result?.duration)} · ${relativeTime(job.created_at)}`
      : `${job.status} · ${relativeTime(job.created_at)}`;
    return `
      <li>
        <button class="hitem ${job.id === state.activeId ? 'is-active' : ''}" data-job="${job.id}">
          <span class="hitem__dot hitem__dot--${job.status}"></span>
          <span>
            <span class="hitem__title">${escapeHtml(jobTitle(job))}</span>
            <span class="hitem__meta">${escapeHtml(meta)}</span>
            ${running ? `<span class="hitem__bar"><i style="width:${pct}%"></i></span>` : ''}
          </span>
          <span class="hitem__kill" data-remove="${job.id}" title="${running ? 'Cancel' : 'Remove'}">×</span>
        </button>
      </li>`;
  }).join('');
}

function renderJob() {
  const job = state.jobs.get(state.activeId);
  const composer = $('#composer');
  const view = $('#job-view');
  if (!job) {
    composer.hidden = false;
    view.hidden = true;
    $('#topbar-title').textContent = 'Transcribe';
    return;
  }
  composer.hidden = true;
  view.hidden = false;
  $('#topbar-title').textContent = jobTitle(job);
  $('#job-title').textContent = jobTitle(job);

  /* meta line ---------------------------------------------------------- */
  const media = job.media || {};
  const result = job.result;
  const bits = [];
  if (media.uploader) bits.push(`<span>${escapeHtml(media.uploader)}</span>`);
  if (media.webpage_url?.startsWith('http')) {
    bits.push(`<a href="${escapeHtml(media.webpage_url)}" target="_blank" rel="noreferrer">source ↗</a>`);
  }
  if (media.extractor) bits.push(`<span class="badge">${escapeHtml(media.extractor)}</span>`);
  bits.push(`<span class="badge badge--accent">${escapeHtml(job.options.model || job.options.engine)}</span>`);
  if (job.options.device) bits.push(`<span class="badge">${escapeHtml(job.options.device.toUpperCase())}</span>`);
  if (result?.language) bits.push(`<span class="badge">${escapeHtml(result.language)}</span>`);
  if (result?.timing_precision === 'approximate') {
    bits.push(`<span class="badge badge--warn" title="This model returns text without timings; segment times are interpolated.">approx. timings</span>`);
  }
  $('#job-meta').innerHTML = bits.join('');

  /* tools -------------------------------------------------------------- */
  const tools = [];
  if (job.status === 'running' || job.status === 'queued') {
    tools.push(`<button class="btn btn--danger btn--sm" data-action="cancel">Cancel</button>`);
  } else {
    tools.push(`<button class="btn btn--ghost btn--sm" data-action="retry">Run again</button>`);
  }
  if (Object.keys(job.outputs || {}).length) {
    tools.push(`<button class="btn btn--ghost btn--sm" data-action="reveal">Show files</button>`);
  }
  tools.push(`<button class="btn btn--ghost btn--sm" data-action="new">New</button>`);
  $('#job-tools').innerHTML = tools.join('');

  /* progress ----------------------------------------------------------- */
  const card = $('#progress-card');
  const active = job.status === 'running' || job.status === 'queued';
  card.hidden = job.status === 'completed';
  card.classList.toggle('is-running', job.status === 'running');
  $('#progress-message').textContent = job.message || '';
  $('#progress-pct').textContent = `${Math.round((job.progress || 0) * 100)}%`;
  $('#progress-fill').style.width = `${clamp((job.progress || 0) * 100, 2, 100)}%`;

  const reached = STAGES.findIndex(([key]) => key === job.stage);
  $('#stages').innerHTML = STAGES.map(([key, label], index) => {
    const done = job.status === 'completed' || (reached > index) || (reached === -1 && job.stage === 'done');
    const now = active && reached === index;
    return `<li class="stage ${done ? 'is-done' : ''} ${now ? 'is-active' : ''}">
      <span class="stage__dot">✓</span>${label}</li>`;
  }).join('');

  /* error -------------------------------------------------------------- */
  const errorCard = $('#error-card');
  if (job.status === 'failed') {
    errorCard.hidden = false;
    $('#error-title').textContent = 'Transcription failed';
    $('#error-message').textContent = job.error?.message || job.message;
    const hint = job.error?.hint;
    $('#error-hint').textContent = hint || '';
    $('#error-hint').hidden = !hint;
  } else {
    errorCard.hidden = true;
  }

  /* stats -------------------------------------------------------------- */
  const stats = $('#stats');
  if (result && !result.partial) {
    const cards = [
      [humanDuration(result.duration), 'Audio length'],
      [result.word_count.toLocaleString(), 'Words'],
      [`${result.segments.length.toLocaleString()}`, 'Segments'],
      [`${Math.round(result.duration / Math.max(result.processing_seconds, 0.01))}×`, 'Realtime speed'],
      [humanDuration(result.processing_seconds), 'Compute time'],
    ];
    stats.hidden = false;
    stats.innerHTML = cards.map(([value, label]) =>
      `<div class="stat"><div class="stat__value">${escapeHtml(String(value))}</div><div class="stat__label">${label}</div></div>`).join('');
  } else {
    stats.hidden = true;
  }

  /* transcript --------------------------------------------------------- */
  const transcript = $('#transcript-card');
  transcript.hidden = !result;
  if (result) {
    const live = Boolean(result.partial);
    $('#live-badge').hidden = !live;
    $('#copy-btn').hidden = live;          // the live view holds only a tail
    $('#downloads').innerHTML = Object.keys(job.outputs || {}).sort()
      .map((fmt) => `<a class="dl" href="/api/jobs/${job.id}/download/${fmt}" download title="Download .${fmt}">${fmt}</a>`)
      .join('');
    renderTranscript(job);
    const body = $('#transcript-body');
    if (live) {
      body.scrollTop = body.scrollHeight;        // follow the newest text
      state.wasLive = job.id;
    } else if (state.wasLive === job.id) {
      body.scrollTop = 0;                        // finished: back to the start
      state.wasLive = null;
    }
  }
}

function renderTranscript(job) {
  const body = $('#transcript-body');
  const segments = job.result?.segments || [];
  const query = state.search.trim().toLowerCase();

  if (!segments.length) {
    body.innerHTML = job.result?.partial
      ? `<p class="empty-note">Listening…</p>`
      : `<p class="empty-note">The audio contained no recognisable speech.</p>`;
    $('#search-count').textContent = '';
    return;
  }

  let matches = 0;
  const highlight = (text) => {
    if (!query) return escapeHtml(text);
    const lower = text.toLowerCase();
    let out = '', from = 0, at;
    while ((at = lower.indexOf(query, from)) !== -1) {
      out += escapeHtml(text.slice(from, at)) + `<mark data-match="${matches++}">` +
             escapeHtml(text.slice(at, at + query.length)) + '</mark>';
      from = at + query.length;
    }
    return out + escapeHtml(text.slice(from));
  };

  if (state.view === 'reading') {
    const chunks = [];
    let buffer = [], length = 0;
    for (const segment of segments) {
      buffer.push(segment.text);
      length += segment.text.length;
      if (length > 620 && /[.!?…]["')\]]?$/.test(segment.text.trim())) { chunks.push(buffer.join(' ')); buffer = []; length = 0; }
    }
    if (buffer.length) chunks.push(buffer.join(' '));
    body.innerHTML = `<div class="reading">${chunks.map((p) => `<p>${highlight(p)}</p>`).join('')}</div>`;
  } else {
    body.innerHTML = segments.map((segment) => `
      <div class="seg" data-start="${segment.start}">
        <button class="seg__time" title="Copy timestamp">${clock(segment.start)}</button>
        <div class="seg__text">${segment.speaker ? `<span class="seg__speaker">${escapeHtml(segment.speaker)}</span>` : ''}${highlight(segment.text)}</div>
      </div>`).join('');
  }

  $('#search-count').textContent = query ? (matches ? `${clamp(state.matchIndex + 1, 1, matches)}/${matches}` : 'no matches') : '';
  focusMatch();
}

function focusMatch() {
  const marks = $$('#transcript-body mark');
  marks.forEach((mark) => mark.classList.remove('is-current'));
  if (!marks.length) return;
  const target = marks[clamp(state.matchIndex, 0, marks.length - 1)];
  target.classList.add('is-current');
  target.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

/* ------------------------------------------------------------- catalogue */
function renderEngines() {
  const selectedModel = state.prefs.model;
  $('#engine-list').innerHTML = state.engines.map((engine) => {
    const models = engine.models.map((model) => {
      const selected = engine.key === state.prefs.engine && (selectedModel || engine.default_model) === model.id;
      const facts = [
        model.wer ? `<span class="fact fact--wer">${model.wer}% WER</span>` : '',
        `<span class="fact fact--vram">${model.vram_gb} GB VRAM</span>`,
        `<span class="fact">${escapeHtml(model.parameters)}</span>`,
        `<span class="fact">${escapeHtml(model.languages)}</span>`,
      ].join('');
      return `
        <button class="model ${selected ? 'is-selected' : ''} ${engine.available ? '' : 'is-disabled'}"
                data-engine="${engine.key}" data-model="${escapeHtml(model.id)}" ${engine.available ? '' : 'disabled'}>
          <span class="model__check">✓</span>
          <div class="model__name">${escapeHtml(model.label)}</div>
          <div class="model__note">${escapeHtml(model.note)}</div>
          <div class="model__facts">${facts}</div>
          ${model.speed ? `<div class="model__note" style="margin-top:6px;opacity:.75">${escapeHtml(model.speed)}</div>` : ''}
        </button>`;
    }).join('');

    return `
      <section class="engine-group">
        <div class="engine-group__head">
          <h4>${escapeHtml(engine.label)}</h4>
          <span class="badge">${escapeHtml(engine.vendor)}</span>
          <span class="badge">${escapeHtml(engine.license)}</span>
          ${engine.available ? '<span class="badge badge--success">installed</span>'
                             : '<span class="badge badge--warn">not installed</span>'}
        </div>
        <p class="engine-group__note">${escapeHtml(engine.tagline)}</p>
        ${engine.available ? '' : `<div class="engine-group__install">${escapeHtml(engine.hint || '')}</div>`}
        <div class="model-grid">${models}</div>
      </section>`;
  }).join('');
}

function renderFormats() {
  $('#formats-list').innerHTML = FORMATS.map(([fmt, description]) => `
    <label class="format-row ${state.prefs.formats.includes(fmt) ? 'is-on' : ''}">
      <input type="checkbox" data-format="${fmt}" ${state.prefs.formats.includes(fmt) ? 'checked' : ''} />
      <span class="format-row__name">.${fmt}</span>
      <span class="format-row__desc">${description}</span>
    </label>`).join('');
}

function syncOptionChips() {
  const engine = state.engines.find((e) => e.key === state.prefs.engine);
  const model = engine?.models.find((m) => m.id === (state.prefs.model || engine.default_model));
  $('#engine-chip-label').textContent = model?.label || state.prefs.engine;
  $('#formats-chip-label').textContent = state.prefs.formats.join(' · ') || 'no files';
  $('#task-select').disabled = !(engine?.supports_translate ?? true);
  $('#words-check').disabled = !(engine?.supports_word_timestamps ?? true);
}

function renderHealth(health) {
  state.health = health;
  const gpu = health.gpu || {};
  const device = $('#chip-device');
  device.className = `chip ${gpu.available ? 'is-ok' : 'is-warn'}`;
  device.lastElementChild.textContent = gpu.label || 'CPU only';
  device.title = gpu.available ? `CUDA GPU detected: ${gpu.label}` : 'No CUDA GPU detected — transcription will run on the CPU';

  const ffmpeg = $('#chip-ffmpeg');
  ffmpeg.className = `chip ${health.ffmpeg?.available ? 'is-ok' : 'is-bad'}`;
  ffmpeg.lastElementChild.textContent = health.ffmpeg?.available ? `ffmpeg ${(health.ffmpeg.version || '').split('-')[0]}` : 'ffmpeg missing';

  $('#output-dir-label').textContent = health.output_dir;
  $('#output-dir-btn').title = `Transcripts are saved to ${health.output_dir} — click to change`;
  $('#version-label').textContent = `v${health.version} · local`;
  if (!state.prefs.output_dir) $('#outdir-input').placeholder = health.output_dir;
}

/* =========================================================================
   Actions
   ========================================================================= */
async function submitJob(event) {
  event?.preventDefault();
  const input = $('#url-input');
  const url = input.value.trim();
  if (!url) { input.focus(); return; }

  const button = $('#submit-btn');
  button.disabled = true;
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      body: {
        url,
        options: {
          engine: state.prefs.engine,
          model: state.prefs.model || undefined,
          language: state.prefs.language || null,
          task: state.prefs.task,
          formats: state.prefs.formats,
          output_dir: state.prefs.output_dir || null,
          vad_filter: state.prefs.vad_filter,
          word_timestamps: state.prefs.word_timestamps,
          beam_size: state.prefs.beam_size,
          initial_prompt: state.prefs.initial_prompt || null,
        },
      },
    });
    state.jobs.set(job.id, job);
    state.activeId = job.id;
    input.value = '';
    renderHistory();
    renderJob();
  } catch (error) {
    toast('Could not start', error.hint ? `${error.message} — ${error.hint}` : error.message, 'error', 8000);
  } finally {
    button.disabled = false;
  }
}

async function cancelJob(id) {
  try { await api(`/api/jobs/${id}/cancel`, { method: 'POST' }); }
  catch (error) { toast('Cancel failed', error.message, 'error'); }
}

async function retryJob(id) {
  try {
    const job = await api(`/api/jobs/${id}/retry`, { method: 'POST' });
    state.jobs.set(job.id, job);
    state.activeId = job.id;
    renderHistory(); renderJob();
  } catch (error) { toast('Retry failed', error.message, 'error'); }
}

async function removeJob(id) {
  const job = state.jobs.get(id);
  if (job && !['completed', 'failed', 'cancelled'].includes(job.status)) { await cancelJob(id); return; }
  try {
    await api(`/api/jobs/${id}`, { method: 'DELETE' });
    state.jobs.delete(id);
    if (state.activeId === id) state.activeId = null;
    renderHistory(); renderJob();
  } catch (error) { toast('Could not remove', error.message, 'error'); }
}

async function copyTranscript() {
  const job = state.jobs.get(state.activeId);
  if (!job) return;
  try {
    const text = job.result?.text ?? await api(`/api/jobs/${job.id}/text`);
    await navigator.clipboard.writeText(text);
    toast('Copied', `${text.split(/\s+/).length.toLocaleString()} words on the clipboard`, 'success', 2600);
  } catch { toast('Copy failed', 'Your browser blocked clipboard access', 'error'); }
}

async function revealFiles(id) {
  try {
    const { revealed } = await api(`/api/jobs/${id}/reveal`, { method: 'POST' });
    if (!revealed) toast('Could not open the folder', state.health?.output_dir || '', 'info');
  } catch (error) { toast('Could not open the folder', error.message, 'error'); }
}

function showJob(id) {
  state.activeId = id;
  state.search = '';
  $('#search-input').value = '';
  renderHistory();
  renderJob();
  $('#scroll').scrollTo({ top: 0, behavior: 'smooth' });
  document.getElementById('app').classList.remove('is-open');
}

function showComposer() {
  state.activeId = null;
  renderHistory();
  renderJob();
  $('#url-input').focus();
}

/* =========================================================================
   Live updates (SSE with backoff reconnect)
   ========================================================================= */
function connectEvents() {
  let retry = 0;
  const open = () => {
    const source = new EventSource('/api/events');

    source.addEventListener('snapshot', (event) => {
      const { jobs } = JSON.parse(event.data);
      state.jobs = new Map(jobs.map((job) => [job.id, job]));
      renderHistory();
      if (state.activeId) renderJob();
    });

    source.addEventListener('job', (event) => {
      const job = JSON.parse(event.data);
      const previous = state.jobs.get(job.id);
      state.jobs.set(job.id, job);
      renderHistory();
      if (state.activeId === job.id) renderJob();
      if (previous && previous.status !== job.status) {
        if (job.status === 'completed') {
          toast('Transcript ready', `${jobTitle(job)} · ${job.result?.word_count.toLocaleString() || 0} words`, 'success');
          refreshHealth();
        } else if (job.status === 'failed') {
          toast('Transcription failed', job.error?.message || job.message, 'error', 9000);
        }
      }
    });

    source.onopen = () => { retry = 0; };
    source.onerror = () => {
      source.close();
      retry = Math.min(retry + 1, 6);
      setTimeout(open, 500 * 2 ** retry);
    };
  };
  open();
}

async function refreshHealth() {
  try { renderHealth(await api('/api/health')); } catch { /* the SSE reconnect will retry */ }
}

/* =========================================================================
   Wiring
   ========================================================================= */
function openModal(id) { $(id).hidden = false; }
function closeModals() { $$('.modal').forEach((modal) => (modal.hidden = true)); }

function bindEvents() {
  $('#url-form').addEventListener('submit', submitJob);
  $('#new-job-btn').addEventListener('click', showComposer);
  $('#sidebar-open').addEventListener('click', () => $('#app').classList.add('is-open'));
  $('#sidebar-close').addEventListener('click', () => $('#app').classList.remove('is-open'));
  $('#theme-btn').addEventListener('click', () =>
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));

  /* history list (delegated) */
  $('#history-list').addEventListener('click', (event) => {
    const remove = event.target.closest('[data-remove]');
    if (remove) { event.stopPropagation(); removeJob(remove.dataset.remove); return; }
    const item = event.target.closest('[data-job]');
    if (item) showJob(item.dataset.job);
  });

  $('#clear-history-btn').addEventListener('click', async () => {
    if (!confirm('Remove all finished jobs from the library? Transcript files stay on disk.')) return;
    await api('/api/history', { method: 'DELETE' });
    [...state.jobs.values()].forEach((job) => {
      if (['completed', 'failed', 'cancelled'].includes(job.status)) state.jobs.delete(job.id);
    });
    if (!state.jobs.has(state.activeId)) state.activeId = null;
    renderHistory(); renderJob();
  });

  /* job tools (delegated) */
  $('#job-tools').addEventListener('click', (event) => {
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (!action || !state.activeId) return;
    ({ cancel: cancelJob, retry: retryJob, reveal: revealFiles, new: showComposer }[action])(state.activeId);
  });
  $('#error-retry').addEventListener('click', () => state.activeId && retryJob(state.activeId));
  $('#copy-btn').addEventListener('click', copyTranscript);

  /* transcript interactions */
  $('#transcript-body').addEventListener('click', (event) => {
    const time = event.target.closest('.seg__time');
    if (!time) return;
    navigator.clipboard?.writeText(time.textContent.trim());
    toast('Timestamp copied', time.textContent.trim(), 'success', 1800);
  });

  const search = $('#search-input');
  search.addEventListener('input', () => {
    state.search = search.value;
    state.matchIndex = 0;
    if (state.activeId) renderTranscript(state.jobs.get(state.activeId));
  });
  search.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    const total = $$('#transcript-body mark').length;
    if (!total) return;
    state.matchIndex = (state.matchIndex + (event.shiftKey ? -1 : 1) + total) % total;
    $('#search-count').textContent = `${state.matchIndex + 1}/${total}`;
    focusMatch();
  });

  $$('.segmented button').forEach((button) => {
    button.addEventListener('click', () => {
      $$('.segmented button').forEach((other) => other.classList.toggle('is-active', other === button));
      state.view = button.dataset.view;
      if (state.activeId) renderTranscript(state.jobs.get(state.activeId));
    });
  });

  /* options */
  $('#engine-chip').addEventListener('click', () => { renderEngines(); openModal('#engine-modal'); });
  $('#formats-chip').addEventListener('click', () => { renderFormats(); openModal('#formats-modal'); });
  $('#engine-list').addEventListener('click', (event) => {
    const model = event.target.closest('[data-model]');
    if (!model || model.disabled) return;
    state.prefs.engine = model.dataset.engine;
    state.prefs.model = model.dataset.model;
    const engine = state.engines.find((e) => e.key === state.prefs.engine);
    if (engine && !engine.supports_translate) { state.prefs.task = 'transcribe'; $('#task-select').value = 'transcribe'; }
    savePrefs(); renderEngines(); syncOptionChips(); closeModals();
  });
  $('#formats-list').addEventListener('change', (event) => {
    const box = event.target.closest('[data-format]');
    if (!box) return;
    const set = new Set(state.prefs.formats);
    box.checked ? set.add(box.dataset.format) : set.delete(box.dataset.format);
    state.prefs.formats = FORMATS.map(([f]) => f).filter((f) => set.has(f));
    if (!state.prefs.formats.length) { state.prefs.formats = ['txt']; renderFormats(); }
    box.closest('.format-row')?.classList.toggle('is-on', box.checked);
    savePrefs(); syncOptionChips();
  });

  $('#advanced-chip').addEventListener('click', (event) => {
    const panel = $('#advanced-panel');
    panel.hidden = !panel.hidden;
    event.currentTarget.setAttribute('aria-expanded', String(!panel.hidden));
  });

  $('#language-select').addEventListener('change', (e) => { state.prefs.language = e.target.value; savePrefs(); });
  $('#task-select').addEventListener('change', (e) => { state.prefs.task = e.target.value; savePrefs(); });
  $('#vad-check').addEventListener('change', (e) => { state.prefs.vad_filter = e.target.checked; savePrefs(); });
  $('#words-check').addEventListener('change', (e) => { state.prefs.word_timestamps = e.target.checked; savePrefs(); });
  $('#prompt-input').addEventListener('input', (e) => { state.prefs.initial_prompt = e.target.value; savePrefs(); });
  $('#outdir-input').addEventListener('change', (e) => { state.prefs.output_dir = e.target.value.trim(); savePrefs(); });
  $('#beam-range').addEventListener('input', (e) => {
    state.prefs.beam_size = Number(e.target.value);
    $('#beam-value').textContent = e.target.value;
    savePrefs();
  });

  $('#output-dir-btn').addEventListener('click', async () => {
    const next = prompt('Save transcripts to which folder?', state.health?.output_dir || '');
    if (!next) return;
    try {
      const payload = await api('/api/settings', { method: 'PATCH', body: { output_dir: next } });
      toast('Output folder updated', payload.output_dir, 'success');
      refreshHealth();
    } catch (error) { toast('Could not use that folder', error.message, 'error'); }
  });

  $$('.samples button').forEach((button) => button.addEventListener('click', () => {
    $('#url-input').value = button.dataset.url;
    $('#url-input').focus();
  }));

  $$('[data-close]').forEach((element) => element.addEventListener('click', closeModals));

  /* keyboard shortcuts */
  document.addEventListener('keydown', (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
    if (event.key === 'Escape') { closeModals(); return; }
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { submitJob(event); return; }
    if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === '/') { event.preventDefault(); ($('#transcript-card').hidden ? $('#url-input') : $('#search-input')).focus(); }
    else if (event.key.toLowerCase() === 'n') { event.preventDefault(); showComposer(); }
    else if (event.key.toLowerCase() === 'm') { event.preventDefault(); renderEngines(); openModal('#engine-modal'); }
    else if (event.key.toLowerCase() === 't') { applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); }
    else if (event.key.toLowerCase() === 'c' && state.activeId) { copyTranscript(); }
  });
}

function hydrateControls() {
  $('#language-select').innerHTML = LANGUAGES
    .map(([code, name]) => `<option value="${code}">${name}</option>`).join('');
  $('#language-select').value = state.prefs.language || '';
  $('#task-select').value = state.prefs.task;
  $('#vad-check').checked = state.prefs.vad_filter;
  $('#words-check').checked = state.prefs.word_timestamps;
  $('#beam-range').value = state.prefs.beam_size;
  $('#beam-value').textContent = state.prefs.beam_size;
  $('#prompt-input').value = state.prefs.initial_prompt || '';
  $('#outdir-input').value = state.prefs.output_dir || '';
}

/* ------------------------------------------------------------------- boot */
async function boot() {
  loadPrefs();
  applyTheme(state.prefs.theme);
  hydrateControls();
  bindEvents();
  connectEvents();

  try {
    const [{ engines }, health] = await Promise.all([api('/api/engines'), api('/api/health')]);
    state.engines = engines;
    if (!engines.some((e) => e.key === state.prefs.engine && e.available)) {
      const fallback = engines.find((e) => e.available) || engines[0];
      if (fallback) { state.prefs.engine = fallback.key; state.prefs.model = fallback.default_model; savePrefs(); }
    }
    renderHealth(health);
    syncOptionChips();
    if (!health.gpu?.available) {
      toast('No CUDA GPU detected', 'Transcription will run on the CPU — expect it to be much slower.', 'info', 9000);
    }
  } catch (error) {
    toast('Could not reach the RosettaAI service', error.message, 'error', 9000);
  }

  try {
    const { jobs } = await api('/api/jobs?limit=50');
    state.jobs = new Map(jobs.map((job) => [job.id, job]));
    renderHistory();
  } catch { /* the SSE snapshot will fill this in */ }

  $('#url-input').focus();
}

boot();
