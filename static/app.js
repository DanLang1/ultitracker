const state = { jobId: null, metadata: null, frame: null, strokes: [], drawing: false, pollToken: 0, heat: null, showHeat: true, analysisTimer: null, awaitingProcess: false };

const $ = (selector) => document.querySelector(selector);
const panels = [...document.querySelectorAll('.panel')];
const steps = [...document.querySelectorAll('.step')];
const canvas = $('#area-canvas');
const context = canvas.getContext('2d');
const outlineWorkspace = $('#outline-workspace');

function showStep(number) {
  panels.forEach((panel, index) => panel.classList.toggle('active', index === number - 1));
  steps.forEach((step, index) => {
    step.classList.toggle('active', index === number - 1);
    step.classList.toggle('done', index < number - 1);
    step.disabled = index >= number;
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60).toString().padStart(2, '0');
  return `${minutes}:${remaining}`;
}

function formatSize(bytes) { return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`; }

function uploadVideo(file) {
  if (!file) return;
  $('#upload-state').classList.remove('hidden');
  $('#upload-name').textContent = file.name;
  $('#upload-message').textContent = 'Copying video into the local workspace…';
  const data = new FormData();
  data.append('video', file);
  const request = new XMLHttpRequest();
  request.open('POST', '/api/upload');
  request.upload.onprogress = (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round(event.loaded / event.total * 100);
    $('#upload-percent').textContent = `${percent}%`;
    $('#upload-bar').style.width = `${percent}%`;
    if (percent === 100) $('#upload-message').textContent = 'Preparing a frame for the active-area outline…';
  };
  request.onload = () => {
    const response = JSON.parse(request.responseText || '{}');
    if (request.status >= 400) {
      $('#upload-message').textContent = response.error || 'Upload failed.';
      return;
    }
    selectJob(response);
    loadCanvasFrame(response.frame_url);
  };
  request.onerror = () => { $('#upload-message').textContent = 'Could not upload the video.'; };
  request.send(data);
}

function loadCanvasFrame(url) {
  const image = new Image();
  image.onload = () => {
    state.frame = image;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    state.strokes = state.metadata.points ? [state.metadata.points.map(point => ({ x: point.x * canvas.width, y: point.y * canvas.height }))] : [];
    updateOutlineActions();
    renderCanvas();
    showStep(2);
    applyAnalysis(state.metadata.analysis);
    watchAnalysis();
  };
  image.onerror = () => { $('#jobs-error').textContent = 'Could not load the saved video frame.'; };
  image.src = `${url}?v=${Date.now()}`;
}

function canvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * canvas.width / bounds.width,
    y: (event.clientY - bounds.top) * canvas.height / bounds.height,
  };
}

function renderCanvas() {
  if (!state.frame) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(state.frame, 0, 0, canvas.width, canvas.height);
  if (state.showHeat && state.heat) context.drawImage(state.heat, 0, 0, canvas.width, canvas.height);
  const points = state.strokes.flat();
  if (points.length < 2) return;
  context.save();
  context.strokeStyle = '#ff2d21';
  context.lineWidth = Math.max(7, canvas.width / 180);
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.shadowColor = 'rgba(0,0,0,.45)';
  context.shadowBlur = 2;
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
  if (!state.drawing && points.length >= 3) context.closePath();
  context.stroke();
  context.restore();
}

function updateOutlineActions() {
  const pointCount = state.strokes.reduce((total, stroke) => total + stroke.length, 0);
  $('#undo-outline').disabled = state.strokes.length === 0;
  $('#clear-outline').disabled = state.strokes.length === 0;
  $('#generate-preview').disabled = state.drawing || pointCount < 3;
}

function suggestionStroke() {
  const points = state.metadata && state.metadata.analysis && state.metadata.analysis.points;
  if (!points || points.length < 3) return null;
  return points.map(point => ({ x: point.x * canvas.width, y: point.y * canvas.height }));
}

function applyAnalysis(analysis) {
  const status = $('#analysis-status');
  const useButton = $('#use-suggestion');
  if (state.metadata) state.metadata.analysis = analysis;
  state.heat = null;
  useButton.disabled = true;
  $('#run-analysis').classList.toggle('hidden', !(!analysis || ['none', 'failed'].includes(analysis.state)));
  if (!analysis || analysis.state === 'none') {
    status.textContent = 'Player analysis has not run yet. Draw the outline by hand.';
  } else if (analysis.state === 'running') {
    status.textContent = `Analyzing players… ${analysis.progress || 0}%. A suggested outline will appear here; you can draw meanwhile.`;
  } else if (analysis.state === 'failed') {
    status.textContent = `Player analysis failed: ${analysis.message || 'see the job log'}. Draw the outline by hand.`;
  } else if (analysis.points) {
    status.textContent = `Suggested outline ready from ${analysis.minutes || '?'} minutes of play. Red shows where people usually stand; keep benches outside.`;
    useButton.disabled = false;
    if (state.strokes.length === 0) {
      state.strokes = [suggestionStroke()];
      updateOutlineActions();
    }
  } else {
    status.textContent = 'Recording too short for an outline suggestion. Draw the outline by hand.';
  }
  if (analysis && analysis.heat_url) {
    const image = new Image();
    image.onload = () => { state.heat = image; renderCanvas(); };
    image.src = `${analysis.heat_url}?v=${Date.now()}`;
  }
  renderCanvas();
}

function watchAnalysis() {
  clearTimeout(state.analysisTimer);
  const analysis = state.metadata && state.metadata.analysis;
  if (!analysis || analysis.state !== 'running') return;
  const jobId = state.jobId;
  state.analysisTimer = setTimeout(async () => {
    if (state.jobId !== jobId) return;
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      const job = await response.json();
      if (!response.ok || state.jobId !== jobId) return;
      state.metadata.awaiting_process = job.awaiting_process;
      if (job.analysis.state !== 'running' || job.analysis.progress !== analysis.progress) applyAnalysis(job.analysis);
      watchAnalysis();
    } catch (error) {
      watchAnalysis();
    }
  }, 2500);
}

$('#run-analysis').addEventListener('click', async () => {
  $('#run-analysis').disabled = true;
  $('#outline-error').textContent = '';
  try {
    const response = await fetch(`/api/analyze/${state.jobId}`, { method: 'POST' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not start the player analysis.');
    applyAnalysis({ state: 'running', progress: result.progress || 0 });
    watchAnalysis();
  } catch (error) {
    $('#outline-error').textContent = error.message;
  } finally {
    $('#run-analysis').disabled = false;
  }
});

$('#use-suggestion').addEventListener('click', () => {
  const stroke = suggestionStroke();
  if (!stroke) return;
  state.strokes = [stroke];
  updateOutlineActions();
  renderCanvas();
});

$('#heat-toggle').addEventListener('change', (event) => {
  state.showHeat = event.target.checked;
  renderCanvas();
});

canvas.addEventListener('pointerdown', (event) => {
  canvas.setPointerCapture(event.pointerId);
  state.strokes.push([canvasPoint(event)]);
  state.drawing = true;
  updateOutlineActions();
  $('#outline-error').textContent = '';
  renderCanvas();
});
canvas.addEventListener('pointermove', (event) => {
  if (!state.drawing) return;
  const point = canvasPoint(event);
  const activeStroke = state.strokes[state.strokes.length - 1];
  const previous = activeStroke[activeStroke.length - 1];
  if (Math.hypot(point.x - previous.x, point.y - previous.y) > 3) {
    activeStroke.push(point);
    renderCanvas();
  }
});
function finishStroke() {
  if (!state.drawing) return;
  state.drawing = false;
  updateOutlineActions();
  renderCanvas();
}

canvas.addEventListener('pointerup', finishStroke);
canvas.addEventListener('pointercancel', finishStroke);

$('#undo-outline').addEventListener('click', () => {
  state.strokes.pop();
  updateOutlineActions();
  renderCanvas();
});

$('#clear-outline').addEventListener('click', () => {
  state.strokes = [];
  updateOutlineActions();
  renderCanvas();
});

function setOutlineFullscreen(fullscreen) {
  outlineWorkspace.classList.toggle('fullscreen-mode', fullscreen);
  document.body.classList.toggle('outline-fullscreen-open', fullscreen);
  $('#fullscreen-outline').textContent = fullscreen ? 'Exit full screen' : 'Full screen';
  renderCanvas();
}

$('#fullscreen-outline').addEventListener('click', () => {
  setOutlineFullscreen(!outlineWorkspace.classList.contains('fullscreen-mode'));
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && outlineWorkspace.classList.contains('fullscreen-mode')) {
    setOutlineFullscreen(false);
  }
});

$('#generate-preview').addEventListener('click', async () => {
  $('#generate-preview').disabled = true;
  $('#outline-error').textContent = '';
  const points = state.strokes.flat().map((point) => ({
    x: point.x / canvas.width,
    y: point.y / canvas.height,
  }));
  try {
    const response = await fetch(`/api/active-area/${state.jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not generate the preview.');
    state.metadata.points = points;
    state.metadata.has_mask = true;
    $('#mask-preview').src = result.preview_url;
    setOutlineFullscreen(false);
    showStep(3);
  } catch (error) {
    $('#outline-error').textContent = error.message;
  } finally {
    updateOutlineActions();
  }
});

$('#adjust-outline').addEventListener('click', () => showStep(2));
$('#approve-preview').addEventListener('click', () => showStep(4));

$('#process-button').addEventListener('click', async () => {
  $('#process-error').textContent = '';
  $('#process-button').disabled = true;
  try {
    const response = await fetch(`/api/process/${state.jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        debug: $('#debug-toggle').checked,
        auto_widen: $('#auto-widen-toggle').checked,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not start processing.');
    state.awaitingProcess = result.mode === 'analyze';
    $('#process-hint').textContent = '';
    $('#process-ready').classList.add('hidden');
    $('#process-running').classList.remove('hidden');
    pollStatus(++state.pollToken);
  } catch (error) {
    $('#process-error').textContent = error.message;
  } finally {
    $('#process-button').disabled = false;
  }
});

async function pollStatus(token) {
  try {
    const response = await fetch(`/api/status/${state.jobId}`);
    const result = await response.json();
    if (token !== state.pollToken) return;
    if (!response.ok) throw new Error(result.error || 'Could not read job status.');
    const analyzing = result.mode === 'analyze';
    $('#process-bar').style.width = `${analyzing ? Math.round((result.progress || 0) / 4) : result.progress || 0}%`;
    $('#process-percent').textContent = `${analyzing ? Math.round((result.progress || 0) / 4) : result.progress || 0}%`;
    $('#process-message').textContent = analyzing ? `${result.message || 'Analyzing players…'} The video will be processed next.` : (result.message || 'Processing…');
    if (result.state === 'analyzed' && !state.awaitingProcess) {
      $('#process-running').classList.add('hidden');
      $('#process-ready').classList.remove('hidden');
      return;
    }
    if (result.state === 'complete') {
      $('#process-running').classList.add('hidden');
      $('#process-complete').classList.remove('hidden');
      $('#output-video').src = result.output_url;
      $('#output-download').href = `${result.output_url}?download=1`;
      if (result.debug_url) {
        $('#debug-download').href = `${result.debug_url}?download=1`;
        $('#debug-download').classList.remove('hidden');
      } else {
        $('#debug-download').removeAttribute('href');
        $('#debug-download').classList.add('hidden');
      }
      return;
    }
    if (['failed', 'interrupted'].includes(result.state)) throw new Error(result.error || result.message || 'Processing failed.');
    setTimeout(() => pollStatus(token), 1600);
  } catch (error) {
    if (token !== state.pollToken) return;
    $('#process-running').classList.add('hidden');
    $('#process-ready').classList.remove('hidden');
    $('#process-error').textContent = error.message;
  }
}

$('#video-input').addEventListener('change', (event) => uploadVideo(event.target.files[0]));

$('#delete-job').addEventListener('click', async () => {
  if (!state.jobId) return;
  const confirmed = window.confirm(
    'Delete this job’s uploaded source, mask, logs, and generated videos? Save the tactical video first.'
  );
  if (!confirmed) return;
  $('#delete-job').disabled = true;
  try {
    const response = await fetch(`/api/jobs/${state.jobId}`, { method: 'DELETE' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not delete the working files.');
    history.replaceState(null, '', '/');
    window.location.reload();
  } catch (error) {
    $('#delete-job').disabled = false;
    $('#process-error').textContent = error.message;
  }
});

const dropZone = $('#drop-zone');
['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => uploadVideo(event.dataTransfer.files[0]));


function selectJob(job) {
  state.pollToken += 1;
  state.jobId = job.job_id;
  state.metadata = job;
  history.replaceState(null, '', `?job=${job.job_id}`);
  $('#video-facts').textContent = `${job.width} × ${job.height} · ${formatDuration(job.duration)} · ${formatSize(job.size)}`;
  $('#process-filename').textContent = job.filename;
  $('#debug-toggle').checked = job.generate_debug || false;
  $('#auto-widen-toggle').checked = job.auto_widen !== false;
  $('#process-ready').classList.remove('hidden');
  $('#process-running').classList.add('hidden');
  $('#process-complete').classList.add('hidden');
  $('#process-error').textContent = '';
  state.awaitingProcess = Boolean(job.awaiting_process);
  const analysis = job.analysis || {};
  $('#process-hint').textContent = analysis.state === 'running'
    ? 'Player analysis is still running. Processing will start automatically when it finishes.'
    : '';
}

async function openJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Could not open the saved job.');
    selectJob(job);
    if (!job.has_mask) {
      loadCanvasFrame(job.frame_url);
    } else {
      showStep(4);
      const analyzingOnly = job.status.mode === 'analyze' && !job.awaiting_process;
      if (['queued', 'processing', 'complete'].includes(job.status.state) && !analyzingOnly) {
        $('#process-ready').classList.add('hidden');
        $('#process-running').classList.remove('hidden');
        pollStatus(state.pollToken);
      } else if (['failed', 'interrupted'].includes(job.status.state) && job.status.mode !== 'analyze') {
        $('#process-error').textContent = job.status.error || job.status.message;
      }
    }
  } catch (error) {
    $('#jobs-error').textContent = error.message;
  }
}

async function loadJobs() {
  try {
    const response = await fetch('/api/jobs');
    if (!response.ok) throw new Error('Could not load saved jobs.');
    const jobs = await response.json();
    const list = $('#saved-job-list');
    list.replaceChildren();
    jobs.forEach(job => {
      const button = document.createElement('button');
      button.className = 'button ghost';
      button.textContent = `${job.filename} · ${job.status.state}`;
      button.addEventListener('click', () => openJob(job.job_id));
      list.append(button);
    });
    $('#saved-jobs').classList.toggle('hidden', jobs.length === 0);
  } catch (error) {
    $('#jobs-error').textContent = error.message;
  }
}

$('#choose-another').addEventListener('click', () => {
  state.pollToken += 1;
  clearTimeout(state.analysisTimer);
  history.replaceState(null, '', '/');
  showStep(1);
  loadJobs();
});
$('#adjust-settings').addEventListener('click', () => {
  $('#process-complete').classList.add('hidden');
  $('#process-ready').classList.remove('hidden');
});
$('#edit-area').addEventListener('click', () => loadCanvasFrame(state.metadata.frame_url));
loadJobs();
const savedJobId = new URLSearchParams(window.location.search).get('job');
if (savedJobId) openJob(savedJobId);
