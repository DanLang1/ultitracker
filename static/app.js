const state = { jobId: null, metadata: null, frame: null, strokes: [], drawing: false, pollToken: 0 };

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

function formatClock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const pad = (value) => value.toString().padStart(2, '0');
  const hours = Math.floor(total / 3600), minutes = Math.floor((total % 3600) / 60);
  return hours ? `${hours}:${pad(minutes)}:${pad(total % 60)}` : `${minutes}:${pad(total % 60)}`;
}

const STAGE_NAMES = { mask: 'Mask', detection: 'Player analysis', disc: 'Disc scan', planning: 'Planning', debug: 'Review video', render: 'Render' };

function describeProgressTiming(status) {
  if (!status.started_at || !status.server_time) return '';
  let text = `Elapsed ${formatClock(status.server_time - status.started_at)}`;
  const fraction = status.stage_fraction || 0;
  const inStage = status.server_time - (status.stage_started_at || status.started_at);
  // Early in a stage the rate is dominated by startup cost, so wait before estimating.
  if (status.stage && fraction >= 0.03 && fraction < 1 && inStage >= 15) {
    text += ` · ${STAGE_NAMES[status.stage] || status.stage}: about ${formatClock(inStage * (1 - fraction) / fraction)} left`;
  }
  return text;
}

function describeTimings(timings) {
  if (!timings) return '';
  const stages = Object.entries(timings.stages)
    .filter(([, seconds]) => seconds >= 1)
    .map(([stage, seconds]) => `${STAGE_NAMES[stage] || stage} ${formatClock(seconds)}`);
  return `Processed in ${formatClock(timings.total_seconds)}` + (stages.length ? ` — ${stages.join(' · ')}` : '');
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
      body: JSON.stringify({ debug: $('#debug-toggle').checked, disc_tracking: $('#disc-toggle').checked }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not start processing.');
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
    $('#process-bar').style.width = `${result.progress || 0}%`;
    $('#process-percent').textContent = `${result.progress || 0}%`;
    $('#process-message').textContent = result.message || 'Processing…';
    $('#process-timing').textContent = describeProgressTiming(result);
    if (result.state === 'complete') {
      $('#process-timings').textContent = describeTimings(result.timings);
      $('#process-running').classList.add('hidden');
      $('#process-complete').classList.remove('hidden');
      $('#output-video').src = result.output_url;
      state.outputUrl = result.output_url;
      if (!$('#output-name').value) $('#output-name').value = defaultOutputName();
      updateOutputDownload();
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


function defaultOutputName() {
  const stem = (state.metadata?.filename || 'game').replace(/\.[^.]+$/, '');
  return `${stem}-tactical`;
}

function updateOutputDownload() {
  if (!state.outputUrl) return;
  const name = $('#output-name').value.trim().replace(/\.mp4$/i, '') || defaultOutputName();
  $('#output-download').href = `${state.outputUrl}?download=1&name=${encodeURIComponent(name)}.mp4`;
}

$('#output-name').addEventListener('input', updateOutputDownload);

function selectJob(job) {
  state.pollToken += 1;
  state.jobId = job.job_id;
  state.metadata = job;
  state.outputUrl = null;
  $('#output-name').value = '';
  history.replaceState(null, '', `?job=${job.job_id}`);
  $('#video-facts').textContent = `${job.width} × ${job.height} · ${formatDuration(job.duration)} · ${formatSize(job.size)}`;
  $('#process-filename').textContent = job.filename;
  $('#debug-toggle').checked = job.generate_debug || false;
  $('#disc-toggle').checked = job.disc_tracking !== false;
  $('#process-ready').classList.remove('hidden');
  $('#process-running').classList.add('hidden');
  $('#process-complete').classList.add('hidden');
  $('#process-error').textContent = '';
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
      if (['queued', 'processing', 'complete'].includes(job.status.state)) {
        $('#process-ready').classList.add('hidden');
        $('#process-running').classList.remove('hidden');
        pollStatus(state.pollToken);
      } else if (['failed', 'interrupted'].includes(job.status.state)) {
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
