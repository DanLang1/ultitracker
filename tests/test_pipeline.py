from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from panstudio.artifacts import load_detections, load_plan, save_detections, source_identity
from panstudio.calibration import Calibration
from panstudio.cli import main
from panstudio.events import ProgressEvent
from panstudio.models import Detection, DetectionSample, DetectionSet, ProcessingConfig, VideoInfo
from panstudio.pipeline import detector_settings, plan_camera, validate_reuse


def detection_set(path, width=4608, height=1728):
    path.write_bytes(b'video')
    return DetectionSet(VideoInfo(width, height, 10, 1), source_identity(path), 0, 1, 2,
                        detector_settings(ProcessingConfig(path, 0, 1, sample_rate=2)),
                        tuple(DetectionSample(t, t, (Detection((2000, 1000, 2100, 1200), .9),)) for t in (0, .5)))


def test_raw_detections_round_trip_and_replanning_does_not_mutate(tmp_path):
    source = tmp_path / 'source.mp4'
    detections = detection_set(source)
    artifact = tmp_path / 'detections.json'
    save_detections(artifact, detections)
    loaded = load_detections(artifact)
    assert loaded == detections
    config = ProcessingConfig(source, 0, 1, sample_rate=2)
    before = asdict(loaded)
    first = plan_camera(config, loaded)
    mask = np.zeros((1728, 4608), dtype=np.uint8)
    second = plan_camera(config, loaded, mask)
    assert first.samples[0]['players']
    assert not second.samples[0]['players']
    assert plan_camera(config, loaded) == first
    assert asdict(loaded) == before


def test_artifact_rejects_changed_source_settings_and_schema(tmp_path):
    source = tmp_path / 'source.mp4'
    data = detection_set(source)
    config = ProcessingConfig(source, 0, 1, sample_rate=2)
    validate_reuse(config, data, data.video)
    for changed in (replace(config, end=.8), replace(config, sample_rate=3), replace(config, confidence=.5)):
        with pytest.raises(ValueError):
            validate_reuse(changed, data, data.video)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed source'):
        validate_reuse(config, data, data.video)
    artifact = tmp_path / 'bad.json'
    artifact.write_text('{"schema":"detections/v999"}')
    with pytest.raises(ValueError, match='schema'):
        load_detections(artifact)


def test_calibration_scales_framing_and_filtering_with_source(tmp_path):
    full = detection_set(tmp_path / 'full.mp4')
    half = replace(full, video=VideoInfo(2304, 864, 10, 1), samples=tuple(
        replace(sample, detections=tuple(Detection(tuple(v / 2 for v in d.box), d.confidence) for d in sample.detections))
        for sample in full.samples))
    config = ProcessingConfig(tmp_path / 'full.mp4', 0, 1)
    full_plan = plan_camera(config, full)
    half_plan = plan_camera(replace(config, crop_width=1280, crop_height=720), half)
    for left, right in zip(full_plan.samples, half_plan.samples):
        assert len(left['players']) == len(right['players']) == 1
        assert right['center_x'] == pytest.approx(left['center_x'] / 2)
        assert right['crop_width'] == pytest.approx(left['crop_width'] / 2)
        assert right['players'][0]['weight'] == left['players'][0]['weight']
    custom = tmp_path / 'calibration.json'
    custom.write_text('{"masked_minimum_player_height":12}')
    assert Calibration.load(custom).masked_minimum_player_height == 12
    with pytest.raises(ValueError):
        Calibration(reference_width=0)


def test_cli_reuses_detections_then_renders_saved_plan_without_inference(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'video.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=gray:size=192x108:rate=10:duration=1',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
    detections = DetectionSet(VideoInfo(192, 108, 10, 1), source_identity(source), 0, 1, 2,
                              detector_settings(ProcessingConfig(source, 0, 1, sample_rate=2)),
                              (DetectionSample(0, 0, ()), DetectionSample(.5, .5, ())))
    artifact = tmp_path / 'detections.json'
    save_detections(artifact, detections)
    monkeypatch.setattr('panstudio.cli.detect', lambda *args: pytest.fail('Inference must not run'))
    common = [str(source), '--start', '0', '--end', '1', '--sample-rate', '2', '--output-width', '128', '--output-height', '72']
    plan_dir, render_dir = tmp_path / 'planned', tmp_path / 'rendered'
    main([*common, '--crop-width', '128', '--crop-height', '72', '--detections-in', str(artifact),
          '--analysis-only', '--output-dir', str(plan_dir)])
    assert not (plan_dir / 'tactical-crop.mp4').exists()
    plan = load_plan(plan_dir / 'camera-plan.json')
    assert plan.samples[0]['crop_width'] == 128
    monkeypatch.setattr('panstudio.cli.plan_camera', lambda *args: pytest.fail('Planning must not run'))
    main([*common, '--plan-in', str(plan_dir / 'camera-plan.json'), '--output-dir', str(render_dir)])
    capture = cv2.VideoCapture(str(render_dir / 'tactical-crop.mp4'))
    try:
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 10
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 128
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 72
    finally:
        capture.release()
    events = [ProgressEvent.parse(line) for line in capsys.readouterr().out.splitlines()]
    assert any(event and event.stage == 'complete' for event in events)


def test_core_import_does_not_load_model_runtime():
    result = subprocess.run([sys.executable, '-c', "import panstudio.pipeline, sys; assert 'ultralytics' not in sys.modules"], capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('line', ['human log', '{}', '{"protocol":"tactical-pan/v1","stage":"render","completed":-1,"total":5}',
                                     '{"protocol":"tactical-pan/v1","stage":"render","completed":"1","total":5}'])
def test_progress_ignores_unrelated_and_invalid_events(line):
    assert ProgressEvent.parse(line) is None


def test_supervised_worker_exits_when_parent_pipe_closes():
    process = subprocess.Popen([sys.executable, '-c',
        "from panstudio.cli import watch_parent; import time; watch_parent(); print('ready', flush=True); time.sleep(60)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        process.stdin.close()
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def test_incomplete_detection_cache_is_rejected(tmp_path):
    data = detection_set(tmp_path / 'source.mp4')
    artifact = tmp_path / 'detections.json'
    save_detections(artifact, replace(data, end=60))
    with pytest.raises(ValueError, match='incomplete'):
        load_detections(artifact)
    invalid_sample = replace(data.samples[-1], time=.4, source_time=.4)
    save_detections(artifact, replace(data, samples=(data.samples[0], invalid_sample)))
    with pytest.raises(ValueError, match='sample rate'):
        load_detections(artifact)


def test_invalid_saved_crop_is_rejected(tmp_path):
    from panstudio.artifacts import save_plan
    data = detection_set(tmp_path / 'source.mp4')
    plan = plan_camera(ProcessingConfig(tmp_path / 'source.mp4', 0, 1), data)
    path = tmp_path / 'plan.json'
    save_plan(path, plan)
    invalid = json.loads(path.read_text())
    invalid['samples'][0]['crop_width'] = 9000
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='out-of-bounds'):
        load_plan(path)


def test_device_selection_prefers_available_gpu_and_respects_overrides(monkeypatch):
    import types
    import panstudio.pipeline as pipeline
    fake_torch = types.SimpleNamespace(backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    assert pipeline.select_device('auto') == 'mps'
    fake_torch.backends.mps.is_available = lambda: False
    assert pipeline.select_device('auto') == 'cpu'
    assert pipeline.select_device('cpu') == 'cpu'
    with pytest.raises(ValueError):
        ProcessingConfig(Path('video.mp4'), 0, 1, device='gpu')
    # The device is a speed choice, so it must not invalidate saved detections.
    assert detector_settings(ProcessingConfig(Path('video.mp4'), 0, 1, device='cpu')) == \
        detector_settings(ProcessingConfig(Path('video.mp4'), 0, 1, device='mps'))


def test_flights_artifact_round_trip_and_validation(tmp_path):
    from panstudio.artifacts import flights_version, load_flights, save_flights
    from panstudio.disc import DETECTOR_VERSION
    path = tmp_path / 'flights.json'
    save_flights(path, [{'start': 1.0, 'end': 2.5, 'x0': 100.0, 'y0': 900.0, 'x1': 1800.0, 'y1': 950.0, 'direction': 1, 'chord': 1700.7, 'frames': 30, 'speed': 56.0, 'points': []}], 0.0, 60.0, DETECTOR_VERSION)
    loaded = load_flights(path)
    assert loaded[0]['x1'] == 1800.0 and loaded[0]['direction'] == 1
    assert flights_version(path) == DETECTOR_VERSION
    path.write_text(json.dumps({'schema': 'flights/v1', 'flights': [{'start': 3, 'end': 1, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0, 'chord': 1}]}))
    with pytest.raises(ValueError):
        load_flights(path)


def test_unversioned_flights_still_load_but_count_as_stale(tmp_path):
    from panstudio.artifacts import flights_version, load_flights
    path = tmp_path / 'flights.json'
    path.write_text(json.dumps({'schema': 'flights/v1', 'start': 0, 'end': 60, 'flights': [
        {'start': 1, 'end': 2, 'x0': 0, 'y0': 0, 'x1': 300, 'y1': 0, 'chord': 300}]}))
    assert len(load_flights(path)) == 1
    assert flights_version(path) is None
    assert flights_version(tmp_path / 'missing.json') is None
    path.write_text('not json')
    assert flights_version(path) is None


def test_output_name_is_a_plain_file_name():
    from panstudio.models import ProcessingConfig
    assert ProcessingConfig(Path('game.mp4'), 0, 10).output_name == 'tactical-crop'
    assert ProcessingConfig(Path('game.mp4'), 0, 10, output_name='vs-dynasty.mp4').output_name == 'vs-dynasty'
    for bad in ('', '../escape', 'a/b', 'debug-wide', '.mp4'):
        with pytest.raises(ValueError):
            ProcessingConfig(Path('game.mp4'), 0, 10, output_name=bad)
