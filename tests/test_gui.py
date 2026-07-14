import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gui


class GuiJobControlTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.jobs_root = Path(self.temporary_directory.name)
        self.job_id = "a" * 32
        self.directory = self.jobs_root / self.job_id
        self.directory.mkdir()
        (self.directory / "active-area-mask.png").write_bytes(b"mask")
        (self.directory / "source.mp4").write_bytes(b"video")
        (self.directory / "job.json").write_text(
            json.dumps(
                {
                    "job_id": self.job_id,
                    "video_path": str(self.directory / "source.mp4"),
                    "duration": 10.0,
                }
            )
        )
        self.jobs_root_patch = patch.object(gui, "JOBS_ROOT", self.jobs_root)
        self.jobs_root_patch.start()
        with gui.status_lock:
            gui.job_status.clear()

    def tearDown(self):
        with gui.status_lock:
            gui.job_status.clear()
        self.jobs_root_patch.stop()
        self.temporary_directory.cleanup()

    def test_process_claims_job_before_starting_worker(self):
        starts = []

        class FakeThread:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def start(self):
                starts.append(self.kwargs)

        client = gui.app.test_client()
        with patch.object(gui.threading, "Thread", FakeThread):
            first = client.post(f"/api/process/{self.job_id}", json={"debug": False})
            second = client.post(f"/api/process/{self.job_id}", json={"debug": True})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(starts), 1)
        self.assertEqual(gui.job_status[self.job_id]["state"], "processing")
        self.assertFalse(gui.read_metadata(self.job_id)["generate_debug"])

    def test_delete_job_removes_working_directory(self):
        client = gui.app.test_client()
        response = client.delete(f"/api/jobs/{self.job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.directory.exists())

    def test_delete_job_refuses_active_processing(self):
        gui.update_status(self.job_id, state="processing")
        client = gui.app.test_client()
        response = client.delete(f"/api/jobs/{self.job_id}")

        self.assertEqual(response.status_code, 409)
        self.assertTrue(self.directory.exists())


class GuiWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "vssdynasty.mp4"
        if not source.exists():
            raise unittest.SkipTest("Local sample video is not available")
        cls.sample = ROOT / "prototype" / "gui-smoke-source.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", "21.8", "-t", "3", "-i", str(source),
                "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", str(cls.sample),
            ],
            check=True,
        )

    def test_full_local_workflow(self):
        client = gui.app.test_client()
        with self.sample.open("rb") as handle:
            response = client.post(
                "/api/upload",
                data={"video": (io.BytesIO(handle.read()), "sample.mp4")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200, response.text)
        upload = response.get_json()
        job_id = upload["job_id"]

        points = [
            {"x": 0.04, "y": 0.48},
            {"x": 0.96, "y": 0.48},
            {"x": 0.90, "y": 0.97},
            {"x": 0.10, "y": 0.97},
        ]
        response = client.post(f"/api/active-area/{job_id}", json={"points": points})
        self.assertEqual(response.status_code, 200, response.text)
        preview = response.get_json()
        self.assertIn("preview_url", preview)
        self.assertTrue((gui.job_directory(job_id) / "active-area-mask.png").exists())

        response = client.post(f"/api/process/{job_id}", json={"debug": False})
        self.assertEqual(response.status_code, 200, response.text)
        deadline = time.time() + 120
        status = {}
        while time.time() < deadline:
            status = client.get(f"/api/status/{job_id}").get_json()
            if status.get("state") in {"complete", "failed"}:
                break
            time.sleep(0.5)
        self.assertEqual(status.get("state"), "complete", json.dumps(status))

        output = gui.job_directory(job_id) / "output" / "tactical-crop.mp4"
        self.assertTrue(output.exists())
        self.assertFalse((gui.job_directory(job_id) / "output" / "debug-wide.mp4").exists())
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
