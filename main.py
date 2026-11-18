#!/usr/bin/env python3
"""Set up gates on an iPhone Continuity Camera frame, then start live tracking.

Example:
    python3 live_iphone_easy_workflow.py --camera 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import yaml

from easy_gate_workflow import Workflow as GateValidator


ROOT = Path(__file__).resolve().parent
EDITOR = ROOT / "outputs" / "easy-multi-gate-editor.html"


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1, help="iPhone Continuity Camera index")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tracker", default="traffic_bytetrack.yaml")
    parser.add_argument("--measure-people", action="store_true", help="Measure pedestrians through gates for this run")
    return parser.parse_args()


class LiveSetup:
    def __init__(self) -> None:
        self.config_path: Path | None = None
        self.shutdown_callback = None
        self.close_browser_callback = None

    def start(self, text: str) -> tuple[bool, str]:
        try:
            config = yaml.safe_load(text)
            GateValidator.validate(config)
        except (yaml.YAMLError, TypeError, ValueError, AttributeError) as error:
            return False, f"Invalid gate setup: {error}"
        outputs = ROOT / "outputs"
        outputs.mkdir(exist_ok=True)
        self.config_path = outputs / "live-iphone-easy-gates.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        print("\nStart live tracker clicked — opening the live iPhone tracker…", flush=True)
        if self.close_browser_callback:
            threading.Timer(0.4, self.close_browser_callback).start()
        if self.shutdown_callback:
            threading.Timer(0.6, self.shutdown_callback).start()
        return True, "Live tracker is starting"


class CameraFeed:
    """One camera owner for the browser setup stream; released before YOLO opens it."""
    def __init__(self, camera_index: int) -> None:
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open iPhone camera. Close Photo Booth and allow Terminal under System Settings → Privacy & Security → Camera.")
        self.lock = threading.Lock()
        self.latest: bytes | None = None
        self.paused = False
        self.running = True
        self.stopped = False
        self.ready = threading.Event()
        self.worker = threading.Thread(target=self.read_frames, daemon=True)
        self.worker.start()

    def read_frames(self) -> None:
        while self.running:
            if self.paused:
                time.sleep(0.04); continue
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.05); continue
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
            if ok:
                with self.lock:
                    self.latest = encoded.tobytes()
                self.ready.set()

    def jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def stop(self) -> None:
        if self.stopped:
            return
        self.running = False
        # Never release an AVFoundation camera while another thread is inside
        # camera.read(); that can crash Python/macOS. Let the reader return,
        # join it, then release from this main workflow thread.
        self.worker.join(timeout=3)
        self.camera.release()
        self.stopped = True


def make_handler(setup: LiveSetup, feed: CameraFeed):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def reply_json(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                page = EDITOR.read_text(encoding="utf-8")
                info = {"imageUrl": "/stream.mjpg", "liveStream": True, "actionLabel": "Start live tracker"}
                data = page.replace("</head>", f"<script>window.GATE_WORKFLOW={json.dumps(info)};</script></head>").encode("utf-8")
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            elif path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
                try:
                    while feed.running and setup.config_path is None:
                        frame = feed.jpeg()
                        if frame:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                        time.sleep(0.035)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif path == "/api/status":
                self.reply_json(HTTPStatus.OK, {"state": "ready"})
            else:
                self.reply_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/pause":
                self.reply_json(HTTPStatus.OK, {"paused": feed.toggle_pause()}); return
            if path != "/api/start":
                self.reply_json(HTTPStatus.NOT_FOUND, {"error": "Not found"}); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size))
                ok, message = setup.start(str(body.get("config", "")))
                self.reply_json(HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST, {"ok": ok, "error": message})
            except (ValueError, json.JSONDecodeError) as error:
                self.reply_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
    return Handler


def main() -> None:
    args = options()
    if not EDITOR.is_file():
        raise FileNotFoundError(f"Cannot find editor: {EDITOR}")
    feed = CameraFeed(args.camera)
    if not feed.ready.wait(4):
        feed.stop()
        raise RuntimeError("The iPhone camera did not return a usable frame.")
    setup = LiveSetup()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(setup, feed))
    setup.shutdown_callback = server.shutdown
    url = f"http://127.0.0.1:{server.server_port}"

    def close_chrome_editor() -> None:
        script = f'''tell application "Google Chrome"
    repeat with browserWindow in windows
        repeat with browserTab in tabs of browserWindow
            if URL of browserTab is "{url}" then
                close browserTab
                return
            end if
        end repeat
    end repeat
end tell'''
        subprocess.run(["osascript", "-e", script], capture_output=True)

    setup.close_browser_callback = close_chrome_editor
    print(f"Opening live iPhone gate setup: {url}\nDraw gates/corridor, enter distances, then click Start live tracker.", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return
    finally:
        server.server_close()
        feed.stop()
    if setup.config_path is None:
        return
    command = [
        sys.executable, str(ROOT / "live_iphone_easy_tracker.py"), "--camera", str(args.camera), "--config", str(setup.config_path),
        "--csv", str(ROOT / "outputs" / "live-iphone-events.csv"), "--record", str(ROOT / "outputs" / "live-iphone-annotated.mp4"),
        "--model", args.model, "--confidence", str(args.confidence), "--imgsz", str(args.imgsz), "--tracker", args.tracker,
    ]
    if args.measure_people:
        command.append("--measure-people")
    subprocess.run(command, cwd=ROOT)


if __name__ == "__main__":
    main()
