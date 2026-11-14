"""Exercise the installed runtime and service in an isolated installation."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("installation", type=Path)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    root = args.installation.resolve()
    python = root / "runtime/python.exe"
    launcher = root / "launcher.py"
    report = {}
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("PYTHON", "PCB_")) or key == "VIRTUAL_ENV":
            environment.pop(key)
    environment["PATH"] = str(root / "runtime") + os.pathsep + os.environ["SystemRoot"] + "\\System32"
    environment["PYTHONUTF8"] = "1"
    check = subprocess.run([str(python), "-c",
        "import sys,torch,cv2,ultralytics,albumentations,flask,waitress,reportlab; "
        "print(sys.executable);print(sys.prefix);print(torch.__version__);"
        "from ultralytics import YOLO;YOLO('training/yolo11n.pt');print('training-base-ok')"],
        cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8", check=True)
    report["bundled_runtime"] = check.stdout.strip()
    service = subprocess.Popen([str(python), str(launcher), "--no-browser", "--port", str(args.port)],
                               cwd=root, env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 180
        while True:
            if service.poll() is not None:
                raise RuntimeError((root / "data/desktop-logs/startup.log").read_text("utf-8"))
            try:
                state = json.loads((root / "data/desktop-session.json").read_text("utf-8"))
                base = f"http://127.0.0.1:{state['port']}"
                with urllib.request.urlopen(base + "/api/model/status", timeout=3) as response:
                    status = json.load(response)
                break
            except (OSError, ValueError):
                if time.monotonic() > deadline:
                    raise TimeoutError("Installed service did not become ready")
                time.sleep(1)
        report["model_status"] = status
        for route in ["/", "/platform/", "/training/", "/api/records", "/api/batches",
                      "/platform/api/projects"]:
            with urllib.request.urlopen(base + route, timeout=15) as response:
                assert response.status == 200, route
        report["pages_and_apis"] = "ok"
        with sqlite3.connect(root / "data/platform/platform.sqlite") as conn:
            for table, column in [("models", "weights"), ("images", "path"), ("alignment_sources", "path")]:
                paths = [Path(row[0]) for row in conn.execute(f"SELECT {column} FROM {table}") if row[0]]
                missing = [str(path) for path in paths if not path.is_file()]
                assert not missing, (table, missing[:5])
                assert all(path.is_relative_to(root) for path in paths), table
                report[f"relocated_{table}"] = len(paths)
            snapshots = [Path(json.loads(row[0])["path"]) for row in conn.execute(
                "SELECT metadata FROM dataset_snapshots") if json.loads(row[0]).get("path")]
            assert all(path.is_relative_to(root) and path.is_file() for path in snapshots), "Historical gallery paths"
            report["relocated_historical_gallery"] = len(snapshots)
        image = root / "training/missing_connection/images/train/board_missing.png"
        station = ""
        with sqlite3.connect(root / "data/platform/platform.sqlite") as conn:
            reference = conn.execute("SELECT s.path,p.station FROM alignment_sources s JOIN projects p ON p.id=s.project_id LIMIT 1").fetchone()
            if reference:
                image, station = Path(reference[0]), reference[1] or ""
        boundary = "PCBInstallerSmokeBoundary"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"preview_only\"\r\n\r\n1\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"station\"\r\n\r\n{station}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"sample.png\"\r\n"
                "Content-Type: image/png\r\n\r\n").encode() + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(base + "/api/preview", data=body,
                    headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        with urllib.request.urlopen(req, timeout=120) as response:
            detection = json.load(response)
        assert "boxes" in detection and "classification" in detection
        report["sample_detection"] = {key: detection[key] for key in ["model_version", "classification"]}
        duplicate = subprocess.run([str(python), str(launcher), "--no-browser"], cwd=root, env=environment,
                                   timeout=15, capture_output=True)
        assert duplicate.returncode == 0 and service.poll() is None
        report["duplicate_start"] = "ok"
    finally:
        subprocess.run([str(python), str(launcher), "--stop"], cwd=root, env=environment, timeout=30, check=True)
        service.wait(timeout=30)
    report["stop_service"] = "ok"
    output = root.parent / "installation-verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
