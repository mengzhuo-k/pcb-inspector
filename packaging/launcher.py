"""Installed Windows launcher. Uses the bundled, relocatable Python runtime."""
import argparse
import ctypes
import json
import logging
import os
from pathlib import Path
import runpy
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def relocate(data):
    marker = data / ".installed-root.json"
    previous = json.loads(marker.read_text("utf-8"))["root"] if marker.exists() else json.loads(
        (ROOT / "install-manifest.json").read_text("utf-8"))["source_root"]
    current = str(ROOT)
    if previous == current:
        return

    def replace(value):
        for old, new in [(previous, current),
                         (previous.replace("\\", "/"), current.replace("\\", "/")),
                         (json.dumps(previous)[1:-1], json.dumps(current)[1:-1]),
                         (json.dumps(previous, ensure_ascii=False)[1:-1],
                          json.dumps(current, ensure_ascii=False)[1:-1])]:
            value = value.replace(old, new)
        return value

    # Includes nested JSON strings (model cards, job configuration and snapshots).
    for database in data.rglob("*.sqlite"):
        with sqlite3.connect(database, timeout=60) as conn:
            for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                if table.startswith("sqlite_"):
                    continue
                quoted = '"' + table.replace('"', '""') + '"'
                for column in conn.execute(f"PRAGMA table_info({quoted})").fetchall():
                    col = '"' + column[1].replace('"', '""') + '"'
                    for rowid, value in conn.execute(
                            f"SELECT rowid,{col} FROM {quoted} WHERE typeof({col})='text'").fetchall():
                        updated = replace(value)
                        if updated != value:
                            conn.execute(f"UPDATE {quoted} SET {col}=? WHERE rowid=?", (updated, rowid))
    for path in data.rglob("*"):
        if path.suffix.lower() in {".json", ".yaml", ".yml"} and path.is_file():
            try:
                original = path.read_text("utf-8")
            except UnicodeError:
                continue
            updated = replace(original)
            if updated != original:
                path.write_text(updated, "utf-8")
    marker.write_text(json.dumps({"root": current}), "utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    state = data / "desktop-session.json"
    if args.stop:
        ctypes.windll.kernel32.OpenProcess.restype = ctypes.c_void_p
        ctypes.windll.kernel32.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
        ctypes.windll.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        if state.exists():
            session = json.loads(state.read_text("utf-8"))
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, session["pid"])
            if handle:
                try:
                    size = ctypes.c_ulong(32768)
                    buffer = ctypes.create_unicode_buffer(size.value)
                    ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
                    if ok and Path(buffer.value).resolve() == Path(sys.executable).resolve():
                        subprocess.run(["taskkill", "/PID", str(session["pid"]), "/T", "/F"],
                                       creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
            state.unlink(missing_ok=True)
        return

    # A separate mutex for each installation prevents duplicate background services.
    import hashlib
    mutex_name = "Local\\PCBInspector-" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:24]
    kernel = ctypes.windll.kernel32
    kernel.CreateMutexW.restype = ctypes.c_void_p
    mutex = kernel.CreateMutexW(None, False, mutex_name)
    if kernel.GetLastError() == 183:
        if not args.no_browser and state.exists():
            port = json.loads(state.read_text("utf-8"))["port"]
            webbrowser.open(f"http://127.0.0.1:{port}/")
        return
    if not mutex:
        raise ctypes.WinError()

    logs = data / "desktop-logs"
    logs.mkdir(exist_ok=True)
    log = open(logs / "startup.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    logging.basicConfig(stream=log, level=logging.INFO)
    try:
        relocate(data)
        # Use a free local port when the development copy is already running.
        port = args.port
        for candidate in range(port, port + 100):
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", candidate))
                except OSError:
                    continue
                port = candidate
                break
        else:
            raise RuntimeError("No available local port")
        os.chdir(ROOT)
        os.environ.update(PCB_DATA=str(data), PCB_TRAIN_DATA=str(data / "learning"),
                          PCB_PLATFORM_DATA=str(data / "platform"), PCB_HOST="127.0.0.1",
                          PCB_PORT=str(port), NO_ALBUMENTATIONS_UPDATE="1",
                          PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        sys.path.insert(0, str(ROOT))
        address = f"http://127.0.0.1:{port}/"
        state.write_text(json.dumps({"pid": os.getpid(), "port": port}), "utf-8")

        def open_when_ready():
            for _ in range(180):
                try:
                    with urllib.request.urlopen(address + "api/model/status", timeout=2) as response:
                        if response.status == 200:
                            webbrowser.open(address)
                            return
                except OSError:
                    time.sleep(1)

        if not args.no_browser:
            threading.Thread(target=open_when_ready, daemon=True).start()
        sys.argv = [str(ROOT / "app.py")]
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    except Exception:
        logging.exception("Startup failed")
        if not args.no_browser:
            ctypes.windll.user32.MessageBoxW(None, "启动失败，请查看：\n" + str(logs / "startup.log"),
                                             "焊视 PCB 检测", 0x10)
        raise
    finally:
        state.unlink(missing_ok=True)
        kernel.CloseHandle(ctypes.c_void_p(mutex))


if __name__ == "__main__":
    main()
