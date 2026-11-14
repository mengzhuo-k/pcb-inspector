"""Build without duplicating the 1.4 GB Python environment on disk."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
BUILD = PACKAGING / "build"


def main():
    python_root = Path(sys.base_prefix)
    compiler = PACKAGING / "tools/InnoSetup/ISCC.exe"
    if not compiler.is_file():
        raise SystemExit("Install Inno Setup in packaging/tools/InnoSetup first.")
    BUILD.mkdir(exist_ok=True)
    # SQLite backup API produces consistent databases even if the developer server is running.
    databases = [ROOT / "data/records.sqlite", *sorted((ROOT / "data/learning").rglob("*.sqlite")),
                 *sorted((ROOT / "data/platform").rglob("*.sqlite"))]
    for source in databases:
        target = BUILD / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
            with closing(sqlite3.connect(target)) as dst:
                src.backup(dst)
                if dst.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise RuntimeError(f"Invalid database: {source}")
    (BUILD / "install-manifest.json").write_text(json.dumps({
        "source_root": str(ROOT), "version": "1.0.0", "python": sys.version,
        "databases": [str(p.relative_to(ROOT)) for p in databases],
    }, ensure_ascii=False, indent=2), "utf-8")
    subprocess.run([str(compiler), f"/DProjectRoot={ROOT}", f"/DPythonRoot={python_root}",
                    str(PACKAGING / "installer.iss")], check=True)


if __name__ == "__main__":
    main()
