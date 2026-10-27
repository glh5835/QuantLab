"""Windows-only smoke checks for the double-click QuantLab launcher.

Runs on GitHub-hosted Windows after requirements have been installed into
<repo>/.venv; never writes to the user's local computer.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "启动QuantLab.bat"


def check(condition: bool, description: str, details: str = "") -> None:
    if not condition:
        raise AssertionError(f"{description}\n{details}")
    print(f"PASS: {description}", flush=True)


def invoke_launcher(project: Path, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cmd.exe", "/d", "/c", str(project / LAUNCHER.name)],
        cwd=project,
        input="\r\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def negative_cases() -> None:
    with tempfile.TemporaryDirectory(prefix="quantlab-launcher-test-") as temp:
        root = Path(temp)
        shutil.copy2(LAUNCHER, root / LAUNCHER.name)

        missing = invoke_launcher(root)
        check(missing.returncode != 0 and "未找到项目虚拟环境" in missing.stdout,
              "Missing .venv produces a helpful error and nonzero exit",
              missing.stdout + missing.stderr)

        fake_python = root / ".venv" / "Scripts" / "python.exe"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.write_bytes(b"intentionally-invalid-executable")
        broken = invoke_launcher(root)
        check(broken.returncode != 0 and "Python 无法运行" in broken.stdout,
              "Broken virtualenv Python is rejected",
              broken.stdout + broken.stderr)
        fake_python.unlink()
        shutil.rmtree(root / ".venv")

        # Fresh isolated environment has pip but not QuantLab runtime dependencies.
        subprocess.run([sys.executable, "-m", "venv", str(root / ".venv")],
                       check=True, timeout=90)
        missing_deps = invoke_launcher(root)
        check(missing_deps.returncode != 0 and "Missing modules:" in missing_deps.stdout
              and "缺少依赖" in missing_deps.stdout,
              "Incomplete .venv is rejected before Streamlit starts",
              missing_deps.stdout + missing_deps.stderr)


def healthcheck(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except (OSError, URLError):
        return False


def success_case() -> None:
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    check(venv_python.is_file(), "Isolated .venv Python exists")
    with tempfile.TemporaryDirectory(prefix="quantlab-path-test-") as temp:
        # Put a deliberately failing global 'python' before the system PATH.
        # A working launcher must use the project .venv executable instead.
        Path(temp, "python.bat").write_text("@echo off\r\nexit /b 97\r\n", encoding="ascii")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = os.environ.copy()
        env["PATH"] = temp + os.pathsep + env.get("PATH", "")
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
        env["STREAMLIT_SERVER_PORT"] = str(port)

        # A logfile avoids blocked stdout pipes while the service is live.
        logfile = Path(temp) / "launcher.log"
        with logfile.open("wb") as out:
            proc = subprocess.Popen(
                ["cmd.exe", "/d", "/c", str(LAUNCHER)],
                cwd=temp,  # launcher must relocate to its own repository
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            try:
                health_url = f"http://127.0.0.1:{port}/_stcore/health"
                deadline = time.monotonic() + 100
                ready = False
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        break
                    if healthcheck(health_url):
                        ready = True
                        break
                    time.sleep(1)
                out.flush()
                details = logfile.read_text(encoding="utf-8", errors="replace")
                check(ready and proc.poll() is None,
                      "Double-click launcher starts a live Streamlit server from .venv",
                      f"exit_code={proc.poll()}\n{details[-8000:]}")
            finally:
                # Kill only the CI launcher subprocess and its descendants.
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False, timeout=20)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    if os.name != "nt":
        raise SystemExit("Windows runner required")
    check(LAUNCHER.is_file(), "Launcher batch file exists")
    negative_cases()
    success_case()
    print("WINDOWS LAUNCHER SMOKE TESTS PASSED", flush=True)
