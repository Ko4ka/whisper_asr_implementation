import os
import sys
import time
import httpx
import subprocess
from dataclasses import dataclass
from typing import List, Optional


CHECK_INTERVAL_SECONDS = 10
HEALTH_TIMEOUT_SECONDS = 60
FAIL_THRESHOLD = 1
RESTART_COOLDOWN_SECONDS = 15
LOG_DIR = "logs"


@dataclass
class ServiceConfig:
    name: str
    cmd: List[str]
    health_url: Optional[str]
    cwd: str = "."


class ServiceState:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.fail_count = 0
        self.last_restart = 0.0

    def _log_paths(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        out_path = os.path.join(LOG_DIR, f"{self.cfg.name}.out.log")
        err_path = os.path.join(LOG_DIR, f"{self.cfg.name}.err.log")
        return out_path, err_path

    def start(self):
        out_path, err_path = self._log_paths()
        out_f = open(out_path, "a", encoding="utf-8", buffering=1)
        err_f = open(err_path, "a", encoding="utf-8", buffering=1)
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(
            self.cfg.cmd,
            cwd=self.cfg.cwd,
            stdout=out_f,
            stderr=err_f,
            creationflags=creationflags,
        )
        self.fail_count = 0
        self.last_restart = time.monotonic()
        print(f"[watchdog] started {self.cfg.name} pid={self.proc.pid}")

    def stop(self):
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        print(f"[watchdog] stopped {self.cfg.name}")

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def healthy(self, client: httpx.Client) -> bool:
        if not self.cfg.health_url:
            return True
        try:
            resp = client.get(self.cfg.health_url, timeout=HEALTH_TIMEOUT_SECONDS)
            return resp.status_code == 200
        except Exception:
            return False

    def should_restart(self) -> bool:
        return (time.monotonic() - self.last_restart) >= RESTART_COOLDOWN_SECONDS


def build_services() -> List[ServiceState]:
    python = sys.executable
    services = [
        ServiceConfig(
            name="service_asr",
            cmd=[python, "-m", "uvicorn", "service_asr_bulk_new:app", "--host", "127.0.0.1", "--port", "8000"],
            health_url="http://127.0.0.1:8000/health",
        ),
        ServiceConfig(
            name="service_diar",
            cmd=[python, "-m", "uvicorn", "service_diar_bulk_new_experiment:app", "--host", "127.0.0.1", "--port", "8001"],
            health_url="http://127.0.0.1:8001/health",
        ),
        ServiceConfig(
            name="service_align",
            cmd=[python, "-m", "uvicorn", "service_allignment:app", "--host", "127.0.0.1", "--port", "8002"],
            health_url="http://127.0.0.1:8002/health",
        ),
        ServiceConfig(
            name="frontman",
            cmd=[python, "-m", "uvicorn", "frontman:app", "--host", "127.0.0.1", "--port", "8005"],
            health_url="http://127.0.0.1:8005/health",
        ),
        ServiceConfig(
            name="asr_worker",
            cmd=[python, "asr_worker.py"],
            health_url=None,
        ),
    ]
    return [ServiceState(cfg) for cfg in services]


def main():
    services = build_services()
    for svc in services:
        svc.start()

    with httpx.Client() as client:
        while True:
            for svc in services:
                if not svc.is_running():
                    if svc.should_restart():
                        print(f"[watchdog] {svc.cfg.name} exited; restarting")
                        svc.start()
                    continue

                if svc.healthy(client):
                    svc.fail_count = 0
                else:
                    svc.fail_count += 1
                    print(f"[watchdog] {svc.cfg.name} health failed ({svc.fail_count})")
                    if svc.fail_count >= FAIL_THRESHOLD and svc.should_restart():
                        print(f"[watchdog] restarting {svc.cfg.name}")
                        svc.stop()
                        svc.start()
            time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

