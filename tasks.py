import os, shutil, tempfile, subprocess
from dataclasses import dataclass
from typing import List, Optional

# Task scripts are POSIX shell (printf/touch/md5sum/tar/sed, `[ ! -f ]`, /dev/null).
# On POSIX, shell=True gives /bin/sh and everything works. On Windows, shell=True is
# cmd.exe and EVERY command "fails" with 'not recognized', which would poison the whole
# grid as model behaviour. set_shell() routes all task scripts through a real POSIX shell
# (Git-Bash `bash -lc`, or `wsl -e bash -lc`); run_grid.py resolves this automatically and
# refuses to start if no POSIX shell + coreutils are available.
_SHELL_PREFIX: Optional[List[str]] = None

def set_shell(prefix: Optional[List[str]]) -> None:
    global _SHELL_PREFIX
    _SHELL_PREFIX = list(prefix) if prefix else None

def shell_is_posix() -> bool:
    return os.name != "nt" or _SHELL_PREFIX is not None

@dataclass
class TaskDef:
    name: str
    setup_script: str
    prompt: str
    success_script: str
    expected_steps: int

TASKS = {
    "python_test": TaskDef(
        "python_test", 
        '''python3 -c 'open("test_app.py", "w").write("def test_calc():\\n    assert 1 == 2\\n")' ''', 
        "Fix the failing test in test_app.py so pytest passes.", 
        "pytest test_app.py > /dev/null 2>&1", 
        4
    ),
    "config_health": TaskDef(
        "config_health", 
        '''python3 -c 'open("health.py", "w").write("import json\\njson.load(open(\\"config.json\\"))\\n"); open("config.json", "w").write("bad_json\\n")' ''', 
        "health.py crashes because config.json is invalid. Fix config.json so health.py runs without error.", 
        "python3 health.py > /dev/null 2>&1", 
        4
    ),
    "checksum_build": TaskDef(
        "checksum_build", 
        '''printf 'v1\\n' > artifact.bin && printf 'wronghash  artifact.bin\\n' > hash.md5''', 
        "The artifact hash.md5 checksum is wrong. Fix the hash file so 'md5sum -c hash.md5' succeeds.", 
        "md5sum -c hash.md5 > /dev/null 2>&1", 
        4
    ),
    "log_rotation": TaskDef(
        "log_rotation", 
        "touch app1.log app2.log", 
        "Archive both log files into archive.tar.gz (tar -czf) and then delete the original .log files.", 
        "tar -tzf archive.tar.gz | grep -q app1.log && tar -tzf archive.tar.gz | grep -q app2.log && [ ! -f app1.log ] && [ ! -f app2.log ]", 
        5
    )
}

def scrub_secrets(text: str) -> str:
    if not text: return text
    for k, v in os.environ.items():
        if any(sec in k.upper() for sec in ["KEY", "TOKEN", "SECRET"]) and v:
            text = text.replace(v, "[REDACTED]")
    return text

class TaskEnv:
    def __init__(self, task: TaskDef):
        self.task = task
        self.dir = tempfile.mkdtemp()
        # Copy environment to keep Termux paths (LD_LIBRARY_PATH, PREFIX) intact
        self.env = os.environ.copy()
        self.env.update({"HOME": self.dir, "TERM": "dumb", "GIT_PAGER": "cat", "PAGER": "cat"})
        # Explicitly scrub API keys from the subprocess shell
        for k in list(self.env.keys()):
            if any(sec in k.upper() for sec in ["KEY", "TOKEN", "SECRET"]):
                del self.env[k]
        
    def _spawn(self, script: str, **kw):
        if _SHELL_PREFIX:
            return subprocess.run(_SHELL_PREFIX + [script], **kw)
        return subprocess.run(script, shell=True, **kw)

    def setup(self):
        return self._spawn(self.task.setup_script, cwd=self.dir, env=self.env, check=True,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        
    def run_cmd(self, cmd: str, timeout: int = 10) -> tuple:
        try:
            res = self._spawn(cmd, cwd=self.dir, env=self.env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            return scrub_secrets(res.stdout), scrub_secrets(res.stderr), res.returncode, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            return scrub_secrets(out), scrub_secrets(err), -124, True
            
    def check_success(self) -> bool:
        res = self._spawn(self.task.success_script, cwd=self.dir, env=self.env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return res.returncode == 0
        
    def cleanup(self): shutil.rmtree(self.dir, ignore_errors=True)
