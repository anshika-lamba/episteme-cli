import os, shutil, tempfile, subprocess
from dataclasses import dataclass

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
        
    def setup(self): subprocess.run(self.task.setup_script, shell=True, cwd=self.dir, env=self.env, check=True)
        
    def run_cmd(self, cmd: str, timeout: int = 10) -> tuple:
        try:
            res = subprocess.run(cmd, shell=True, cwd=self.dir, env=self.env, capture_output=True, text=True, timeout=timeout)
            return scrub_secrets(res.stdout), scrub_secrets(res.stderr), res.returncode, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return scrub_secrets(out), scrub_secrets(err), -124, True
            
    def check_success(self) -> bool:
        res = subprocess.run(self.task.success_script, shell=True, cwd=self.dir, env=self.env)
        return res.returncode == 0
        
    def cleanup(self): shutil.rmtree(self.dir, ignore_errors=True)
