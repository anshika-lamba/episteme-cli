import os, shutil, tempfile, subprocess, ntpath
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Task scripts are POSIX shell (printf/touch/md5sum/tar/sed, `[ ! -f ]`, /dev/null).
# On POSIX, shell=True is /bin/sh and everything works. On Windows, shell=True is
# cmd.exe / PowerShell: `touch` is "not recognized", setup() raises CalledProcessError,
# and every trial looks like a model failure. So on Windows every task script is routed
# through a real POSIX shell, resolved automatically the first time TaskEnv is used
# (unit tests, REPLs, run_grid — none of them have to remember to call set_shell()).
#
# Git for Windows does NOT put bash on PATH (only cmd\git.exe). These two layouts are
# what the installer actually ships; both are checked explicitly, then bash next to
# git.exe, then WSL.
GIT_BASH_EXPLICIT = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
)
# -c, not -lc: a login shell can cd to $HOME and replace PATH, dropping python.
_SHELL_FLAG = "-c"

AUTO = "auto"
NO_WINDOWS_SHELL_MSG = (
    "no usable POSIX shell found for the sandbox. Windows PowerShell/cmd.exe cannot run the "
    "task scripts (`touch`, `md5sum`, `tar`, `/dev/null` are not commands there). Git for Windows "
    r"does not put bash on PATH; checked C:\Program Files\Git\bin\bash.exe and "
    r"C:\Program Files\Git\usr\bin\bash.exe, plus bash.exe next to git.exe, and WSL. "
    "Install Git for Windows or WSL, or pass --sandbox-shell <path to bash.exe>."
)
_SHELL_PREFIX = AUTO


def _is_windows() -> bool:
    return os.name == "nt"


def _probe_shell(prefix: List[str], timeout: int = 20) -> bool:
    """A candidate counts only if it actually executes a command (a WSL stub with no
    distro, or a renamed binary, must not)."""
    try:
        r = subprocess.run(prefix + ["echo __EPISTEME_SHELL_OK__"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode == 0 and "__EPISTEME_SHELL_OK__" in (r.stdout or "")
    except Exception:
        return False


def _bash_under(root: str) -> List[str]:
    """Join path *components*. Never unpack a path string — `join(root, *'bin\\bash.exe')`
    yields `C:\\b\\a\\s\\h\\.exe` and the real bash.exe is never found."""
    return [
        ntpath.normpath(ntpath.join(root, "bin", "bash.exe")),
        ntpath.normpath(ntpath.join(root, "usr", "bin", "bash.exe")),
    ]


def _git_roots(git_exe: str) -> List[str]:
    """Walk up from git.exe so both ...\\Git\\cmd\\git.exe and ...\\Git\\bin\\git.exe resolve
    to the install root (a fixed two-dirname climb misses one of those layouts)."""
    cur = ntpath.dirname(ntpath.abspath(git_exe))
    roots = []
    for _ in range(5):
        if not cur or cur in roots:
            break
        roots.append(cur)
        parent = ntpath.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return roots


def _candidate_shells(which: Optional[Callable[[str], Optional[str]]] = None,
                      exists: Optional[Callable[[str], bool]] = None,
                      environ: Optional[Dict[str, str]] = None) -> List[List[str]]:
    """Ordered argv prefixes. Path logic only (no probing) so it is testable anywhere.
    Git-for-Windows bash is checked BEFORE anything on PATH and before WSL: PowerShell
    typically has git.exe on PATH and bash.exe only at the two paths above, and a WSL
    stub can hang for tens of seconds."""
    which = shutil.which if which is None else which
    exists = os.path.isfile if exists is None else exists
    environ = os.environ if environ is None else environ
    cands: List[List[str]] = []
    seen = set()

    def add(argv: List[str]) -> None:
        key = tuple(ntpath.normcase(a) if i == 0 else a for i, a in enumerate(argv))
        if key not in seen:
            seen.add(key)
            cands.append(argv)

    def add_bash(path: str) -> None:
        if path and exists(path):
            add([path, _SHELL_FLAG])

    for path in GIT_BASH_EXPLICIT:
        add_bash(path)
    for extra in (r"C:\Program Files (x86)\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\usr\bin\bash.exe"):
        add_bash(extra)

    git = which("git") or which("git.exe")
    if git:
        for root in _git_roots(git):
            for path in _bash_under(root):
                add_bash(path)
    for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = environ.get(env_name)
        if not base:
            continue
        root = ntpath.join(base, "Programs", "Git") if env_name == "LOCALAPPDATA" else ntpath.join(base, "Git")
        for path in _bash_under(root):
            add_bash(path)

    for exe in ("bash", "bash.exe", "sh", "sh.exe"):
        found = which(exe)
        if found:
            add([found, _SHELL_FLAG])
    wsl = which("wsl") or which("wsl.exe")
    if wsl:
        add([wsl, "-e", "bash", _SHELL_FLAG])
    return cands


def find_posix_shell() -> Optional[List[str]]:
    """First candidate that executes a command; None on POSIX (native /bin/sh suffices)."""
    if not _is_windows():
        return None
    for pref in _candidate_shells():
        if _probe_shell(pref):
            return pref
    return None


def set_shell(prefix) -> None:
    """list -> use this shell prefix; None -> force native shell=True; AUTO -> re-detect."""
    global _SHELL_PREFIX
    if prefix is AUTO or prefix == AUTO:
        _SHELL_PREFIX = AUTO
    elif prefix:
        _SHELL_PREFIX = list(prefix)
    else:
        _SHELL_PREFIX = None


def ensure_shell_for_direct_use() -> Optional[List[str]]:
    """Resolve and cache the sandbox shell. Called from the pytest session fixture and from
    the first TaskEnv use, so tests that never call set_shell() still run `touch` under Git
    bash instead of PowerShell. None means native shell=True (POSIX). Raises RuntimeError
    on Windows when no bash can be found."""
    global _SHELL_PREFIX
    if _SHELL_PREFIX is AUTO or _SHELL_PREFIX == "notfound":
        found = find_posix_shell()
        if found is None and _is_windows():
            _SHELL_PREFIX = "notfound"
            raise RuntimeError(NO_WINDOWS_SHELL_MSG)
        _SHELL_PREFIX = found
    if _SHELL_PREFIX == "notfound":
        raise RuntimeError(NO_WINDOWS_SHELL_MSG)
    return _SHELL_PREFIX


def _active_shell() -> Optional[List[str]]:
    return ensure_shell_for_direct_use()


def current_shell() -> Optional[List[str]]:
    return None if _SHELL_PREFIX in (AUTO, "notfound") else _SHELL_PREFIX


def shell_is_posix() -> bool:
    return not _is_windows() or _SHELL_PREFIX not in (AUTO, "notfound", None)


def _posixize_script(script: str) -> str:
    """Inside Git bash, touch/md5sum/tar live in /usr/bin even though they are absent from
    the Windows PATH (and from PowerShell). Prepend that without dropping the inherited
    PATH — python.exe lives there, and Git Bash often has no `python3`."""
    if _is_windows():
        return 'export PATH="/usr/bin:/bin:${PATH:-}"; ' + script
    return script


@dataclass
class TaskDef:
    name: str
    setup_script: str
    prompt: str
    success_script: str
    expected_steps: int


# Setup/success scripts run BEFORE/AFTER the agent and are never shown to it, so they may
# be portable without touching the experiment: find any python (python3 | python | py —
# Git Bash and bare Windows installs rarely expose `python3`), and run pytest as a module.
_PYSHIM = (
    'PYBIN=""; for p in python3 python py; do command -v "$p" >/dev/null 2>&1 && { PYBIN="$p"; break; }; done; '
    '[ -n "$PYBIN" ] || { echo "no python" >&2; exit 127; }; '
)

TASKS = {
    "python_test": TaskDef(
        "python_test",
        _PYSHIM + '''"$PYBIN" -c 'open("test_app.py", "w").write("def test_calc():\\n    assert 1 == 2\\n")' ''',
        "Fix the failing test in test_app.py so pytest passes.",
        _PYSHIM + '"$PYBIN" -m pytest test_app.py > /dev/null 2>&1',
        4,
    ),
    "config_health": TaskDef(
        "config_health",
        _PYSHIM + '''"$PYBIN" -c 'open("health.py", "w").write("import json\\njson.load(open(\\"config.json\\"))\\n"); open("config.json", "w").write("bad_json\\n")' ''',
        "health.py crashes because config.json is invalid. Fix config.json so health.py runs without error.",
        _PYSHIM + '"$PYBIN" health.py > /dev/null 2>&1',
        4,
    ),
    "checksum_build": TaskDef(
        "checksum_build",
        '''printf 'v1\\n' > artifact.bin && printf 'wronghash  artifact.bin\\n' > hash.md5''',
        "The artifact hash.md5 checksum is wrong. Fix the hash file so 'md5sum -c hash.md5' succeeds.",
        "md5sum -c hash.md5 > /dev/null 2>&1",
        4,
    ),
    "log_rotation": TaskDef(
        "log_rotation",
        "touch app1.log app2.log",
        "Archive both log files into archive.tar.gz (tar -czf) and then delete the original .log files.",
        "tar -tzf archive.tar.gz | grep -q app1.log && tar -tzf archive.tar.gz | grep -q app2.log && [ ! -f app1.log ] && [ ! -f app2.log ]",
        5,
    ),
}


def scrub_secrets(text: str) -> str:
    if not text:
        return text
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
        # Git bash rewrites arguments that look like POSIX paths (/dev/null, /usr/bin) unless told not to.
        self.env.setdefault("MSYS_NO_PATHCONV", "1")
        self.env.setdefault("MSYS2_ARG_CONV_EXCL", "*")
        # Scrub secrets, and pytest's own process vars: python_test's success script *is*
        # `python -m pytest`, and a leaked PYTEST_CURRENT_TEST makes that nested run lie.
        for k in list(self.env.keys()):
            if any(sec in k.upper() for sec in ["KEY", "TOKEN", "SECRET"]) or k.startswith("PYTEST_"):
                del self.env[k]

    def _spawn(self, script: str, **kw):
        shell = _active_shell()  # resolves AUTO on first use; raises on Windows with no POSIX shell
        if shell:
            return subprocess.run(shell + [_posixize_script(script)], **kw)
        return subprocess.run(script, shell=True, **kw)

    def setup(self):
        return self._spawn(self.task.setup_script, cwd=self.dir, env=self.env, check=True,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")

    def run_cmd(self, cmd: str, timeout: int = 10) -> tuple:
        try:
            res = self._spawn(cmd, cwd=self.dir, env=self.env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
            return scrub_secrets(res.stdout), scrub_secrets(res.stderr), res.returncode, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            return scrub_secrets(out), scrub_secrets(err), -124, True

    def check_success(self) -> bool:
        res = self._spawn(self.task.success_script, cwd=self.dir, env=self.env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
        return res.returncode == 0

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)
