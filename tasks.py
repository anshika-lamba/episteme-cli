import os, shutil, sys, tempfile, subprocess, ntpath, shlex
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Task scripts are POSIX shell (printf/touch/md5sum/tar/sed, `[ ! -f ]`, /dev/null).
# On POSIX, shell=True is /bin/sh and everything works. On Windows, shell=True is
# cmd.exe: every Linux command fails, and that failure is the environment, not the model.
# Windows trials therefore run as `wsl --exec bash -c` (list form, shell=False), never
# cmd.exe. Git bash is not the sandbox: it is not a Linux userspace, and a WSL default
# user often has passwordless sudo, which would persist `apt-get install` across trials.
# Decision: sudo does not work in a trial. Commands run as the unprivileged user
# `episteme` (created once via `wsl -u root` if missing, removed from sudo/wheel).
# A distro name is never hardcoded; `wsl -l -v` must show a default (the line marked *).
GIT_BASH_EXPLICIT = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
)
# -c, not -lc: a login shell can cd to $HOME and replace PATH, dropping python.
_SHELL_FLAG = "-c"

AUTO = "auto"
NO_WINDOWS_SHELL_MSG = (
    "WSL is not ready, so the sandbox will not fall through to cmd.exe. "
    "Run `wsl -l -v` and confirm one distro is marked * (`wsl --set-default <name>` if not). "
    "A distro name is not hardcoded. The harness then runs `wsl --exec true` once, untimed, "
    "so cold-boot latency is not charged to the first trial."
)
# Unprivileged trial account. Not a distro name. Created once; trials never run as root
# and never as the default user if that user can passwordless-sudo.
TRIAL_USER = "episteme"
# Function, not a PATH shim: `/usr/bin/sudo` is blocked by running as TRIAL_USER, who is
# not in the sudo group. The function catches a bare `sudo` in the same bash -c.
SUDO_LOCK = (
    "sudo() { printf '%s\\n' 'episteme: sudo is disabled in the trial sandbox' >&2; return 127; }; "
)
_WSL_READY = False
_SHELL_PREFIX = AUTO
_WSL_EXIT_EXACT = True


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


def decode_wsl_output(raw: bytes) -> str:
    """`wsl -l -v` writes UTF-16 LE on Windows. Decoding that as UTF-8 hides the `*`."""
    if not raw:
        return ""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff") or (len(raw) > 3 and raw[1] == 0):
        return raw.decode("utf-16", errors="replace").replace("\x00", "")
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


def default_distro_name(listing: str) -> Optional[str]:
    """Name of the distro marked `*` in `wsl -l -v`. None if there is no default.
    The name is only used to fail fast; commands never pass `-d <name>`."""
    for line in (listing or "").splitlines():
        s = line.strip().lstrip("\ufeff")
        if s.startswith("*"):
            rest = s[1:].strip()
            return rest.split()[0] if rest else None
    return None


def record_exit_prefix(rc_file: str) -> str:
    """Bash prefix that writes the Linux $? even if the script calls `exit`.

    Used only when wsl.exe collapses a nonzero code (the laptop returned 1 for
    `exit 17`). The path is assigned first so a space in the username does not
    have to be quoted inside the trap string.
    """
    quoted = shlex.quote(rc_file)
    return (
        f"__rcfile={quoted}; trap '__rc=$?; printf \"%s\\n\" \"$__rc\" > \"$__rcfile\"' EXIT; "
    )


def wrap_trial_script(script: str, wsl_cwd: str) -> str:
    """`cd` into the trial dir, disable bare sudo, then the script exactly as written.

    Only the cwd path is shlex.quoted. The script is one argv element of
    `bash -c` (list form, shell=False). Quoting the whole string would
    double-escape nested quotes and break python_test's setup_script.
    HOME is set to that same path: the Windows env value is not a Linux home,
    and the trial directory path may contain spaces (wsl.exe's inherited cwd
    has failed on those).
    """
    if not wsl_cwd or not str(wsl_cwd).startswith("/"):
        raise RuntimeError(f"refusing to cd to a non-WSL path: {wsl_cwd!r}")
    quoted = shlex.quote(wsl_cwd)
    return f"cd {quoted} && export HOME={quoted} && {SUDO_LOCK}{script}"


def wsl_exec_argv(body: str, user: Optional[str] = TRIAL_USER) -> List[str]:
    """List-form argv. The body is not quoted again."""
    argv = ["wsl"]
    if user:
        argv.extend(["-u", user])
    argv.extend(["--exec", "bash", "-c", body])
    return argv


class _WslResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_wsl(argv: List[str], timeout: Optional[int] = None, env: Optional[Dict[str, str]] = None):
    """Bytes in, decoded text out. `wsl.exe` errors are sometimes UTF-16, like `wsl -l -v`."""
    kw = dict(capture_output=True, shell=False)
    if timeout is not None:
        kw["timeout"] = timeout
    if env is not None:
        kw["env"] = env
    r = subprocess.run(argv, **kw)
    return _WslResult(r.returncode, decode_wsl_output(r.stdout or b""), decode_wsl_output(r.stderr or b""))


def _exec_prefixes() -> List[List[str]]:
    """Argv prefixes ending in `-c`. No `-d` and no distro name.

    `--exec` / `-e` skip the default shell. Some builds treat a following `-c`
    as their own flag unless `--` stops parsing, and some distros have `/bin/sh`
    but not `bash`. The probe measures which form actually runs.
    """
    shells = ("bash", "/bin/bash", "sh", "/bin/sh")
    prefixes: List[List[str]] = []
    for sh in shells:
        prefixes.append(["wsl", "--exec", sh, "-c"])
        prefixes.append(["wsl", "-e", sh, "-c"])
        prefixes.append(["wsl", "--exec", "--", sh, "-c"])
    for sh in shells:
        prefixes.append(["wsl", "--", sh, "-c"])
        prefixes.append(["wsl", sh, "-c"])
    return prefixes


def _snippet(text: str, n: int = 220) -> str:
    return (text or "").replace("\r", " ").replace("\n", " ")[:n]


def _with_user(prefix: List[str], user: str) -> List[str]:
    return [prefix[0], "-u", user] + list(prefix[1:])


def select_wsl_exec_prefix() -> tuple:
    """-> (argv prefix ending in `-c`, exact_exit).

    exact_exit means `exit 17` came back as 17, so wsl.exe's code is the Linux
    code. Otherwise the script ran, `exit 0` stayed 0, and a nonzero code
    collapsed (this build returned 1 for 17). success_script only needs zero vs
    nonzero; the harness still records the Linux $? itself. A form that turns
    `exit 17` into 0 is not used.
    """
    attempts = []

    def ran(result, marker: str) -> bool:
        return marker in ((result.stdout or "") + "\n" + (result.stderr or ""))

    def consider(prefix: List[str], kind_exec: bool, found: dict) -> None:
        argv = prefix + ["echo EPISTEME_RAN; exit 17"]
        result = _run_wsl(argv)
        attempts.append((argv, result))
        if not ran(result, "EPISTEME_RAN"):
            return
        if result.returncode == 17:
            found.setdefault("exact_exec" if kind_exec else "exact_any", list(prefix))
            return
        if result.returncode == 0:
            return
        slot = "collapsed_exec" if kind_exec else "collapsed_any"
        if slot in found:
            return
        zero_argv = prefix + ["echo EPISTEME_ZERO; exit 0"]
        zero = _run_wsl(zero_argv)
        attempts.append((zero_argv, zero))
        if zero.returncode == 0 and ran(zero, "EPISTEME_ZERO"):
            found[slot] = list(prefix)

    bare = ["wsl", "--exec", "bash", "-c", "exit 17"]
    bare_result = _run_wsl(bare)
    attempts.append((bare, bare_result))
    if bare_result.returncode == 17:
        return ["wsl", "--exec", "bash", "-c"], True

    found: Dict[str, List[str]] = {}
    for prefix in _exec_prefixes():
        consider(prefix, "--exec" in prefix or "-e" in prefix, found)
        if "exact_exec" in found:
            return found["exact_exec"], True
    if "exact_any" in found:
        return found["exact_any"], True
    chosen = found.get("collapsed_exec") or found.get("collapsed_any")
    if chosen:
        return chosen, False

    lines = []
    for argv, result in attempts:
        lines.append(
            f"  rc={result.returncode} argv={argv!r} stdout={_snippet(result.stdout, 80)!r} stderr={_snippet(result.stderr, 160)!r}"
        )
    raise RuntimeError(
        "wsl did not run a deliberate exit 17 (no form printed EPISTEME_RAN, or exit 17 came back as 0). "
        "Refusing to trust success_script. First attempts:\n" + "\n".join(lines)
    )


_ENSURE_TRIAL_USER = r"""
set -eu
shell={shell}
if ! [ -x "$shell" ]; then shell=/bin/sh; fi
if ! id -u episteme >/dev/null 2>&1; then
  if command -v useradd >/dev/null 2>&1; then
    useradd --create-home --shell "$shell" --user-group episteme
  elif command -v adduser >/dev/null 2>&1; then
    adduser --disabled-password --gecos "" --shell "$shell" episteme 2>/dev/null || adduser -D -s "$shell" episteme
  else
    echo "no useradd or adduser" >&2
    exit 1
  fi
fi
for g in sudo wheel admin; do
  if id -nG episteme | tr ' ' '\n' | grep -qx "$g"; then
    gpasswd -d episteme "$g" >/dev/null 2>&1 || true
  fi
done
rm -f /etc/sudoers.d/episteme
passwd -l episteme >/dev/null 2>&1 || true
"""


def _login_shell(prefix: List[str]) -> str:
    """Shell token in an argv prefix that ends with `-c`."""
    if prefix and prefix[-1] == "-c" and len(prefix) >= 2:
        sh = prefix[-2]
        if sh in ("sh", "/bin/sh"):
            return "/bin/sh"
    return "/bin/bash"


def prepare_wsl_sandbox() -> List[str]:
    """Fail fast unless a default distro exists, a deliberate exit code is observable,
    and the trial user cannot sudo. Warm `wsl --exec true` once, with no timeout,
    before any trial. Returns the argv prefix ending in `-c` (TaskEnv appends the script).
    """
    global _WSL_READY, _SHELL_PREFIX, _WSL_EXIT_EXACT
    if _WSL_READY and isinstance(_SHELL_PREFIX, list) and _SHELL_PREFIX[:1] == ["wsl"]:
        return list(_SHELL_PREFIX)
    if not (shutil.which("wsl") or shutil.which("wsl.exe")):
        raise RuntimeError(NO_WINDOWS_SHELL_MSG + " `wsl.exe` was not found on PATH.")
    listed = subprocess.run(["wsl", "-l", "-v"], capture_output=True, shell=False)
    listing = decode_wsl_output(listed.stdout or b"")
    distro = default_distro_name(listing)
    if listed.returncode != 0 or not distro:
        detail = decode_wsl_output(listed.stderr or b"")[:300]
        raise RuntimeError(NO_WINDOWS_SHELL_MSG + f" wsl -l -v rc={listed.returncode} {detail!r}")
    # Untimed: a short timeout here would kill the cold boot and then blame the model.
    boot = _run_wsl(["wsl", "--exec", "true"])
    if boot.returncode != 0:
        raise RuntimeError(f"wsl --exec true failed (rc={boot.returncode}): {(boot.stderr or '')[:300]}")
    prefix, exact = select_wsl_exec_prefix()
    _WSL_EXIT_EXACT = exact
    if not exact:
        print(
            "[sandbox] wsl.exe did not return 17 for `exit 17`, but the script ran and `exit 0` stayed 0. "
            "success checks use zero vs nonzero; the Linux $? is recorded in the trial directory. "
            f"exec={' '.join(prefix)}",
            file=sys.stderr,
        )
    ensure = _ENSURE_TRIAL_USER.format(shell=_login_shell(prefix))
    created = _run_wsl(_with_user(prefix, "root") + [ensure])
    if created.returncode != 0:
        raise RuntimeError(
            "could not create unprivileged trial user 'episteme' via `wsl -u root`. "
            "Refusing to run as the default user: passwordless sudo would let a trial "
            f"`sudo apt-get install` and persist that change. stderr={(created.stderr or '')[:400]}"
        )
    for probe_cmd in ("sudo -n true", "/usr/bin/sudo -n true"):
        locked = _run_wsl(_with_user(prefix, TRIAL_USER) + [probe_cmd])
        if locked.returncode == 0:
            raise RuntimeError(
                f"{TRIAL_USER!r} can run {probe_cmd!r} without a password. "
                "Sudo is disabled for trials; refusing to start."
            )
    _WSL_READY = True
    return _with_user(prefix, TRIAL_USER)


def to_wsl_path(win_path: str) -> str:
    """Windows path -> /mnt/... via wslpath. Do not guess the mount point.

    Forward slashes avoid list2cmdline doubling backslashes inside quotes, which
    breaks paths that contain a space (`C:\\Users\\Aadit Lamba\\...`).
    """
    if str(win_path).startswith("/"):
        return win_path
    forwarded = win_path.replace("\\", "/")
    last = None
    for argv in (
        ["wsl", "--exec", "wslpath", "-u", forwarded],
        ["wsl", "--", "wslpath", "-u", forwarded],
        ["wsl", "wslpath", "-u", forwarded],
    ):
        last = _run_wsl(argv)
        lines = [ln.strip() for ln in (last.stdout or "").splitlines() if ln.strip().startswith("/")]
        if last.returncode == 0 and lines:
            return lines[-1]
    detail = "" if last is None else (last.stderr or "")[:300]
    raise RuntimeError(f"wslpath -u failed for {win_path!r}: {detail}")


def wsl_self_check() -> int:
    """Warm WSL, lock sudo, and run python_test's setup through the wrapper. Windows only."""
    prefix = prepare_wsl_sandbox()
    set_shell(prefix)
    verify_python_test_setup()
    print("wsl self-check OK")
    print(f"trial user: {TRIAL_USER} (sudo disabled; distro name is not hardcoded)")
    return 0


def verify_python_test_setup() -> None:
    """The setup_script has nested single and double quotes. Run it through the WSL
    wrapper and read the file back. A quoting bug creates no file, or the wrong file."""
    env = TaskEnv(TASKS["python_test"])
    try:
        env.setup()
        out, err, code, _timed = env.run_cmd("test -s test_app.py && cat test_app.py")
        if code != 0 or "def test_calc" not in out or "assert 1 == 2" not in out:
            raise RuntimeError(
                "python_test setup_script did not create test_app.py through the WSL wrapper "
                f"(rc={code}). Nested quotes were likely escaped. stdout={out!r} stderr={err!r}"
            )
    finally:
        env.cleanup()


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
    """Resolve and cache the sandbox shell. On Windows this is WSL, after the cold-boot
    warmup, the exit-code probe, the sudo lock, and the python_test quote check.
    None means native shell=True (POSIX). Raises RuntimeError if WSL is not ready."""
    global _SHELL_PREFIX
    if not _is_windows():
        if _SHELL_PREFIX is AUTO:
            _SHELL_PREFIX = None
        if _SHELL_PREFIX == "notfound":
            raise RuntimeError(NO_WINDOWS_SHELL_MSG)
        return None if _SHELL_PREFIX in (None, AUTO) else _SHELL_PREFIX
    if _SHELL_PREFIX is None:
        raise RuntimeError("native shell on Windows is cmd.exe; refusing. " + NO_WINDOWS_SHELL_MSG)
    if _SHELL_PREFIX == "notfound":
        raise RuntimeError(NO_WINDOWS_SHELL_MSG)
    if _SHELL_PREFIX is AUTO:
        try:
            _SHELL_PREFIX = prepare_wsl_sandbox()
            verify_python_test_setup()
        except Exception as e:
            _SHELL_PREFIX = "notfound"
            raise RuntimeError(str(e)) from e
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

    def _recorded_exit(self, wsl_code: int) -> int:
        """wsl.exe's code, or the Linux $? written beside the trial when codes collapse."""
        if _WSL_EXIT_EXACT:
            return wsl_code
        path = os.path.join(self.dir, ".episteme_rc")
        try:
            text = open(path, encoding="utf-8").read().strip()
            return int(text.splitlines()[0])
        except (OSError, ValueError, IndexError):
            return wsl_code

    def _spawn(self, script: str, **kw):
        shell = _active_shell()  # resolves AUTO on first use; raises on Windows with no POSIX shell
        if shell and os.path.basename(str(shell[0])).lower() in ("wsl", "wsl.exe"):
            kw.pop("cwd", None)  # wsl.exe mishandles a Windows cwd that contains spaces
            wsl_cwd = to_wsl_path(self.dir)
            body = wrap_trial_script(_posixize_script(script), wsl_cwd)
            if not _WSL_EXIT_EXACT:
                rc_path = os.path.join(self.dir, ".episteme_rc")
                try:
                    os.remove(rc_path)
                except OSError:
                    pass
                # EXIT trap, not a trailer: `exit` inside the script skips a trailer,
                # and that is when wsl.exe's collapsed 1 must not be what we record.
                body = record_exit_prefix(wsl_cwd + "/.episteme_rc") + body
            return subprocess.run(shell + [body], **kw)
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
            return scrub_secrets(res.stdout), scrub_secrets(res.stderr), self._recorded_exit(res.returncode), False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            return scrub_secrets(out), scrub_secrets(err), -124, True

    def check_success(self) -> bool:
        res = self._spawn(self.task.success_script, cwd=self.dir, env=self.env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
        return self._recorded_exit(res.returncode) == 0

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)
