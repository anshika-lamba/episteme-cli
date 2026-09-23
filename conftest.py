"""Session fixture: initialize the sandbox shell BEFORE any test touches TaskEnv.

test_cumulative.py calls TaskEnv.setup() directly and never calls tasks.set_shell().
On Windows that used to mean shell=True -> cmd.exe, so `touch` failed and looked like
a model failure. ensure_shell_for_direct_use() now requires a default WSL distro,
warms it, and checks python_test's nested quotes through the wrapper.
On POSIX this resolves to native /bin/sh and is a no-op.
"""
import pytest

import tasks


@pytest.fixture(autouse=True, scope="session")
def posix_sandbox_shell():
    tasks.set_shell(tasks.AUTO)  # don't inherit a stale prefix from an imported module
    try:
        tasks.ensure_shell_for_direct_use()
    except RuntimeError as e:
        pytest.exit(str(e), returncode=2)
