"""Test package for shared harness imports."""

import atexit
import os
import shutil
import tempfile


TEST_STATE_ROOT = tempfile.mkdtemp(prefix="ysupport-test-state-")
os.environ["TICKET_EXECUTION_STATE_ROOT"] = TEST_STATE_ROOT
atexit.register(shutil.rmtree, TEST_STATE_ROOT, ignore_errors=True)
