"""Repo-wide pytest configuration.

Key guarantee: tests never write to the developer's real workspace paths.

Some tools default to relative filesystem paths such as ``results/artifacts``.
Without this fixture, tests that construct runtimes without explicitly
overriding paths would pollute the dev workspace.

We make every test run inside its own ``tmp_path`` by ``chdir``-ing into
it before the test body executes. Any relative paths then resolve under
the tmp dir and are cleaned up when pytest tears down the fixture.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
