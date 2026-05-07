"""Runner lifecycle — build and cache SharedTurnRunner instances per session.

This module is the only place the server layer constructs runtimes. Keeping it
separate from route models and ownership helpers makes the import graph cleaner
and makes it possible to swap the runner model later (e.g. to an Arq worker)
without touching route definitions.
"""

from __future__ import annotations

import os

from agent.policy.access_control import AccessController
from agent.policy.policy_models import get_policy
from agent.runtime.guards import RuntimeConfig
from agent.runtime.sandbox import create_session_sandbox
from agent.runtime.shared_runner import SharedTurnRunner
from agent.session.engine import SessionEngine

from apex_server.deps import AppState


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SERVER_MAX_STEPS = int(os.environ.get("APEX_SERVER_MAX_STEPS", "200"))
_SERVER_TIMEOUT_SECONDS = int(os.environ.get("APEX_SERVER_TIMEOUT_SECONDS", "600"))


def get_or_build_runner(state: AppState, session_id: str, model: str) -> SharedTurnRunner:
    """Return the live session runner, creating it lazily for MVP mode."""
    runner = state.runners.get(session_id)
    if runner is not None:
        _configure_server_runner(runner)
        return runner

    engine = SessionEngine(model=model, context_strategy="truncate")
    policy_name = os.environ.get("APEX_POLICY", "auto")
    access = AccessController(policy=get_policy(policy_name))
    _configure_access_roots(access)
    runner = SharedTurnRunner(
        session_engine=engine,
        access_controller=access,
        cost_tracker=None,
        model=model,
        runtime_config=RuntimeConfig(
            max_steps=_SERVER_MAX_STEPS,
            timeout_seconds=_SERVER_TIMEOUT_SECONDS,
        ),
        archive=state.archive,
        event_bus=state.event_bus,
        sandbox=create_session_sandbox(session_id=session_id, cwd=_REPO_ROOT),
        session_id=session_id,
    )
    runner.runtime.artifact_store = state.artifact_store
    state.runners[session_id] = runner
    state.runtimes[session_id] = runner.runtime
    return runner


def _configure_access_roots(access: AccessController) -> None:
    roots = (_REPO_ROOT,)
    access.policy.readable_roots = roots
    access.policy.writable_roots = roots


def _configure_server_runner(runner: SharedTurnRunner) -> None:
    _configure_access_roots(runner.access_controller)
    runner.runtime.runtime_config.max_steps = _SERVER_MAX_STEPS
    runner.runtime.runtime_config.timeout_seconds = _SERVER_TIMEOUT_SECONDS
