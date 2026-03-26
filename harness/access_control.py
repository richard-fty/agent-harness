"""Access control — tool allowlist/blocklist per session.

Controls which tools the agent is allowed to call. Used for:
  - Benchmarking: restrict tools to test agent behavior under constraints
  - Safety: block dangerous tools (run_command, write_file) in untrusted scenarios
  - Research: test how models handle denied tool calls
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.models import ToolCall


@dataclass
class AccessPolicy:
    """Defines what the agent is allowed to do."""

    # If set, ONLY these tools are allowed (whitelist mode)
    allowed_tools: set[str] | None = None

    # These tools are always blocked (blacklist mode)
    blocked_tools: set[str] = field(default_factory=set)

    # Max calls per tool (0 = unlimited)
    tool_call_limits: dict[str, int] = field(default_factory=dict)

    # Tools that require confirmation (for future TUI integration)
    confirm_tools: set[str] = field(default_factory=set)


# Preset policies
POLICY_UNRESTRICTED = AccessPolicy()

POLICY_READONLY = AccessPolicy(
    blocked_tools={"write_file", "edit_file", "run_command"},
)

POLICY_NO_SHELL = AccessPolicy(
    blocked_tools={"run_command"},
)

POLICY_TOOLS_ONLY = AccessPolicy(
    blocked_tools={"run_command", "write_file", "edit_file", "web_fetch", "web_search"},
)

PRESET_POLICIES: dict[str, AccessPolicy] = {
    "unrestricted": POLICY_UNRESTRICTED,
    "readonly": POLICY_READONLY,
    "no_shell": POLICY_NO_SHELL,
    "tools_only": POLICY_TOOLS_ONLY,
}


def get_policy(name: str) -> AccessPolicy:
    """Get a preset policy by name."""
    policy = PRESET_POLICIES.get(name)
    if policy is None:
        available = ", ".join(PRESET_POLICIES.keys())
        raise ValueError(f"Unknown policy: {name}. Available: {available}")
    return policy


@dataclass
class AccessController:
    """Enforces access control during an agent run."""

    policy: AccessPolicy
    call_counts: dict[str, int] = field(default_factory=dict)
    denied_calls: list[dict[str, Any]] = field(default_factory=list)

    def check(self, tool_call: ToolCall) -> str | None:
        """Check if a tool call is allowed.

        Returns None if allowed, or a denial reason string.
        """
        name = tool_call.name

        # Blocklist check
        if name in self.policy.blocked_tools:
            reason = f"Tool '{name}' is blocked by access policy"
            self.denied_calls.append({"tool": name, "reason": reason})
            return reason

        # Allowlist check
        if self.policy.allowed_tools is not None and name not in self.policy.allowed_tools:
            reason = f"Tool '{name}' is not in the allowed tools list"
            self.denied_calls.append({"tool": name, "reason": reason})
            return reason

        # Rate limit check
        if name in self.policy.tool_call_limits:
            limit = self.policy.tool_call_limits[name]
            current = self.call_counts.get(name, 0)
            if limit > 0 and current >= limit:
                reason = f"Tool '{name}' call limit reached ({limit})"
                self.denied_calls.append({"tool": name, "reason": reason})
                return reason

        # Allowed — increment counter
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        return None

    def summary(self) -> dict[str, Any]:
        """Return access control summary."""
        return {
            "total_calls": sum(self.call_counts.values()),
            "call_counts": dict(self.call_counts),
            "denied_count": len(self.denied_calls),
            "denied_calls": self.denied_calls,
        }
