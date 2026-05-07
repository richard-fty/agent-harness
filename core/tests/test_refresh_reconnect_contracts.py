from __future__ import annotations

import asyncio

import pytest

from agent.events import AssistantSnapshot, AssistantToken, InMemoryEventBus


@pytest.mark.asyncio
async def test_live_only_zero_seq_is_not_replayed_by_cursor() -> None:
    bus = InMemoryEventBus()
    token = AssistantToken(session_id="s1", turn_id="t1", text="live")
    snapshot = AssistantSnapshot(
        session_id="s1",
        turn_id="t1",
        seq=7,
        content="durable",
    )

    await bus.publish("s1", token)
    await bus.publish("s1", snapshot)

    assert token.seq == 0

    subscription = bus.subscribe("s1", since_seq=0).__aiter__()
    event = await asyncio.wait_for(subscription.__anext__(), timeout=1.0)

    assert event.type == "assistant_snapshot"
    assert event.seq == 7
