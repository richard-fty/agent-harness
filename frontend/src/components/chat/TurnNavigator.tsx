import { useEffect, useMemo, useState } from "react";
import { useStore } from "../../store";
import { getJSON } from "../../lib/api";
import type { AgentEvent, TurnSummary } from "../../types";

export function TurnNavigator({ sessionId }: { sessionId: string }) {
  const turns = useStore((s) => s.turnsBySession[sessionId] ?? []);
  const itemCount = useStore((s) => s.sessions[sessionId]?.items.length ?? 0);
  const setTurns = useStore((s) => s.setTurns);
  const replaceSessionEvents = useStore((s) => s.replaceSessionEvents);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [hoveredTurnId, setHoveredTurnId] = useState<string | null>(null);
  const [loadingHistoryTurnId, setLoadingHistoryTurnId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rows = await getJSON<TurnSummary[]>(
          `/sessions/${encodeURIComponent(sessionId)}/turns`,
        );
        if (!cancelled) setTurns(sessionId, rows);
      } catch (err) {
        console.warn("Failed to load turn summaries", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId, setTurns]);

  useEffect(() => {
    const root = getScrollRoot(sessionId);
    if (!root) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      const next = findActiveTurnId(root);
      if (next) setActiveTurnId(next);
    };
    const schedule = () => {
      if (frame) return;
      frame = requestAnimationFrame(update);
    };
    update();
    root.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      if (frame) cancelAnimationFrame(frame);
      root.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [sessionId, itemCount, turns.length]);

  const orderedTurns = useMemo(
    () => [...turns].sort((a, b) => a.started_seq - b.started_seq),
    [turns],
  );
  const hoveredTurn = orderedTurns.find((turn) => turn.turn_id === hoveredTurnId) ?? null;

  if (orderedTurns.length <= 1) return null;

  return (
    <nav
      aria-label="Turn navigator"
      className="pointer-events-none absolute right-3 top-1/2 z-20 hidden -translate-y-1/2 lg:block"
    >
      <div className="pointer-events-auto flex flex-col items-end gap-2 py-2">
        {orderedTurns.map((turn) => {
          const isActive = turn.turn_id === activeTurnId;
          return (
            <button
              key={`${turn.turn_id}:${turn.started_seq}`}
              type="button"
              aria-label={`Scroll to ${turn.user_preview || "turn"}`}
              className={[
                "h-2 rounded-full transition-all duration-150",
                isActive ? "w-8 bg-primary" : statusClass(turn.status),
              ].join(" ")}
              onMouseEnter={() => setHoveredTurnId(turn.turn_id)}
              onMouseLeave={() => setHoveredTurnId(null)}
              onFocus={() => setHoveredTurnId(turn.turn_id)}
              onBlur={() => setHoveredTurnId(null)}
              onClick={async () => {
                const didScroll = scrollToTurn(sessionId, turn.turn_id);
                setActiveTurnId(turn.turn_id);
                if (didScroll) return;

                setLoadingHistoryTurnId(turn.turn_id);
                try {
                  const events = await getJSON<AgentEvent[]>(
                    `/sessions/${encodeURIComponent(sessionId)}/events/history?limit=10000`,
                  );
                  replaceSessionEvents(sessionId, events);
                  requestAnimationFrame(() => {
                    scrollToTurn(sessionId, turn.turn_id);
                    setActiveTurnId(turn.turn_id);
                  });
                } catch (err) {
                  console.warn("Failed to hydrate turn history", err);
                } finally {
                  setLoadingHistoryTurnId(null);
                }
              }}
            />
          );
        })}
      </div>
      {hoveredTurn && (
        <div className="pointer-events-none absolute right-12 top-1/2 w-64 -translate-y-1/2 rounded-lg border border-border bg-secondary/95 px-3 py-2 text-right shadow-xl">
          <div className="truncate text-xs font-medium text-foreground">
            {hoveredTurn.user_preview || "Untitled turn"}
          </div>
          {loadingHistoryTurnId === hoveredTurn.turn_id && (
            <div className="mt-1 text-[11px] leading-4 text-muted-foreground">
              Loading turn history...
            </div>
          )}
          {hoveredTurn.assistant_preview && (
            <div className="mt-1 line-clamp-2 text-[11px] leading-4 text-muted-foreground">
              {hoveredTurn.assistant_preview}
            </div>
          )}
        </div>
      )}
    </nav>
  );
}

function scrollToTurn(sessionId: string, turnId: string): boolean {
  const root = getScrollRoot(sessionId);
  const el = root?.querySelector<HTMLElement>(`[data-turn-id="${escapeCss(turnId)}"]`);
  if (!root || !el) return false;
  const rootTop = root.getBoundingClientRect().top;
  const elTop = el.getBoundingClientRect().top;
  root.scrollTo({
    top: root.scrollTop + elTop - rootTop - 24,
    behavior: "smooth",
  });
  return true;
}

function findActiveTurnId(root: HTMLElement): string | null {
  const anchors = Array.from(root.querySelectorAll<HTMLElement>("[data-turn-id]"));
  if (anchors.length === 0) return null;
  const rootTop = root.getBoundingClientRect().top;
  const marker = rootTop + Math.min(160, root.clientHeight * 0.3);
  let active = anchors[0];
  for (const anchor of anchors) {
    if (anchor.getBoundingClientRect().top <= marker) active = anchor;
    else break;
  }
  return active.getAttribute("data-turn-id");
}

function getScrollRoot(sessionId: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(
    `[data-chat-scroll-root="${escapeCss(sessionId)}"]`,
  );
}

function escapeCss(value: string): string {
  return typeof CSS !== "undefined" && CSS.escape ? CSS.escape(value) : value;
}

function statusClass(status: TurnSummary["status"]) {
  switch (status) {
    case "running":
    case "waiting_approval":
      return "w-6 bg-blue-500/85 hover:w-8 hover:bg-blue-400";
    case "failed":
      return "w-5 bg-destructive/80 hover:w-8 hover:bg-destructive";
    case "cancelled":
      return "w-4 bg-muted-foreground/45 hover:w-8 hover:bg-muted-foreground/70";
    case "completed":
    default:
      return "w-4 bg-muted-foreground/45 hover:w-8 hover:bg-muted-foreground/80";
  }
}
