import { useEffect, useMemo, useState } from "react";
import { useStore } from "../../store";
import { getJSON } from "../../lib/api";
import type { TurnSummary } from "../../types";

export function TurnNavigator({ sessionId }: { sessionId: string }) {
  const turns = useStore((s) => s.turnsBySession[sessionId] ?? []);
  const setTurns = useStore((s) => s.setTurns);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [hoveredTurnId, setHoveredTurnId] = useState<string | null>(null);

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
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        const turnId = visible?.target.getAttribute("data-turn-id");
        if (turnId) setActiveTurnId(turnId);
      },
      { root: null, rootMargin: "-35% 0px -50% 0px", threshold: [0, 0.25, 0.5] },
    );
    document.querySelectorAll<HTMLElement>("[data-turn-id]").forEach((el) => {
      observer.observe(el);
    });
    return () => observer.disconnect();
  }, [turns.length]);

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
              onClick={() => {
                scrollToTurn(turn.turn_id);
                setActiveTurnId(turn.turn_id);
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

function scrollToTurn(turnId: string) {
  const escaped = typeof CSS !== "undefined" && CSS.escape ? CSS.escape(turnId) : turnId;
  const el = document.querySelector(`[data-turn-id="${escaped}"]`);
  el?.scrollIntoView({ behavior: "smooth", block: "start" });
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
