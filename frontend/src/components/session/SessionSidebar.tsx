import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { MessageSquare, PanelLeftClose, PanelLeftOpen, Plus, Trash2 } from "lucide-react";
import { del, getJSON, postJSON } from "../../lib/api";
import type { Session } from "../../types";

export function SessionSidebar({ activeSessionId }: { activeSessionId: string }) {
  const [open, setOpen] = useState(true);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const navigate = useNavigate();

  async function refresh() {
    setLoading(true);
    try {
      setSessions(await getJSON<Session[]>("/sessions"));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, [activeSessionId]);

  const sortedSessions = useMemo(
    () => [...sessions].sort((a, b) => b.created_at - a.created_at),
    [sessions],
  );

  async function createSession() {
    if (creating) return;
    setCreating(true);
    try {
      const session = await postJSON<Session>("/sessions", {
        model: "deepseek/deepseek-chat",
      });
      setSessions((current) => [session, ...current.filter((item) => item.id !== session.id)]);
      navigate(`/session/${session.id}`);
    } finally {
      setCreating(false);
    }
  }

  async function deleteSession(sessionId: string) {
    if (!window.confirm("Delete this session?")) return;
    setDeletingId(sessionId);
    try {
      await del(`/sessions/${sessionId}`);
      const remaining = sortedSessions.filter((item) => item.id !== sessionId);
      setSessions(remaining);
      if (sessionId === activeSessionId) {
        const nextSession = remaining[0];
        navigate(nextSession ? `/session/${nextSession.id}` : "/chat", { replace: true });
      }
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <aside
      className={`shrink-0 bg-secondary/20 transition-[width] duration-200 ${
        open ? "w-72" : "w-14"
      }`}
    >
      <div className="flex h-full min-h-0 flex-col">
        <div className="flex items-center gap-2 px-3 py-3">
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            aria-label={open ? "Close sidebar" : "Open sidebar"}
          >
            {open ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
          </button>
          {open && (
            <button
              type="button"
              onClick={createSession}
              disabled={creating}
              className="flex min-w-0 flex-1 items-center gap-2 rounded-md border border-border bg-background/60 px-3 py-2 text-sm transition-colors hover:bg-secondary"
            >
              <Plus className="h-4 w-4" />
              <span>New chat</span>
            </button>
          )}
        </div>

        {open ? (
          <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
            <div className="px-2 pb-2 pt-1 text-xs uppercase text-muted-foreground">
              Sessions
            </div>
            {loading && sortedSessions.length === 0 ? (
              <div className="px-2 py-3 text-sm text-muted-foreground">Loading...</div>
            ) : sortedSessions.length === 0 ? (
              <div className="px-2 py-3 text-sm text-muted-foreground">No sessions yet.</div>
            ) : (
              <div className="space-y-1">
                {sortedSessions.map((session) => {
                  const active = session.id === activeSessionId;
                  return (
                    <Link
                      key={session.id}
                      to={`/session/${session.id}`}
                      className={`group flex items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors ${
                        active ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-secondary/70 hover:text-foreground"
                      }`}
                    >
                      <MessageSquare className="h-4 w-4 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="truncate font-medium">{sessionTitle(session)}</div>
                        <div className="truncate text-xs text-muted-foreground">
                          {formatSessionDate(session.created_at)}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          void deleteSession(session.id);
                        }}
                        disabled={deletingId === session.id}
                        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground opacity-0 transition hover:bg-background hover:text-destructive group-hover:opacity-100 disabled:opacity-50"
                        aria-label="Delete session"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-1 flex-col items-center gap-2 px-2 py-2">
            <button
              type="button"
              onClick={createSession}
              disabled={creating}
              className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              aria-label="New chat"
            >
              <Plus className="h-4 w-4" />
            </button>
            <Link
              to={`/session/${activeSessionId}`}
              className="flex h-9 w-9 items-center justify-center rounded-md bg-secondary text-foreground"
              aria-label="Current session"
            >
              <MessageSquare className="h-4 w-4" />
            </Link>
          </div>
        )}
      </div>
    </aside>
  );
}

function sessionTitle(session: Session): string {
  return `Chat ${session.id.slice(0, 8)}`;
}

function formatSessionDate(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}
