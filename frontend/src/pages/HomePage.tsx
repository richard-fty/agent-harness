import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { getJSON, postJSON } from "../lib/api";
import type { Session } from "../types";
import { TopBar } from "../components/TopBar";

export function HomePage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [creating, setCreating] = useState<"chat" | "stock" | null>(null);
  const navigate = useNavigate();

  async function refresh() {
    const nextSessions = await getJSON<Session[]>("/sessions");
    setSessions(nextSessions);
  }
  useEffect(() => { void refresh(); }, []);

  async function newSession() {
    setCreating("chat");
    try {
      const s = await postJSON<Session>("/sessions", { model: "deepseek/deepseek-chat" });
      navigate(`/session/${s.id}`);
    } finally {
      setCreating(null);
    }
  }

  async function startStockDemo() {
    setCreating("stock");
    try {
      const s = await postJSON<Session>("/sessions", { model: "deepseek/deepseek-chat" });
      await postJSON(`/sessions/${s.id}/turns`, {
        user_input: "Brief me on Apple stock. Use the latest news, market data, and technical indicators.",
      });
      navigate(`/session/${s.id}`);
    } finally {
      setCreating(null);
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar />
      <div className="flex-1 max-w-5xl mx-auto w-full p-8">
        <div className="rounded-[28px] border border-border bg-secondary/15 p-8">
          <div className="inline-flex rounded-full border border-border bg-background/80 px-3 py-1 text-xs uppercase tracking-[0.16em] text-muted-foreground">
            Stock analysis demo
          </div>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight">
            Research one stock with live tools.
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-7 text-muted-foreground">
            Start a focused stock briefing that loads the stock analysis skill, searches the
            company&apos;s latest news, fetches market data, and computes technical indicators.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button onClick={startStockDemo} disabled={creating !== null}>
              {creating === "stock" ? "Starting…" : "Run Apple stock demo"}
            </Button>
            <Button variant="outline" onClick={newSession} disabled={creating !== null}>
              {creating === "chat" ? "Creating…" : "Open empty session"}
            </Button>
          </div>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <section className="rounded-3xl border border-border bg-background/60 p-6">
            <h2 className="text-lg font-semibold">Demo flow</h2>
            <div className="mt-4 space-y-3 text-sm text-muted-foreground">
              <div className="rounded-2xl border border-border bg-secondary/20 p-4">
                Load <span className="text-foreground">Stock & Crypto Analysis</span>
              </div>
              <div className="rounded-2xl border border-border bg-secondary/20 p-4">
                Search <span className="text-foreground">Apple latest news</span>
              </div>
              <div className="rounded-2xl border border-border bg-secondary/20 p-4">
                Fetch price data and compute RSI/SMA indicators
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-border bg-background/60 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Recent sessions</h2>
            </div>
            {sessions.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No sessions yet. Run the stock demo or open an empty session.
              </p>
            ) : (
              <ul className="space-y-3">
                {sessions.map((s) => (
                  <li
                    key={s.id}
                    className="cursor-pointer rounded-2xl border border-border p-4 transition-colors hover:bg-secondary/20"
                    onClick={() => navigate(`/session/${s.id}`)}
                  >
                    <div className="font-mono text-sm">{s.id.slice(0, 8)}</div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {s.model} · {s.state} · {new Date(s.created_at * 1000).toLocaleString()}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
