import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getJSON, postJSON } from "../lib/api";
import type { Session } from "../types";

let pendingOpenChat: Promise<Session> | null = null;

export function ChatEntryPage() {
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    async function openChat() {
      try {
        const session = await getOrCreateChatSession();
        if (!cancelled) navigate(`/session/${session.id}`, { replace: true });
      } catch (err) {
        if (!cancelled) {
          console.error("Failed to open chat session", err);
        }
      }
    }
    void openChat();
    return () => {
      cancelled = true;
    };
  }, [navigate]);

  return (
    <div className="min-h-screen flex items-center justify-center p-6 text-sm text-muted-foreground">
      Opening chat…
    </div>
  );
}

async function getOrCreateChatSession(): Promise<Session> {
  if (pendingOpenChat) return pendingOpenChat;

  pendingOpenChat = (async () => {
    const sessions = await getJSON<Session[]>("/sessions");
    const existing = [...sessions].sort((a, b) => b.created_at - a.created_at)[0];
    if (existing) return existing;
    return postJSON<Session>("/sessions", {
      model: "deepseek/deepseek-chat",
    });
  })();

  try {
    return await pendingOpenChat;
  } finally {
    pendingOpenChat = null;
  }
}
