import { useRef, useState } from "react";
import { Button } from "../ui/button";
import { useStore } from "../../store";
import { postFile, postJSON } from "../../lib/api";
import { ArrowUp, Loader2, Paperclip, X } from "lucide-react";

type UploadedFile = {
  id: string;
  filename: string;
  path: string;
  size: number;
  content_type: string;
};

const ACCEPTED_DATA_FILES = ".csv,.tsv,.json,.jsonl,.txt";

export function Composer({ sessionId }: { sessionId: string }) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [sending, setSending] = useState(false);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const session = useStore((s) => s.sessions[sessionId]);
  const pending = session?.pending ?? null;
  const status = session?.status ?? "idle";

  async function send() {
    const msg = text.trim();
    if ((!msg && files.length === 0) || sending) return;
    setSending(true);
    try {
      for (const file of files) {
        await postFile<UploadedFile>(`/sessions/${sessionId}/uploads`, file);
      }
      await postJSON(`/sessions/${sessionId}/turns`, {
        user_input: msg || "Use the uploaded file to build a data visualization story.",
      });
      setText("");
      setFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (taRef.current) taRef.current.style.height = "auto";
    } catch (e) {
      console.error(e);
    } finally {
      setSending(false);
    }
  }

  async function resolveApproval(action: "approve_once" | "deny") {
    await postJSON(`/sessions/${sessionId}/approvals`, { action });
  }

  // Approval strip takes over the composer when agent is waiting for input.
  if (pending) {
    return (
      <div className="bg-background">
        <div className="mx-auto max-w-3xl px-6 py-4 space-y-2">
          <div className="text-sm">
            <span className="font-medium">Approval required</span> for{" "}
            <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs">
              {pending.tool_name}
            </code>{" "}
            <span className="text-muted-foreground">· {pending.reason}</span>
          </div>
          <div className="flex gap-2">
            <Button onClick={() => resolveApproval("approve_once")}>Approve once</Button>
            <Button variant="outline" onClick={() => resolveApproval("deny")}>Deny</Button>
          </div>
        </div>
      </div>
    );
  }

  const disabled = status === "running" || sending;
  const canSend = text.trim().length > 0 || files.length > 0;

  return (
    <div className="bg-background">
      <form
        className="mx-auto max-w-3xl px-6 pt-2 pb-3"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        {files.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {files.map((file, index) => (
              <span
                key={`${file.name}-${file.lastModified}-${index}`}
                className="inline-flex max-w-full items-center gap-2 rounded-md border border-border bg-secondary/50 px-2.5 py-1 text-xs"
              >
                <Paperclip className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <span className="min-w-0 truncate">{file.name}</span>
                <span className="shrink-0 text-muted-foreground">{humanSize(file.size)}</span>
                <button
                  type="button"
                  className="shrink-0 rounded p-0.5 text-muted-foreground hover:bg-background hover:text-foreground"
                  aria-label={`Remove ${file.name}`}
                  onClick={() => {
                    setFiles((current) => current.filter((_, i) => i !== index));
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }}
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </span>
            ))}
          </div>
        )}
        <div className="relative flex items-end rounded-2xl border border-input bg-[#2b2c30] focus-within:border-ring focus-within:ring-1 focus-within:ring-ring">
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept={ACCEPTED_DATA_FILES}
            multiple
            onChange={(e) => {
              const selected = Array.from(e.currentTarget.files ?? []);
              setFiles((current) => [...current, ...selected]);
            }}
          />
          <Button
            type="button"
            size="icon"
            variant="ghost"
            disabled={disabled}
            title="Attach data file"
            aria-label="Attach data file"
            className="absolute left-2 bottom-2 h-8 w-8 rounded-full text-muted-foreground hover:text-foreground"
            onClick={() => fileInputRef.current?.click()}
          >
            <Paperclip className="h-4 w-4" />
          </Button>
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              // auto-grow up to ~8 lines
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = Math.min(el.scrollHeight, 240) + "px";
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={disabled ? "Agent is working…" : "Ask anything, create anything"}
            disabled={disabled}
            rows={1}
            className="flex-1 resize-none bg-transparent py-3 pl-12 pr-12 text-sm leading-relaxed placeholder:text-muted-foreground focus:outline-none disabled:opacity-60"
          />
          <Button
            type="submit"
            size="icon"
            disabled={disabled || !canSend}
            className="absolute right-2 bottom-2 h-8 w-8 rounded-full"
            aria-label="Send"
          >
            {sending || status === "running" ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <ArrowUp className="h-4 w-4" />
            )}
          </Button>
        </div>
      </form>
    </div>
  );
}

function humanSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}
