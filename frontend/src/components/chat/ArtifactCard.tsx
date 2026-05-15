import { useStore } from "../../store";
import {
  FileText,
  Code2,
  FileJson,
  FileBarChart,
  Terminal,
  File,
  Image as ImageIcon,
  FileType2,
  MonitorPlay,
} from "lucide-react";
import type { ArtifactKind } from "../../types";

function iconFor(kind: ArtifactKind) {
  switch (kind) {
    case "markdown": return <FileText className="h-5 w-5" />;
    case "code":     return <Code2 className="h-5 w-5" />;
    case "json":     return <FileJson className="h-5 w-5" />;
    case "wealth_snapshot": return <FileBarChart className="h-5 w-5" />;
    case "path_comparison": return <FileBarChart className="h-5 w-5" />;
    case "action_checklist": return <FileText className="h-5 w-5" />;
    case "image":    return <ImageIcon className="h-5 w-5" />;
    case "pdf":      return <FileType2 className="h-5 w-5" />;
    case "terminal_log": return <Terminal className="h-5 w-5" />;
    case "plan":     return <FileBarChart className="h-5 w-5" />;
    case "app_preview": return <MonitorPlay className="h-5 w-5" />;
    default:         return <File className="h-5 w-5" />;
  }
}

function humanSize(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function firstLine(content: string, max = 140): string {
  const stripped = content
    .replace(/^#+\s*/gm, "")      // drop heading hashes
    .split("\n")
    .map((l) => l.trim())
    .find((l) => l.length > 0) ?? "";
  return stripped.length > max ? stripped.slice(0, max) + "…" : stripped;
}

export function ArtifactCard({
  sessionId,
  artifactId,
}: {
  sessionId: string;
  artifactId: string;
}) {
  const artifact = useStore((s) => s.sessions[sessionId]?.artifacts[artifactId]);
  const openArtifact = useStore((s) => s.openArtifact);
  const setArtifactView = useStore((s) => s.setArtifactView);
  const active = useStore(
    (s) => s.ui.panel.kind === "artifact" && s.ui.panel.artifactId === artifactId,
  );

  if (!artifact) return null;

  const size = artifact.size ?? new Blob([artifact.content]).size;
  const isAppPreview = artifact.kind === "app_preview";
  const subtitle = isAppPreview
    ? artifact.description || "Live browser preview"
    : artifact.description || firstLine(artifact.content);
  const meta = isAppPreview
    ? "app preview"
    : `${artifact.kind} · ${humanSize(size)}${artifact.language ? ` · ${artifact.language}` : ""}`;

  return (
    <button
      onClick={() => {
        if (isAppPreview) setArtifactView("preview");
        openArtifact(artifactId);
      }}
      className={`w-full text-left rounded-xl border transition-colors
        ${active ? "border-primary/50 bg-secondary/70" : "border-border bg-secondary/30 hover:bg-secondary/60"}
        p-4 flex gap-3`}
    >
      <div className="h-10 w-10 rounded-lg bg-secondary flex items-center justify-center shrink-0">
        {iconFor(artifact.kind)}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium truncate">{artifact.name}</span>
          {!artifact.finalized && (
            <span className="text-xs text-muted-foreground animate-pulse">streaming…</span>
          )}
          {isAppPreview && artifact.finalized && (
            <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
              Browser
            </span>
          )}
        </div>
        {subtitle && (
          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{subtitle}</p>
        )}
        <div className="text-xs text-muted-foreground mt-1">
          {meta}
        </div>
      </div>
    </button>
  );
}
