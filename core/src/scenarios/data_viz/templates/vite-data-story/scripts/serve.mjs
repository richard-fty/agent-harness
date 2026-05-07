import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";

const root = resolve("dist");
const portArg = process.argv.findIndex((arg) => arg === "--port" || arg === "-p");
const port = Number(
  portArg >= 0 && process.argv[portArg + 1]
    ? process.argv[portArg + 1]
    : process.env.PORT || 3000,
);

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

const server = createServer((req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  let file = normalize(decodeURIComponent(url.pathname)).replace(/^\/+/, "");
  if (!file) file = "index.html";
  let path = join(root, file);
  if (!path.startsWith(root) || !existsSync(path) || statSync(path).isDirectory()) {
    path = join(root, "index.html");
  }
  res.setHeader("Content-Type", types[extname(path)] || "application/octet-stream");
  createReadStream(path).pipe(res);
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Serving ${root} at http://127.0.0.1:${port}`);
});
