import { chmodSync, copyFileSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { extname, join, relative, resolve } from "node:path";

const dist = resolve("dist");
rmSync(dist, { recursive: true, force: true });
mkdirSync(resolve(dist, "src"), { recursive: true });

for (const [source, target] of [
  ["index.html", "index.html"],
  ["src/main.js", "src/main.js"],
  ["src/styles.css", "src/styles.css"],
]) {
  const output = resolve(dist, target);
  copyFileSync(source, output);
  chmodSync(output, 0o644);
}

for (const entry of readdirSync(".")) {
  if ([".csv", ".json"].includes(extname(entry).toLowerCase()) && statSync(entry).isFile()) {
    copyFileSync(entry, resolve(dist, entry));
  }
}

function copyDir(source, target) {
  mkdirSync(target, { recursive: true });
  for (const entry of readdirSync(source)) {
    const src = join(source, entry);
    const dst = join(target, entry);
    if (statSync(src).isDirectory()) {
      copyDir(src, dst);
    } else {
      copyFileSync(src, dst);
      chmodSync(dst, 0o644);
    }
  }
}

if (statSync("src").isDirectory()) {
  copyDir("src", resolve(dist, "src"));
}

console.log(`Build completed at ${relative(process.cwd(), dist)}.`);
