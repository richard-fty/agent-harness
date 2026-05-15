# Data Viz

Planning Mode: on

Use this skill for data visualization and data-story app tasks where the goal is to turn a dataset into a polished, interactive report artifact.

Workflow:
1. If the user uploaded files, inspect the provided paths with `read_file` before designing charts.
2. Inspect the dataset/schema before designing charts.
3. Create or update a concise plan with steps for dataset inspection, metric derivation, story layout, filters/interactions, verification, and preview.
4. Derive metrics from source data, not hardcoded final values.
5. Build from the existing Vite data-story template. Split work across small files instead of producing one giant standalone HTML file.
6. Build the data-story app with a clear headline insight, KPI strip, primary chart, supporting breakdown, filters, data notes, and details table.
7. Verify calculations against the dataset and run the relevant build/test command when available.
8. Call `start_app_preview` after the app builds or when the user asks to see it. Set `cwd` to the generated app directory that contains `package.json` or `index.html`, not the repository root.
9. Surface patches and previews as artifacts.

Rules:
- Use the minimum stack already present in the template.
- Keep TodoItem statuses to `pending`, `in_progress`, `completed`, or `failed`.
- Use `update_plan`, not generic todo tools, for data-viz app checklists.
- Do not write a full app as a single large `.html` file. Use the template files: `index.html`, `src/main.js`, `src/styles.css`, and `src/data.js` or external data files.
- Keep uploaded datasets external. Do not inline a full CSV or large dataset into HTML or JavaScript.
- Use `write_file` only for small files. For large generated content, split it into smaller files or use `append_file(path, content, reset=true)` for the first chunk and `append_file(path, content)` for later chunks.
- Prefer `emit_artifact=false` when writing internal source files; the final `app_preview` is the user-facing artifact.
- When working inside the data-story template, use `npm run analyze:csv -- <path>` or direct `read_file` to inspect data before coding the visuals.
- Label chart axes, legends, units, and filters clearly.
- Written insights must be traceable to a computed metric or visible chart value.
- Do not ask the user to paste data that is already available as an uploaded file path.
- Do not load stock/trading capability packs for ordinary sales, revenue, operational, or product datasets unless the user explicitly asks about traded securities or market prices.
- Avoid fake insights, placeholder chart marks, decorative landing-page composition, and one-color dashboards with weak hierarchy.
- The final user-visible deliverable should include an `app_preview` artifact unless the preview command fails.
