# Data Viz

Planning Mode: on

Use this skill for data visualization and data-story app tasks where the goal is to turn a dataset into a polished, interactive report artifact.

Workflow:
1. If the user uploaded files, inspect the provided paths with `read_file` before designing charts.
2. Inspect the dataset/schema before designing charts.
3. Create or update a concise plan with steps for dataset inspection, metric derivation, story layout, filters/interactions, verification, and preview.
4. Derive metrics from source data, not hardcoded final values.
5. Build the data-story app with a clear headline insight, KPI strip, primary chart, supporting breakdown, filters, data notes, and details table.
6. Verify calculations against the dataset and run the relevant build/test command when available.
7. Call `start_app_preview` after the app builds or when the user asks to see it.
8. Surface patches and previews as artifacts.

Rules:
- Use the minimum stack already present in the template.
- Keep TodoItem statuses to `pending`, `in_progress`, `completed`, or `failed`.
- Use `update_plan`, not generic todo tools, for data-viz app checklists.
- Label chart axes, legends, units, and filters clearly.
- Written insights must be traceable to a computed metric or visible chart value.
- Do not ask the user to paste data that is already available as an uploaded file path.
- Do not load stock/trading capability packs for ordinary sales, revenue, operational, or product datasets unless the user explicitly asks about traded securities or market prices.
- Avoid fake insights, placeholder chart marks, decorative landing-page composition, and one-color dashboards with weak hierarchy.
- The final user-visible deliverable should include an `app_preview` artifact unless the preview command fails.
