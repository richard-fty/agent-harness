# Data Viz Reference

## Story Structure

Use this order for generated reports:

1. Headline insight that answers the user's main question.
2. KPI strip with 3-5 important values.
3. Main chart for the primary trend, comparison, or relationship.
4. Breakdown charts that explain the driver behind the headline.
5. Filters for the dimensions the user needs to compare.
6. Data notes with formulas, missing-data caveats, and definitions.
7. Details table for inspectability.

## Chart Selection

- Time series: line chart or monthly/weekly bars.
- Ranked categories: horizontal bars.
- Part-to-whole: stacked bars for comparison; avoid pie by default.
- Rates: show numerator/denominator definition in data notes.
- Multi-series comparison: include a legend and clear color mapping.

## Eval-Friendly Markup

Expose deterministic values without compromising UI polish:

```html
<section data-testid="kpi-total-revenue" data-value="1842500">$1.84M revenue</section>
<svg data-testid="monthly-revenue-chart">
  <g data-series="monthly-revenue" data-points="12"></g>
</svg>
```

Use accessible names for controls and chart sections.

## App File Layout

Use the Vite data-story template as a small multi-file app. Do not generate a single large standalone HTML document.

Recommended layout:

```text
index.html
src/main.js       # state, metric derivation, DOM rendering, interactions
src/styles.css    # all visual styling and responsive layout
src/data.js       # small derived arrays or loader helpers
public/*.csv      # source datasets copied or referenced externally
```

Keep each file focused:

- `index.html`: root element and module script only.
- `src/main.js`: parse/load data, compute metrics, render sections, wire filters.
- `src/styles.css`: layout, chart styling, color tokens, responsive rules.
- `src/data.js`: reusable data helpers, constants, or compact derived data.

If a file would exceed roughly 8-12KB, split it by responsibility or write it with `append_file` chunks.

## Chunked File Writing

Function-call JSON is a fragile transport for very large HTML/CSS/JS strings. Prefer small files. When unavoidable, write in chunks:

```text
append_file("src/main.js", "first chunk...", reset=true, emit_artifact=false)
append_file("src/main.js", "second chunk...", emit_artifact=false)
append_file("src/main.js", "final chunk...", emit_artifact=false)
```

Use `write_file` for small files only:

```text
write_file("src/data.js", "export const DATA_URL = '/sales.csv';", emit_artifact=false)
```

Internal source files should usually use `emit_artifact=false`. The `app_preview` is the user-facing artifact.

## Preview CWD

Start previews from the app folder, not the repository root.

Good:

```text
start_app_preview(cwd=".", command="pnpm dev --host 127.0.0.1 --port {port}")
start_app_preview(cwd="sales_story_app", command="pnpm dev --host 127.0.0.1 --port {port}")
```

Bad:

```text
start_app_preview(cwd="/Users/.../apex_agent")
```

The `cwd` directory must contain the generated app's `package.json` or `index.html`.

## Dataset Handling

Keep source data external:

- read uploaded CSV files with `read_file`
- in the data-story template, run `npm run analyze:csv -- <csv-path>` for a compact schema/sample/numeric summary
- copy or reference CSV under `public/` when the app must load it
- compute summaries from the source data
- never inline a full CSV into JavaScript or HTML

For CSV-driven stories, prefer this pattern:

```js
async function loadCsv(path) {
  const text = await fetch(path).then((res) => res.text());
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}
```

For eval cases, expose computed values in `data-testid` and `data-value` attributes so Playwright can verify metrics.
