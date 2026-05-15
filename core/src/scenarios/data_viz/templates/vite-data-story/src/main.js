import { DATA_URL, loadCsv, summarizeRows } from "./data.js";

const root = document.querySelector("#root");

async function renderStarter() {
  let summary = null;
  try {
    const rows = await loadCsv(DATA_URL);
    summary = summarizeRows(rows);
  } catch {
    summary = null;
  }

  root.innerHTML = `
    <main class="story-shell">
      <section class="empty-state">
        <p class="eyebrow">Data story workspace</p>
        <h1>Build the requested data visualization story</h1>
        <p>Inspect the provided CSV, compute the metrics, and replace this starter view with a focused, interactive report.</p>
        <dl class="starter-grid" aria-label="Dataset starter summary">
          <div>
            <dt>Data file</dt>
            <dd>${DATA_URL}</dd>
          </div>
          <div>
            <dt>Rows detected</dt>
            <dd>${summary ? summary.rowCount.toLocaleString() : "Run the app after data is available"}</dd>
          </div>
          <div>
            <dt>Columns</dt>
            <dd>${summary ? summary.columns.join(", ") : "Inspect with read_file first"}</dd>
          </div>
        </dl>
      </section>
    </main>
  `;
}

renderStarter();
