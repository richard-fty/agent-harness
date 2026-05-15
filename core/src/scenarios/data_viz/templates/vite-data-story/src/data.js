export const DATA_URL = "/sales.csv";

export async function loadCsv(path = DATA_URL) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load CSV: ${response.status}`);
  }
  const text = await response.text();
  return parseCsv(text);
}

export function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const headers = splitCsvLine(lines[0]).map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""]),
    );
  });
}

export function summarizeRows(rows) {
  const columns = rows.length > 0 ? Object.keys(rows[0]) : [];
  return {
    rowCount: rows.length,
    columns,
  };
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];
    if (char === '"' && quoted && next === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}
