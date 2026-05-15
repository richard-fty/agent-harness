import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const file = resolve(process.argv[2] || "sales.csv");
const text = readFileSync(file, "utf-8");
const rows = parseCsv(text);
const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

const numericColumns = columns
  .map((column) => {
    const values = rows
      .map((row) => Number(String(row[column] ?? "").replace(/[$,%]/g, "")))
      .filter((value) => Number.isFinite(value));
    return {
      name: column,
      numeric_count: values.length,
      min: values.length ? Math.min(...values) : null,
      max: values.length ? Math.max(...values) : null,
      sum: values.length ? values.reduce((total, value) => total + value, 0) : null,
    };
  })
  .filter((column) => column.numeric_count > 0);

console.log(JSON.stringify({
  file,
  row_count: rows.length,
  columns,
  numeric_columns: numericColumns,
  sample_rows: rows.slice(0, 5),
}, null, 2));

function parseCsv(input) {
  const lines = input.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const headers = splitCsvLine(lines[0]).map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""]),
    );
  });
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
