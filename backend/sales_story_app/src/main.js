// ===== Sales Storyboard 2025 — Main Application =====

// ---------- Data ----------
const MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const CATEGORIES = ["Platform","Subscriptions","Hardware","Services","Training"];
const REGIONS = ["North America","EMEA","APAC","LATAM"];

const CAT_COLORS = {
  Platform: "#4361ee",
  Subscriptions: "#f72585",
  Hardware: "#4cc9f0",
  Services: "#7209b7",
  Training: "#f8961e"
};
const REG_COLORS = {
  "North America": "#4361ee",
  "EMEA": "#f72585",
  "APAC": "#4cc9f0",
  "LATAM": "#f8961e"
};

let rawData = [];
let filteredData = [];
let state = { region: "all", category: "all" };

// ---------- Data Loading ----------
async function loadData() {
  const resp = await fetch("/public/sales.csv");
  const text = await resp.text();
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(",");
    const row = {};
    headers.forEach((h, idx) => {
      const v = vals[idx];
      if (h === "revenue" || h === "gross_margin") row[h] = parseFloat(v);
      else if (h === "units") row[h] = parseInt(v, 10);
      else row[h] = v;
    });
    rawData.push(row);
  }
  applyFilters();
}

// ---------- Filtering ----------
function applyFilters() {
  filteredData = rawData.filter(d => {
    if (state.region !== "all" && d.region !== state.region) return false;
    if (state.category !== "all" && d.category !== state.category) return false;
    return true;
  });
  renderAll();
}

// ---------- Aggregation Helpers ----------
function monthlyRevenue(data) {
  const map = {};
  MONTH_ORDER.forEach(m => map[m] = 0);
  data.forEach(d => { map[d.month] += d.revenue; });
  return map;
}

function regionRevenue(data) {
  const map = {};
  REGIONS.forEach(r => map[r] = 0);
  data.forEach(d => { map[d.region] += d.revenue; });
  return map;
}

function categoryRevenue(data) {
  const map = {};
  CATEGORIES.forEach(c => map[c] = 0);
  data.forEach(d => { map[d.category] += d.revenue; });
  return map;
}

function categoryMargin(data) {
  const groups = {};
  CATEGORIES.forEach(c => groups[c] = []);
  data.forEach(d => { if (groups[d.category]) groups[d.category].push(d.gross_margin); });
  const avg = {};
  Object.keys(groups).forEach(c => {
    const arr = groups[c];
    avg[c] = arr.length ? arr.reduce((a,b) => a+b, 0) / arr.length : 0;
  });
  return avg;
}

function monthlyRegionRevenue(data) {
  const map = {};
  REGIONS.forEach(r => { map[r] = {}; MONTH_ORDER.forEach(m => map[r][m] = 0); });
  data.forEach(d => { if (map[d.region]) map[d.region][d.month] += d.revenue; });
  return map;
}

function regionCategoryRevenue(data) {
  const map = {};
  REGIONS.forEach(r => { map[r] = {}; CATEGORIES.forEach(c => map[r][c] = 0); });
  data.forEach(d => { if (map[d.region]) map[d.region][d.category] += d.revenue; });
  return map;
}

// ---------- Number formatting ----------
function fmt(n) {
  if (n >= 1000000) return "$" + (n / 1000000).toFixed(2) + "M";
  if (n >= 1000) return "$" + (n / 1000).toFixed(1) + "K";
  return "$" + n.toFixed(0);
}
function fmtPct(n) { return (n * 100).toFixed(1) + "%"; }
// ===== Chart Rendering Functions =====

// ---------- Chart 1: Monthly Revenue Line Chart ----------
function renderMonthlyChart() {
  const svg = document.getElementById("monthly-revenue-chart");
  const data = monthlyRevenue(filteredData);
  const values = MONTH_ORDER.map(m => data[m]);
  const maxVal = Math.max(...values, 1);
  const W = 900, H = 400;
  const pad = { top: 30, right: 30, bottom: 50, left: 70 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const xScale = (i) => pad.left + (i / (MONTH_ORDER.length - 1)) * chartW;
  const yScale = (v) => pad.top + chartH - (v / maxVal) * chartH;

  let html = `<defs>
    <linearGradient id="lineGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4361ee" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#4361ee" stop-opacity="0"/>
    </linearGradient>
  </defs>`;

  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * chartH;
    const val = maxVal - (i / 4) * maxVal;
    html += `<line x1="${pad.left}" y1="${y}" x2="${W - pad.right}" y2="${y}" stroke="#e2e5ec" stroke-width="1"/>`;
    html += `<text x="${pad.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#6b7280">${fmt(val)}</text>`;
  }

  // Area fill
  const areaPoints = values.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
  const basePoints = `${xScale(values.length-1)},${pad.top + chartH} ${xScale(0)},${pad.top + chartH}`;
  html += `<polygon points="${areaPoints} ${basePoints}" fill="url(#lineGrad)" stroke="none"/>`;

  // Line
  const linePoints = values.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
  html += `<polyline points="${linePoints}" fill="none" stroke="#4361ee" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`;

  // Dots
  values.forEach((v, i) => {
    html += `<circle cx="${xScale(i)}" cy="${yScale(v)}" r="5" fill="#4361ee" stroke="#fff" stroke-width="2" class="chart-dot" data-month="${MONTH_ORDER[i]}" data-value="${fmt(v)}"/>`;
  });

  // X axis labels
  MONTH_ORDER.forEach((m, i) => {
    html += `<text x="${xScale(i)}" y="${H - 14}" text-anchor="middle" font-size="10" fill="#6b7280">${m}</text>`;
  });

  // Axis titles
  html += `<text x="${W/2}" y="${H - 4}" text-anchor="middle" font-size="11" fill="#6b7280">Month (2025)</text>`;
  html += `<text x="12" y="${H/2}" text-anchor="middle" font-size="11" fill="#6b7280" transform="rotate(-90,12,${H/2})">Revenue</text>`;

  svg.innerHTML = html;

  // Tooltip
  svg.addEventListener("mouseover", (e) => {
    if (e.target.classList.contains("chart-dot")) {
      const month = e.target.dataset.month;
      const val = e.target.dataset.value;
      showTooltip(e, `${month}: ${val}`);
    }
  });
  svg.addEventListener("mouseout", (e) => {
    if (e.target.classList.contains("chart-dot")) hideTooltip();
  });
}

// ---------- Chart 2: Region Donut ----------
function renderRegionDonut() {
  const svg = document.getElementById("region-donut");
  const data = regionRevenue(filteredData);
  const total = Object.values(data).reduce((a,b) => a+b, 0) || 1;
  const cx = 200, cy = 200, r = 130, ir = 70;
  let startAngle = -Math.PI / 2;

  let html = "";
  REGIONS.forEach((reg, i) => {
    const val = data[reg];
    const pct = val / total;
    const angle = pct * 2 * Math.PI;
    const endAngle = startAngle + angle;

    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);
    const largeArc = angle > Math.PI ? 1 : 0;

    const ix1 = cx + ir * Math.cos(startAngle);
    const iy1 = cy + ir * Math.sin(startAngle);
    const ix2 = cx + ir * Math.cos(endAngle);
    const iy2 = cy + ir * Math.sin(endAngle);

    if (pct > 0) {
      html += `<path d="M ${ix1} ${iy1} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} L ${ix2} ${iy2} A ${ir} ${ir} 0 ${largeArc} 0 ${ix1} ${iy1} Z" fill="${REG_COLORS[reg]}" stroke="#fff" stroke-width="1.5" class="donut-seg" data-region="${reg}" data-value="${fmt(val)} (${(pct*100).toFixed(1)}%)"/>`;
    }

    const labelAngle = startAngle + angle / 2;
    const lr = r + 22;
    const lx = cx + lr * Math.cos(labelAngle);
    const ly = cy + lr * Math.sin(labelAngle);
    html += `<text x="${lx}" y="${ly}" text-anchor="middle" font-size="10" fill="#6b7280" dominant-baseline="middle">${reg.split(" ")[0]}</text>`;

    startAngle = endAngle;
  });

  html += `<text x="${cx}" y="${cy - 6}" text-anchor="middle" font-size="22" font-weight="700" fill="#1a1a2e">${fmt(total)}</text>`;
  html += `<text x="${cx}" y="${cy + 14}" text-anchor="middle" font-size="10" fill="#6b7280">Total</text>`;

  svg.innerHTML = html;

  svg.addEventListener("mouseover", (e) => {
    if (e.target.classList.contains("donut-seg")) {
      showTooltip(e, `${e.target.dataset.region}: ${e.target.dataset.value}`);
    }
  });
  svg.addEventListener("mouseout", (e) => {
    if (e.target.classList.contains("donut-seg")) hideTooltip();
  });
}

// ---------- Chart 3: Region Trend ----------
function renderRegionTrend() {
  const svg = document.getElementById("region-trend-chart");
  const data = monthlyRegionRevenue(filteredData);
  const W = 500, H = 350;
  const pad = { top: 20, right: 20, bottom: 40, left: 55 };

  let maxVal = 0;
  REGIONS.forEach(r => MONTH_ORDER.forEach(m => { if (data[r][m] > maxVal) maxVal = data[r][m]; }));
  maxVal = Math.max(maxVal, 1);

  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const xScale = (i) => pad.left + (i / (MONTH_ORDER.length - 1)) * chartW;
  const yScale = (v) => pad.top + chartH - (v / maxVal) * chartH;

  let html = "";

  for (let i = 0; i <= 3; i++) {
    const y = pad.top + (i / 3) * chartH;
    html += `<line x1="${pad.left}" y1="${y}" x2="${W - pad.right}" y2="${y}" stroke="#e2e5ec" stroke-width="1"/>`;
  }

  REGIONS.forEach(reg => {
    const points = MONTH_ORDER.map((m, i) => `${xScale(i)},${yScale(data[reg][m])}`).join(" ");
    html += `<polyline points="${points}" fill="none" stroke="${REG_COLORS[reg]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="region-line" data-region="${reg}"/>`;
  });

  ["Jan","Mar","May","Jul","Sep","Nov"].forEach((m, i) => {
    const idx = MONTH_ORDER.indexOf(m);
    html += `<text x="${xScale(idx)}" y="${H - 14}" text-anchor="middle" font-size="9" fill="#6b7280">${m}</text>`;
  });

  let lx = pad.left;
  const ly = 14;
  REGIONS.forEach(reg => {
    html += `<rect x="${lx}" y="${ly - 7}" width="10" height="10" rx="2" fill="${REG_COLORS[reg]}"/>`;
    html += `<text x="${lx + 14}" y="${ly + 1}" font-size="9" fill="#6b7280">${reg.split(" ")[0]}</text>`;
    lx += 60 + reg.split(" ")[0].length * 6;
    if (lx > W - 40) lx = pad.left;
  });

  svg.innerHTML = html;
}

// ---------- Chart 4: Category Bar ----------
function renderCategoryBar() {
  const svg = document.getElementById("category-bar");
  const data = categoryRevenue(filteredData);
  const W = 500, H = 350;
  const pad = { top: 20, right: 30, bottom: 50, left: 100 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const maxVal = Math.max(...Object.values(data), 1);
  const barH = chartH / CATEGORIES.length * 0.7;
  const gap = chartH / CATEGORIES.length * 0.3;

  let html = "";

  CATEGORIES.forEach((cat, i) => {
    const val = data[cat];
    const barW = (val / maxVal) * chartW;
    const y = pad.top + i * (barH + gap) + gap/2;
    const pct = (val / Object.values(data).reduce((a,b) => a+b, 0) * 100).toFixed(1);

    html += `<rect x="${pad.left}" y="${y}" width="${barW}" height="${barH}" rx="4" fill="${CAT_COLORS[cat]}" class="cat-bar" data-category="${cat}" data-value="${fmt(val)} (${pct}%)"/>`;
    html += `<text x="${pad.left - 8}" y="${y + barH/2 + 4}" text-anchor="end" font-size="11" fill="#1a1a2e">${cat}</text>`;
    html += `<text x="${pad.left + barW + 6}" y="${y + barH/2 + 4}" font-size="10" fill="#6b7280">${fmt(val)}</text>`;
  });

  svg.innerHTML = html;

  svg.addEventListener("mouseover", (e) => {
    if (e.target.classList.contains("cat-bar")) {
      showTooltip(e, `${e.target.dataset.category}: ${e.target.dataset.value}`);
    }
  });
  svg.addEventListener("mouseout", (e) => {
    if (e.target.classList.contains("cat-bar")) hideTooltip();
  });
}

// ---------- Chart 5: Margin Bar ----------
function renderMarginBar() {
  const svg = document.getElementById("margin-bar");
  const data = categoryMargin(filteredData);
  const W = 500, H = 350;
  const pad = { top: 20, right: 30, bottom: 50, left: 100 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const maxVal = 0.8;
  const barH = chartH / CATEGORIES.length * 0.7;
  const gap = chartH / CATEGORIES.length * 0.3;

  let html = "";

  CATEGORIES.forEach((cat, i) => {
    const val = data[cat];
    const barW = (val / maxVal) * chartW;
    const y = pad.top + i * (barH + gap) + gap/2;

    html += `<rect x="${pad.left}" y="${y}" width="${barW}" height="${barH}" rx="4" fill="${CAT_COLORS[cat]}" opacity="0.7" class="margin-bar" data-category="${cat}" data-value="${fmtPct(val)}"/>`;
    html += `<text x="${pad.left - 8}" y="${y + barH/2 + 4}" text-anchor="end" font-size="11" fill="#1a1a2e">${cat}</text>`;
    html += `<text x="${pad.left + barW + 6}" y="${y + barH/2 + 4}" font-size="10" fill="#6b7280">${fmtPct(val)}</text>`;
  });

  svg.innerHTML = html;

  svg.addEventListener("mouseover", (e) => {
    if (e.target.classList.contains("margin-bar")) {
      showTooltip(e, `${e.target.dataset.category}: ${e.target.dataset.value}`);
    }
  });
  svg.addEventListener("mouseout", (e) => {
    if (e.target.classList.contains("margin-bar")) hideTooltip();
  });
}

// ---------- Chart 6: Heatmap ----------
function renderHeatmap() {
  const svg = document.getElementById("heatmap-chart");
  const data = regionCategoryRevenue(filteredData);
  const W = 700, H = 400;
  const pad = { top: 60, left: 130, right: 20, bottom: 20 };

  const cellW = (W - pad.left - pad.right) / CATEGORIES.length;
  const cellH = (H - pad.top - pad.bottom) / REGIONS.length;

  let maxVal = 0;
  REGIONS.forEach(r => CATEGORIES.forEach(c => { if (data[r][c] > maxVal) maxVal = data[r][c]; }));
  maxVal = Math.max(maxVal, 1);

  let html = "";

  CATEGORIES.forEach((cat, i) => {
    html += `<text x="${pad.left + i * cellW + cellW/2}" y="${pad.top - 20}" text-anchor="middle" font-size="11" fill="#1a1a2e" font-weight="600">${cat}</text>`;
  });

  REGIONS.forEach((reg, ri) => {
    html += `<text x="${pad.left - 10}" y="${pad.top + ri * cellH + cellH/2 + 4}" text-anchor="end" font-size="11" fill="#1a1a2e" font-weight="600">${reg}</text>`;

    CATEGORIES.forEach((cat, ci) => {
      const val = data[reg][cat];
      const intensity = val / maxVal;
      const r = Math.round(66 + (67 - 66) * intensity);
      const g = Math.round(97 + (97 - 97) * intensity);
      const b = Math.round(238 + (255 - 238) * intensity);
      const color = `rgb(${r},${g},${b})`;
      const textColor = intensity > 0.5 ? "#fff" : "#1a1a2e";
      const x = pad.left + ci * cellW;
      const y = pad.top + ri * cellH;

      html += `<rect x="${x}" y="${y}" width="${cellW - 2}" height="${cellH - 2}" rx="4" fill="${color}" class="heat-cell" data-region="${reg}" data-category="${cat}" data-value="${fmt(val)}"/>`;
      html += `<text x="${x + (cellW - 2)/2}" y="${y + (cellH - 2)/2 + 4}" text-anchor="middle" font-size="10" fill="${textColor}" font-weight="600">${fmt(val)}</text>`;
    });
  });

  svg.innerHTML = html;

  svg.addEventListener("mouseover", (e) => {
    if (e.target.classList.contains("heat-cell")) {
      showTooltip(e, `${e.target.dataset.region} · ${e.target.dataset.category}: ${e.target.dataset.value}`);
    }
  });
  svg.addEventListener("mouseout", (e) => {
    if (e.target.classList.contains("heat-cell")) hideTooltip();
  });
}
// ===== Tooltip =====
let tooltipEl = null;
function getTooltip() {
  if (!tooltipEl) {
    tooltipEl = document.createElement("div");
    tooltipEl.className = "tooltip";
    document.body.appendChild(tooltipEl);
  }
  return tooltipEl;
}
function showTooltip(e, text) {
  const t = getTooltip();
  t.textContent = text;
  t.style.opacity = 1;
  const rect = e.target.getBoundingClientRect();
  t.style.left = (rect.left + rect.width / 2 - t.offsetWidth / 2) + "px";
  t.style.top = (rect.top - t.offsetHeight - 8) + "px";
}
function hideTooltip() {
  const t = getTooltip();
  t.style.opacity = 0;
}

// ===== Data Table =====
function renderTable() {
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";
  filteredData.forEach(d => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${d.date}</td><td>${d.region}</td><td>${d.category}</td><td>${d.units}</td><td>$${d.revenue.toFixed(2)}</td><td>${(d.gross_margin * 100).toFixed(1)}%</td>`;
    tbody.appendChild(tr);
  });
}

// ===== KPI Updates =====
function updateKPIs() {
  const totalRev = filteredData.reduce((s, d) => s + d.revenue, 0);
  const totalUnits = filteredData.reduce((s, d) => s + d.units, 0);

  const card1 = document.querySelector('[data-testid="kpi-total-revenue"] .kpi-value');
  const card2 = document.querySelector('[data-testid="kpi-total-units"] .kpi-value');
  if (card1) { card1.textContent = fmt(totalRev); card1.parentElement.dataset.value = totalRev.toFixed(2); }
  if (card2) { card2.textContent = totalUnits.toLocaleString(); card2.parentElement.dataset.value = totalUnits; }

  // Category breakdown
  const catRev = categoryRevenue(filteredData);
  const bestCat = Object.entries(catRev).sort((a, b) => b[1] - a[1])[0];
  const catCard = document.querySelector('[data-testid="kpi-best-category"]');
  if (catCard && bestCat) {
    const pct = (bestCat[1] / totalRev * 100).toFixed(1);
    catCard.querySelector(".kpi-value").textContent = bestCat[0];
    catCard.querySelector(".kpi-sub").textContent = fmt(bestCat[1]) + " (" + pct + "%)";
    catCard.dataset.value = bestCat[0];
  }

  // Region breakdown
  const regRev = regionRevenue(filteredData);
  const bestReg = Object.entries(regRev).sort((a, b) => b[1] - a[1])[0];
  const regCard = document.querySelector('[data-testid="kpi-best-region"]');
  if (regCard && bestReg) {
    const pct = (bestReg[1] / totalRev * 100).toFixed(1);
    regCard.querySelector(".kpi-value").textContent = bestReg[0];
    regCard.querySelector(".kpi-sub").textContent = fmt(bestReg[1]) + " (" + pct + "%)";
    regCard.dataset.value = bestReg[0];
  }

  // Best margin
  const margins = categoryMargin(filteredData);
  const bestMargin = Object.entries(margins).sort((a, b) => b[1] - a[1])[0];
  const marginCard = document.querySelector('[data-testid="kpi-highest-margin"]');
  if (marginCard && bestMargin) {
    marginCard.querySelector(".kpi-value").textContent = bestMargin[0];
    marginCard.querySelector(".kpi-sub").textContent = fmtPct(bestMargin[1]) + " avg gross margin";
    marginCard.dataset.value = bestMargin[1].toFixed(3);
  }

  // Update headline
  const totalAll = rawData.reduce((s, d) => s + d.revenue, 0);
  const revPct = totalAll ? (totalRev / totalAll * 100).toFixed(1) : 0;
  const headline2 = document.querySelector("#headline h2");
  if (headline2) {
    const monthsShowing = new Set(filteredData.map(d => d.month));
    const isFiltered = state.region !== "all" || state.category !== "all";
    if (isFiltered) {
      headline2.textContent = `📊 Filtered View: ${fmt(totalRev)} (${revPct}% of total) — ${filteredData.length} records shown`;
    } else {
      headline2.textContent = `📈 Revenue surged 58% from February to December — Subscriptions lead profitability`;
    }
  }
}

// ===== Render All =====
function renderAll() {
  renderMonthlyChart();
  renderRegionDonut();
  renderRegionTrend();
  renderCategoryBar();
  renderMarginBar();
  renderHeatmap();
  renderTable();
  updateKPIs();
}

// ===== Init =====
document.addEventListener("DOMContentLoaded", () => {
  // Filter handlers
  document.getElementById("region-filter").addEventListener("change", (e) => {
    state.region = e.target.value;
    applyFilters();
  });
  document.getElementById("category-filter").addEventListener("change", (e) => {
    state.category = e.target.value;
    applyFilters();
  });

  loadData();
});
