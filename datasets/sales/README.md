# Sales test dataset

Synthetic monthly sales data for ad-hoc testing of the data-viz agent. Not used by any eval case.

## File

`sales.csv` — 240 rows (12 months × 4 regions × 5 categories).

## Schema

| column | type | description |
|---|---|---|
| `date` | ISO date | first day of the month |
| `year` | int | calendar year (all 2025) |
| `month` | string | three-letter month abbreviation |
| `region` | string | one of `North America`, `EMEA`, `APAC`, `LATAM` |
| `category` | string | one of `Platform`, `Services`, `Hardware`, `Subscriptions`, `Training` |
| `units` | int | units sold |
| `revenue` | float | revenue in USD |
| `gross_margin` | float | gross margin as a fraction (0–1) |

## Built-in patterns

- **Growth trend**: ~1.8% month-over-month.
- **Seasonality**: dips in Feb and Aug, peaks in Jun and Nov/Dec.
- **Region weighting**: North America > EMEA > APAC > LATAM.
- **Region × category interactions** (good signal for narrative claims):
  - APAC over-indexes on Hardware (×1.35)
  - EMEA over-indexes on Subscriptions (×1.25)
  - North America over-indexes on Platform (×1.15)
  - LATAM under-indexes on Platform (×0.65)
- **Per-row noise**: ±8% on units, ±5% on price, ±0.04 on margin.

## Reference values

Generated with `random.seed(42)`. Totals:

- Total revenue: **$4,311,320.46**
- Top region: **North America** ($1,655,650.47)
- Top category: **Platform** ($1,564,588.35)

## Suggested test prompts

- "Build an interactive sales story for 2025. Show monthly revenue, revenue by region, top product categories, and a region filter."
- "Which region drove the most revenue and which category contributed the most to it?"
- "Compare gross margin across categories and explain which mix shifts would lift overall margin."
