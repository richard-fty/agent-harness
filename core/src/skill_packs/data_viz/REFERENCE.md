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
