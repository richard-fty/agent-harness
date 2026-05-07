import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const expected = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "expected_metrics.json"), "utf-8"),
);

test("[runtime] app loads without console errors", async ({ page }) => {
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  await page.goto("/");
  await expect(page.getByRole("main")).toBeVisible();
  expect(errors).toEqual([]);
});

test("[data] total revenue KPI matches source data", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("kpi-total-revenue")).toHaveAttribute(
    "data-value",
    String(expected.total_revenue),
  );
});

test("[data] monthly chart has 12 buckets and December value", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("monthly-revenue-chart")).toBeVisible();
  await expect(page.getByTestId("monthly-revenue-series")).toHaveAttribute(
    "data-points",
    String(expected.monthly_buckets),
  );
  await expect(page.getByTestId("month-Dec")).toHaveAttribute(
    "data-value",
    String(expected.december_revenue),
  );
});

test("[data] top region and category claims match expected metrics", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("top-region")).toHaveAttribute("data-value", expected.top_region);
  await expect(page.getByTestId("top-category")).toHaveAttribute("data-value", expected.top_category);
});

test("[viz] report includes primary chart, breakdowns, labels, and data notes", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /what drove 2025 revenue growth/i })).toBeVisible();
  await expect(page.getByTestId("monthly-revenue-chart")).toHaveAttribute("aria-label", /monthly revenue/i);
  await expect(page.getByTestId("region-breakdown-chart")).toBeVisible();
  await expect(page.getByTestId("category-breakdown-chart")).toBeVisible();
  await expect(page.getByText(/revenue is summed from sales.csv/i)).toBeVisible();
});

test("[viz] region filter recomputes KPI values", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Region", { exact: true }).selectOption("North");
  await expect(page.getByTestId("kpi-total-revenue")).toHaveAttribute(
    "data-value",
    String(expected.north_revenue),
  );
});

test("[story] headline names the correct growth driver", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("headline-insight")).toContainText(expected.top_region);
  await expect(page.getByTestId("headline-insight")).toContainText(expected.top_category);
});

test("[story] mobile layout has no horizontal overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
});
