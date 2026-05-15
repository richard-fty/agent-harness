import csv
from collections import defaultdict

rows = []
with open("public/sales.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["revenue"] = float(r["revenue"])
        r["units"] = int(r["units"])
        r["gross_margin"] = float(r["gross_margin"])
        rows.append(r)

# Total revenue
total_rev = sum(r["revenue"] for r in rows)
print(f"Total Revenue: ${total_rev:,.2f}")

# Total units
total_units = sum(r["units"] for r in rows)
print(f"Total Units: {total_units:,}")

# Revenue by region
region_rev = defaultdict(float)
for r in rows:
    region_rev[r["region"]] += r["revenue"]
print("\nRevenue by Region:")
for reg, rev in sorted(region_rev.items(), key=lambda x: -x[1]):
    print(f"  {reg}: ${rev:,.2f} ({rev/total_rev*100:.1f}%)")

# Revenue by category
cat_rev = defaultdict(float)
for r in rows:
    cat_rev[r["category"]] += r["revenue"]
print("\nRevenue by Category:")
for cat, rev in sorted(cat_rev.items(), key=lambda x: -x[1]):
    print(f"  {cat}: ${rev:,.2f} ({rev/total_rev*100:.1f}%)")

# Monthly revenue
month_rev = defaultdict(float)
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for r in rows:
    month_rev[r["month"]] += r["revenue"]
print("\nMonthly Revenue:")
for m in month_order:
    print(f"  {m}: ${month_rev[m]:,.2f}")

# Avg gross margin by category
cat_margin = defaultdict(list)
for r in rows:
    cat_margin[r["category"]].append(r["gross_margin"])
print("\nAvg Gross Margin by Category:")
for cat, margins in sorted(cat_margin.items(), key=lambda x: -sum(x[1])/len(x[1])):
    avg = sum(margins)/len(margins)
    print(f"  {cat}: {avg:.1%}")

# Revenue by region and category
reg_cat = defaultdict(lambda: defaultdict(float))
for r in rows:
    reg_cat[r["region"]][r["category"]] += r["revenue"]
print("\nRegion x Category Revenue:")
for reg in ["North America", "EMEA", "APAC", "LATAM"]:
    print(f"\n  {reg}:")
    for cat, rev in sorted(reg_cat[reg].items(), key=lambda x: -x[1]):
        print(f"    {cat}: ${rev:,.2f}")

# Best month
print(f"\nBest Month by Revenue: {max(month_rev, key=month_rev.get)} (${max(month_rev.values()):,.2f})")
print(f"Worst Month: {min(month_rev, key=month_rev.get)} (${min(month_rev.values()):,.2f})")
