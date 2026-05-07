# 📊 GLD (SPDR Gold Shares ETF) — Strategy-Backed Analysis Report

**Report Date:** April 28, 2026  
**Ticker:** GLD  
**Asset Class:** Gold ETF (Physical Gold)  
**Analysis Type:** Technical Analysis + Strategy Backtest

---

## 1. 📈 Executive Summary

| Metric | Value |
|---|---|
| **Current Price** | $420.90 |
| **1-Year High** | $509.70 |
| **1-Year Low** | $291.78 |
| **1-Year Change** | **+36.18%** |
| **Avg Daily Volume** | 13.7M shares |

Gold has had a powerful bull run over the last year, rising over 36%. However, the ETF has pulled back significantly from its **$509.70 high**, currently trading ~17% below the peak. The recent decline has accelerated in the last few sessions, with GLD dropping from $440 to $420 in under a week.

---

## 2. 🏷️ Market Snapshot

| Detail | Value |
|---|---|
| Open (Today) | $420.72 |
| High (Today) | $421.72 |
| Low (Today) | $418.40 |
| Close (Today) | $420.90 |
| Volume | 5,270,957 |
| SMA(50) | $446.06 |
| EMA(50) | $437.45 |

---

## 3. 🛠️ Technical Indicators

### RSI (14) — **39.88** ⚠️ Neutral-to-Bearish
The RSI has fallen from the mid-50s to **39.88**, approaching oversold territory. While not yet below 30 (the traditional oversold threshold), the steep 1-week decline signals **strong selling pressure**. The RSI has been declining for 3 consecutive sessions.

### MACD (12, 26, 9) — **Bearish Crossover 🔴**
- MACD Line: **-2.86**
- Signal Line: **-2.28**
- Histogram: **-0.59**
- **Signal: Bearish crossover (sell signal)** — The MACD crossed below the signal line, and both are in negative territory. This is a confirmed bearish signal suggesting downside momentum.

### Bollinger Bands (20, 2σ)
| Band | Value |
|---|---|
| Upper Band | $447.24 |
| Middle Band | $434.77 |
| Lower Band | $422.31 |
| **Price** | **$420.90** |

**🔵 Signal: Price below lower band — Oversold / Potential Bounce**

The price has broken below the lower Bollinger Band ($422.31), which can indicate an **oversold condition**. Historically, this often precedes a short-term bounce or consolidation, though it can also signal the start of a sustained downtrend in strong bear markets.

### Moving Averages
- **SMA(50):** $446.06 — Price **5.7% below** ❌ (Bearish)
- **EMA(50):** $437.45 — Price **3.8% below** ❌ (Bearish)

The price is well below both the 50-day SMA and EMA, confirming the **short-term bearish trend**.

---

## 4. 🔄 Backtest Results (Default Trend-Following Strategy)

The strategy uses a trend-following approach:
- **BUY** when: Price > SMA20 > SMA50 (uptrend) + MACD bullish + RSI < 70
- **SELL** when: Price < SMA20 or MACD bearish or RSI > 75

| Metric | Value |
|---|---|
| **Initial Capital** | $10,000 |
| **Final Value** | $9,989.82 |
| **Strategy Return** | **-0.10%** |
| **Buy & Hold Return** | **+37.75%** |
| **Alpha** | **-37.86%** |
| **Sharpe Ratio** | 0.039 |
| **Max Drawdown** | -5.31% |
| **Total Trades** | 20 |
| **Win Rate** | 50.0% |

### 📋 Verdict: **Strategy does NOT beat Buy-and-Hold**

The backtest shows the default strategy **significantly underperformed** a simple buy-and-hold approach over the past year:

- **Buy & Hold returned +37.75%** — capturing the entire gold bull run
- **Strategy returned -0.10%** — the trend-following rules triggered frequent whipsaws (especially early in the move) and exited too early during the rally
- **20 total trades** with a **50% win rate** — the strategy was choppy
- **Sharpe of 0.039** — near risk-free returns, essentially flat
- **Max drawdown of -5.31%** — better than buy-and-hold's drawdown, but at the cost of missing most of the upside

**Key issue:** The strategy was overly sensitive to pullbacks in a strong uptrend, causing premature exits. Gold's rally was powerful but had intermediate shakeouts that triggered the sell rules.

---

## 5. 📰 Fundamental / Macro Context

Gold has seen an extraordinary rally driven by:
- **Global macroeconomic uncertainty and trade tensions**
- **Central bank gold purchases** at record levels
- **Weakening USD** for much of the period
- **Inflation concerns** and real interest rate dynamics

The recent pullback from $509 to $420 likely reflects:
- Profit-taking after a massive run
- Potential strength in the USD or rising real rates
- Technical correction after becoming overextended

---

## 6. 🧠 Final Assessment

### Short-Term View (1–4 weeks) ⚠️ Bearish / Oversold Bounce Possible

| Signal | Direction |
|---|---|
| RSI (39.88) | Neutral / Approaching oversold |
| MACD | 🔴 Bearish Crossover |
| Bollinger Bands | Price below lower band — oversold bounce candidate |
| vs SMA(50) | 🔴 Bearish |
| vs EMA(50) | 🔴 Bearish |

The technical picture is **bearish in the short term**. The MACD sell signal and breakdown below moving averages suggest continued downside risk. However, the break below the lower Bollinger Band raises the possibility of a **short-term bounce**. A move back above $422 (Bollinger lower band) and ideally $435 (mid-band) would be needed to stabilize.

### Medium-Term View (1–6 months) 🤷 Neutral — Awaiting Direction

The trend has broken from its strong uptrend. Key levels to watch:
- **Support:** $418 (today's low), then **$400** (psychological)
- **Resistance:** $434 (mid-Bollinger), **$446** (SMA50), **$509** (all-time high)

A reclamation of the SMA(50) at $446 would signal the uptrend is resuming. Failure at $418 could open a move toward $400.

### Backtest Verdict

The **trend-following strategy does not work well** on gold's volatile but strong uptrend over the past year. Buy-and-hold dramatically outperformed. This suggests that for gold ETFs in a strong bull market, **a simple long hold or a dip-buying approach** is more effective than a trend-following strategy with tight exits.

---

## 7. ⚠️ Risks & Limitations

1. **Backtest limitations:** Past performance does not guarantee future results. The 1-year window captures only one market regime.
2. **Gold is macro-driven:** Gold prices are heavily influenced by USD strength, real interest rates, and geopolitical events — factors not captured in technical-only analysis.
3. **The default strategy** (used in backtest) is a generic trend-follower; custom strategies tailored to gold's volatility patterns could perform differently.
4. **The ETF tracks gold spot** but has a small expense ratio (0.40%) which is not fully modeled.
5. **Liquidity risk:** None — GLD is highly liquid with ~$70B+ AUM.

---

*Report generated by Apex Agent using Yahoo Finance market data, technical indicators, and strategy backtesting. Not financial advice — always do your own research.*
