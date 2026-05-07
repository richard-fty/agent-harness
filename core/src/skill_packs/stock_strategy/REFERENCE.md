# Stock Analysis Reference

## Technical Indicators

### RSI (Relative Strength Index)
- Range: 0-100
- Oversold: < 30 (potential buy)
- Overbought: > 70 (potential sell)
- Formula: RSI = 100 - (100 / (1 + RS)), where RS = avg gain / avg loss over N periods
- Default period: 14

### MACD (Moving Average Convergence Divergence)
- MACD Line = EMA(12) - EMA(26)
- Signal Line = EMA(9) of MACD Line
- Histogram = MACD Line - Signal Line
- Buy signal: MACD crosses above Signal (bullish crossover)
- Sell signal: MACD crosses below Signal (bearish crossover)

### Bollinger Bands
- Middle Band = SMA(20)
- Upper Band = SMA(20) + 2 * StdDev(20)
- Lower Band = SMA(20) - 2 * StdDev(20)
- Price touching lower band in uptrend = potential buy
- Price touching upper band in downtrend = potential sell
- Band squeeze (narrow bands) = volatility expansion expected

### Moving Averages
- SMA(N) = Simple Moving Average over N periods
- EMA(N) = Exponential Moving Average (more weight on recent)
- Golden Cross: SMA(50) crosses above SMA(200) — bullish
- Death Cross: SMA(50) crosses below SMA(200) — bearish

## Strategy Metrics

### Sharpe Ratio
- (Return - Risk-Free Rate) / Standard Deviation of Return
- \> 1.0 = acceptable, > 2.0 = good, > 3.0 = excellent

### Max Drawdown
- Largest peak-to-trough decline
- < 10% = conservative, 10-20% = moderate, > 20% = aggressive

### Win Rate
- Number of winning trades / total trades
- > 50% with positive expectancy = viable strategy

## Yahoo Finance Symbols
- US Stocks: AAPL, TSLA, GOOG, MSFT, AMZN, NVDA, META
- Crypto: BTC-USD, ETH-USD, SOL-USD, DOGE-USD
- Indices: ^SPX, ^DJI, ^IXIC
- ETFs: SPY, QQQ, IWM, GLD, TLT

## Period / Interval Reference
- Intraday (1m-15m intervals): max 7 days of data
- Hourly (1h interval): max 730 days
- Daily (1d interval): unlimited history
- Weekly/Monthly: unlimited history

## Query Construction

When using `web_research` for stocks, keep each query simple. The backend does not rewrite stock web queries; the query shown in `queries_used` should match the tool input after whitespace normalization.

### Query Rules
- Prefer the company name over the ticker when possible
- Use `"<Company> stock ticker"` only when ticker lookup is needed
- Use `"<Company> latest news"` for general company news
- For source-specific context, use short source queries such as `"<Company> Reuters"`, `"<Company> CNBC"`, `"<Company> Wall Street Journal"`, `"<Company> CNN"`, `"<Company> New York Times"`, or `"<Company> Washington Post"`
- Do not add a year unless the user explicitly asks for that year
- Request `num_results=5`, `fetch_top=0`, `topic="news"`, and `time_range="week"` for company news
- If the response includes `queries_used`, treat that as the actual query set that ran

### Good Query Patterns
- `<Company> stock ticker`
- `<Company> latest news`
- `<Company> Reuters`
- `<Company> CNBC`
- `<Company> Wall Street Journal`

### Weak Query Patterns
- `<Ticker> stock analysis 2026`
- `<Ticker> financial performance`
- `<Ticker> recent information`
- `<Ticker> news earnings financial performance 2026`
- `<Company> latest news latest news 2025`
- `<Company> latest earnings`
- `<Company> stock catalyst risk`
- `<Company> stock news`

## Default Backtest Strategy

Use this strategy when the user asks for a stock report and does not provide custom rules.

```python
def signal(row, prev_row):
    bullish_trend = row["close"] > row["sma_20"] and row["sma_20"] > row["sma_50"]
    bullish_momentum = row["macd"] > row["macd_signal"] and row["rsi"] < 70
    bearish_exit = (
        row["close"] < row["sma_20"]
        or row["macd"] < row["macd_signal"]
        or row["rsi"] > 75
    )

    if bullish_trend and bullish_momentum:
        return "BUY"
    if bearish_exit:
        return "SELL"
    return "HOLD"
```

### Strategy Intent
- Trend filter: price must be above SMA20, and SMA20 must be above SMA50
- Momentum filter: MACD must be above signal, and RSI must not already be overbought
- Exit: price loses SMA20, MACD turns bearish, or RSI becomes stretched above 75

### Required Backtest Metrics
- Strategy return
- Buy-and-hold return
- Alpha versus buy-and-hold
- Sharpe ratio
- Max drawdown
- Win rate
- Total trades

### Interpretation Rules
- Prefer the strategy only if it beats buy-and-hold, has positive Sharpe, controlled drawdown, and enough trades to be meaningful
- If strategy return is lower than buy-and-hold, say the backtest does not support the strategy even when current indicators look bullish
- If trade count is zero or very low, say the rule did not trigger enough to form a strong conclusion
- Never hide weak backtest results behind bullish indicator language

## Strategy Report Outline

Use this when the user asks for a written stock report. Treat it as content guidance only.
Do not assume the final output format unless the user explicitly asks for one.

```md
# NVDA Strategy-Backed Stock Report

## Executive Summary
Two or three short paragraphs on the current signal, strategy backtest result, and main risks.

## Market Snapshot

| Metric | Value | Takeaway |
|---|---:|---|
| Latest Close | $... | ... |
| 1Y Change | ...% | ... |
| RSI(14) | ... | ... |
| MACD | ... | ... |
| Bollinger Bands | ... | ... |
| SMA(20) | $... | ... |
| SMA(50) | $... | ... |

## Strategy Rules

| Rule | Condition |
|---|---|
| Buy | close > SMA20, SMA20 > SMA50, MACD > signal, RSI < 70 |
| Sell | close < SMA20, or MACD < signal, or RSI > 75 |

## Backtest Results

| Metric | Value | Takeaway |
|---|---:|---|
| Strategy Return | ...% | ... |
| Buy-and-Hold Return | ...% | ... |
| Alpha | ...% | ... |
| Sharpe Ratio | ... | ... |
| Max Drawdown | ...% | ... |
| Win Rate | ...% | ... |
| Total Trades | ... | ... |

## Indicator Read

| Indicator | Value | Signal |
|---|---|---|
| RSI(14) | ... | ... |
| MACD | ... | ... |
| Bollinger Bands | ... | ... |
| SMA20/SMA50 | ... | ... |

## News From Web Search

| Source | Key Point |
|---|---|
| [Headline 1](https://example.com/1) | One sentence takeaway |
| [Headline 2](https://example.com/2) | One sentence takeaway |
| [Headline 3](https://example.com/3) | One sentence takeaway |

## Final Assessment

State whether the backtested strategy supports, weakens, or contradicts the current indicator setup.

## Risks And Limits

| Risk | Why It Matters |
|---|---|
| Key risk 1 | ... |
| Key risk 2 | ... |
```
