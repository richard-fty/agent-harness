# Stock & Crypto Analysis

Planning Mode: off

## When to Use
- User asks about stock/crypto prices, analysis, or trading strategies
- User mentions ticker symbols (AAPL, BTC-USD, etc.)
- User wants technical indicators, charts, backtesting, or portfolio analysis

## Workflow
1. Resolve the company and ticker.
   Use common ticker knowledge when possible. If the ticker is unknown or ambiguous, web research may be used once with `"<Company> stock ticker"`.
2. Call Yahoo Finance market data:
   `fetch_market_data(symbol=<TICKER>, period="1y", interval="1d")`
3. Call web research with a simple company query:
   `web_research(query="<Company> latest news", num_results=5, fetch_top=0, topic="news", time_range="week")`
   If source-specific context is needed, keep it short, for example `"<Company> CNBC"` or `"<Company> Reuters"`.
4. Compute indicators:
   - `compute_indicator` RSI with `window=14`
   - `compute_indicator` MACD
   - `compute_indicator` BOLLINGER with `window=20`
   - `compute_indicator` SMA with `window=20`
   - `compute_indicator` SMA with `window=50`
5. Run the default backtest:
   `run_backtest(symbol=<TICKER>, period="1y", initial_capital=10000, strategy_code=<default strategy below>)`
6. Write the final markdown report as an artifact card:
   `write_file(path="<TICKER>_strategy_report.md", content=<markdown report>)`
7. In chat, give a short summary and point the user to the report artifact.

## Default Strategy Script
Use this strategy when the user did not provide custom rules:

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

Default strategy name: `SMA20/SMA50 MACD RSI Trend Strategy`.
Default interpretation:
- Stronger result: strategy return > buy-and-hold, positive Sharpe, controlled max drawdown, enough trades to be meaningful
- Weaker result: strategy return below buy-and-hold, negative alpha, poor Sharpe, high drawdown, or too few trades
- If there are zero or very few trades, say the strategy did not trigger often enough to support a strong conclusion

## Required Report Structure
Use this exact structure for stock reports. Do not call `read_skill_reference` just to get a report outline.

```md
# <Company> (<Ticker>) Strategy-Backed Stock Report

## Executive Summary
Two or three short paragraphs based on current indicators plus backtest results.

## Market Snapshot
| Metric | Value | Takeaway |
|---|---:|---|
| Latest Close | ... | ... |
| 1Y Change | ... | ... |
| RSI(14) | ... | ... |
| MACD | ... | ... |
| Bollinger Bands | ... | ... |
| SMA(20) | ... | ... |
| SMA(50) | ... | ... |

## Strategy Rules
| Rule | Condition |
|---|---|
| Buy | close > SMA20, SMA20 > SMA50, MACD > signal, RSI < 70 |
| Sell | close < SMA20, or MACD < signal, or RSI > 75 |

## Backtest Results
| Metric | Value | Takeaway |
|---|---:|---|
| Strategy Return | ... | ... |
| Buy-and-Hold Return | ... | ... |
| Alpha | ... | ... |
| Sharpe Ratio | ... | ... |
| Max Drawdown | ... | ... |
| Win Rate | ... | ... |
| Total Trades | ... | ... |

## Indicator Read
| Indicator | Signal |
|---|---|
| RSI | ... |
| MACD | ... |
| Bollinger Bands | ... |
| SMA20/SMA50 | ... |

## News From Web Search
| Source | Key Point |
|---|---|
| [Headline](url) | ... |

## Final Assessment
State whether the backtested strategy supports, weakens, or contradicts the current indicator setup.

## Risks And Limits
Include key risks and the past-performance limitation.
```

## Hard Limits
- Planning: do not create or update a todo/checklist plan for stock reports.
- Web research: keep queries short and natural. Do not add extra years, repeated words, or broad phrases like "stock analysis financial performance".
- Default news search: use `topic="news"` and `time_range="week"` with `fetch_top=0`.
- Do not make duplicate web searches when one search already returned useful sources.
- No extra lookup tools: do not call `read_skill_reference`, `recall_session`, or any RAG tool.
- Final output: always call `write_file` with a markdown report artifact; chat response should be a short summary only.
- Evidence: use only real tool results, always include backtest metrics, and never make a final assessment from indicators alone.
- Scope: no PDF, no generated chart, no second web query, no session/RAG indexing unless the user asks for a different non-demo workflow.

## Stock Skill Tools
- `fetch_market_data(symbol, period, interval)` — Yahoo Finance OHLCV data
- `compute_indicator(symbol, indicator, params)` — Technical indicators
- `run_backtest(strategy_code)` — Execute and evaluate the strategy

## Built-In Tools Used By This Skill
- `web_research(query, num_results, fetch_top, topic, time_range)` — Search web/news sources
- `write_file(path, content)` — Create the clickable markdown report artifact
