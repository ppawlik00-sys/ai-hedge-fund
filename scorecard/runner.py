"""Runner: executes ai-hedge-fund agents and saves results to scorecard DB.

Usage:
    python -m scorecard.runner --tickers PLTR,GOOGL,SPY
    python -m scorecard.runner --watchlist          # uses default watchlist
    python -m scorecard.runner --tickers PLTR --model deepseek-chat --provider DeepSeek
"""

import argparse
import time
import uuid
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from scorecard.database import save_run, save_agent_signal, save_price_snapshot

# Default watchlist (same as tradeBOT)
DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "DIA", "IWM",
    "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "META", "AAPL",
    "XLE", "XLV", "XLI", "XLC", "XLK", "XLF", "XLP",
    "TLT", "GLD", "USO",
    "PLTR", "AMD", "ADBE", "DDOG",
]

# All available analysts
ALL_ANALYSTS = [
    "warren_buffett",
    "charlie_munger",
    "michael_burry",
    "ben_graham",
    "peter_lynch",
    "phil_fisher",
    "cathie_wood",
    "stanley_druckenmiller",
    "bill_ackman",
    "mohnish_pabrai",
    "rakesh_jhunjhunwala",
    "aswath_damodaran",
    "technical_analyst",
    "fundamentals_analyst",
    "sentiment_analyst",
    "growth_analyst",
    "valuation_analyst",
]

# Recommended subset (cost-effective)
CORE_ANALYSTS = [
    "warren_buffett",
    "michael_burry",
    "stanley_druckenmiller",
    "technical_analyst",
    "fundamentals_analyst",
    "sentiment_analyst",
    "growth_analyst",
    "valuation_analyst",
]


def signal_to_score(signal: str, confidence: float) -> float:
    """Convert bullish/bearish/neutral + confidence to -1.0 to +1.0 score.

    Bullish:  confidence/100  → +0.0 to +1.0
    Neutral:  0.0 (agent has no directional view)
    Bearish: -confidence/100  → -1.0 to -0.0
    """
    conf_normalized = min(confidence, 100) / 100
    if signal == "bullish":
        return round(conf_normalized, 4)
    elif signal == "bearish":
        return round(-conf_normalized, 4)
    else:
        return 0.0


def run(tickers: list[str], agents: list[str],
        model_name: str = "claude-haiku-4-5-20251001",
        model_provider: str = "Anthropic") -> dict:
    """Run ai-hedge-fund agents and save results."""

    # Disable Rich progress display (crashes on Windows cp1250)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    from src.utils.progress import progress
    progress.start = lambda: None
    progress.stop = lambda: None
    progress.update_status = lambda *a, **k: None

    from src.main import run_hedge_fund

    run_id = str(uuid.uuid4())[:8]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    portfolio = {"cash": 100000, "positions": {}}

    print(f"[{run_id}] Starting run: {len(tickers)} tickers × {len(agents)} agents ({model_name})")
    t0 = time.time()

    try:
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=False,
            selected_analysts=agents,
            model_name=model_name,
            model_provider=model_provider,
        )
    except Exception as e:
        print(f"[{run_id}] ERROR: {e}")
        return {"error": str(e)}

    duration_ms = (time.time() - t0) * 1000

    # Extract analyst signals from result
    analyst_signals = {}
    if result:
        analyst_signals = result.get("analyst_signals", {})

    # Save run metadata
    save_run(run_id, tickers, agents, model_name, model_provider, duration_ms)

    # Save per-agent per-ticker signals
    signals_saved = 0
    for agent_id, ticker_signals in analyst_signals.items():
        # Clean agent name (remove "_agent" suffix for display)
        agent_display = agent_id.replace("_agent", "").replace("_analyst", "")

        for ticker, signal_data in ticker_signals.items():
            if not isinstance(signal_data, dict):
                continue

            sig = signal_data.get("signal", "neutral")
            conf = signal_data.get("confidence", 0)
            reasoning = signal_data.get("reasoning", "")
            score = signal_to_score(sig, conf)

            save_agent_signal(
                run_id=run_id,
                ticker=ticker,
                agent_name=agent_display,
                signal=sig,
                confidence=conf,
                reasoning=reasoning,
                score=score,
            )
            signals_saved += 1

    # Save price snapshots
    try:
        from src.tools.api import get_prices
        for ticker in tickers:
            prices = get_prices(ticker, end_date, end_date)
            if prices:
                save_price_snapshot(ticker, prices[-1].close)
    except Exception as e:
        print(f"[{run_id}] Price snapshot warning: {e}")

    print(f"[{run_id}] Done: {signals_saved} signals saved ({duration_ms:.0f}ms)")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RUN {run_id} SUMMARY")
    print(f"{'='*60}")

    for ticker in tickers:
        scores = {}
        for agent_id, ticker_signals in analyst_signals.items():
            agent_display = agent_id.replace("_agent", "").replace("_analyst", "")
            sig_data = ticker_signals.get(ticker, {})
            if isinstance(sig_data, dict) and sig_data.get("signal"):
                score = signal_to_score(sig_data["signal"], sig_data.get("confidence", 0))
                scores[agent_display] = score

        if scores:
            avg_score = sum(scores.values()) / len(scores)
            scores_str = " | ".join(f"{k}:{v:+.2f}" for k, v in sorted(scores.items()))
            print(f"{ticker:6s} | avg={avg_score:+.3f} | {scores_str}")

    return {
        "run_id": run_id,
        "signals_saved": signals_saved,
        "duration_ms": duration_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ai-hedge-fund scorecard")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    parser.add_argument("--watchlist", action="store_true", help="Use default watchlist")
    parser.add_argument("--agents", type=str, default="core",
                        choices=["core", "all"], help="Agent set (core=8, all=17)")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--provider", type=str, default="Anthropic")
    args = parser.parse_args()

    if args.watchlist:
        tickers = DEFAULT_WATCHLIST
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        print("Specify --tickers PLTR,GOOGL or --watchlist")
        sys.exit(1)

    agents = ALL_ANALYSTS if args.agents == "all" else CORE_ANALYSTS

    run(tickers, agents, args.model, args.provider)


if __name__ == "__main__":
    main()
