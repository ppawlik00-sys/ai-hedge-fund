"""SQLite database for ai-hedge-fund scorecard."""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "hedge_fund.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            reasoning TEXT,
            score REAL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            tickers TEXT NOT NULL,
            agents TEXT NOT NULL,
            model_name TEXT,
            model_provider TEXT,
            num_tickers INTEGER,
            num_agents INTEGER,
            duration_ms REAL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            captured_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            signal TEXT NOT NULL,
            score REAL,
            price_at_signal REAL,
            price_1d REAL,
            price_3d REAL,
            price_5d REAL,
            return_1d REAL,
            return_3d REAL,
            return_5d REAL,
            correct_1d INTEGER,
            correct_3d INTEGER,
            correct_5d INTEGER,
            evaluated_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_signals_ticker ON agent_signals(ticker);
        CREATE INDEX IF NOT EXISTS idx_signals_agent ON agent_signals(agent_name);
        CREATE INDEX IF NOT EXISTS idx_signals_run ON agent_signals(run_id);
        CREATE INDEX IF NOT EXISTS idx_signals_created ON agent_signals(created_at);
        CREATE INDEX IF NOT EXISTS idx_outcomes_agent ON outcomes(agent_name);
        CREATE INDEX IF NOT EXISTS idx_outcomes_ticker ON outcomes(ticker);
        CREATE INDEX IF NOT EXISTS idx_prices_ticker ON price_snapshots(ticker, captured_at);
    """)
    conn.commit()
    conn.close()


def save_run(run_id: str, tickers: list, agents: list,
             model_name: str, model_provider: str, duration_ms: float):
    conn = get_db()
    conn.execute(
        "INSERT INTO run_log (run_id, tickers, agents, model_name, model_provider, "
        "num_tickers, num_agents, duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (run_id, json.dumps(tickers), json.dumps(agents), model_name, model_provider,
         len(tickers), len(agents), duration_ms,
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def save_agent_signal(run_id: str, ticker: str, agent_name: str,
                      signal: str, confidence: float, reasoning: str,
                      score: float | None = None):
    conn = get_db()
    conn.execute(
        "INSERT INTO agent_signals (run_id, ticker, agent_name, signal, confidence, "
        "reasoning, score, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (run_id, ticker, agent_name, signal, confidence,
         reasoning if isinstance(reasoning, str) else json.dumps(reasoning),
         score, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def save_price_snapshot(ticker: str, price: float):
    conn = get_db()
    conn.execute(
        "INSERT INTO price_snapshots (ticker, price, captured_at) VALUES (?,?,?)",
        (ticker, price, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def get_latest_signals(limit: int = 100) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT s.*, r.model_name FROM agent_signals s "
        "JOIN run_log r ON s.run_id = r.run_id "
        "ORDER BY s.created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_signals_by_run(run_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM agent_signals WHERE run_id = ? ORDER BY ticker, agent_name",
        (run_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_runs(limit: int = 20) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM run_log ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_agent_history(agent_name: str, ticker: str = None,
                      days: int = 30) -> list[dict]:
    conn = get_db()
    query = "SELECT * FROM agent_signals WHERE agent_name = ?"
    params = [agent_name]
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    query += " AND created_at >= datetime('now', ?)"
    params.append(f"-{days} days")
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize on import
init_db()
