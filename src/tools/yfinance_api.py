"""Drop-in replacement for api.py using yfinance instead of Financial Datasets API.

All function signatures and return types are identical to api.py.
"""

import datetime
import logging
import os
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)

_cache = get_cache()


# ── Prices ───────────────────────────────────────────────────────────────────

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch daily OHLCV from yfinance."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

    try:
        stock = yf.Ticker(ticker)
        # Add 1 day to end_date because yfinance end is exclusive
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)
        df = stock.history(start=start_date, end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True)

        if df.empty:
            return []

        prices = []
        for idx, row in df.iterrows():
            prices.append(Price(
                open=round(row["Open"], 4),
                close=round(row["Close"], 4),
                high=round(row["High"], 4),
                low=round(row["Low"], 4),
                volume=int(row["Volume"]),
                time=idx.strftime("%Y-%m-%dT%H:%M:%S"),
            ))

        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices
    except Exception as e:
        logger.warning("yfinance get_prices failed for %s: %s", ticker, e)
        return []


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to DataFrame (identical to original)."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    return prices_to_df(get_prices(ticker, start_date, end_date, api_key=api_key))


# ── Financial Metrics ────────────────────────────────────────────────────────

def _safe_get(info: dict, key: str) -> float | None:
    val = info.get(key)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


def _compute_growth(values: list) -> float | None:
    """Compute YoY growth from a list of values (oldest first)."""
    clean = [v for v in values if v is not None and v != 0]
    if len(clean) < 2:
        return None
    return (clean[-1] - clean[-2]) / abs(clean[-2])


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Build FinancialMetrics from yfinance info + financials."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached]

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Get financial statements for growth calculations
        income = stock.quarterly_income_stmt
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow

        # Revenue growth
        revenue_growth = None
        if income is not None and not income.empty:
            rev_row = income.loc["Total Revenue"] if "Total Revenue" in income.index else None
            if rev_row is not None and len(rev_row) >= 2:
                revenue_growth = _compute_growth(rev_row.iloc[::-1].tolist())

        # Earnings growth
        earnings_growth = None
        if income is not None and not income.empty:
            ni_row = income.loc["Net Income"] if "Net Income" in income.index else None
            if ni_row is not None and len(ni_row) >= 2:
                earnings_growth = _compute_growth(ni_row.iloc[::-1].tolist())

        # FCF growth
        fcf_growth = None
        if cashflow is not None and not cashflow.empty:
            fcf_row = cashflow.loc["Free Cash Flow"] if "Free Cash Flow" in cashflow.index else None
            if fcf_row is not None and len(fcf_row) >= 2:
                fcf_growth = _compute_growth(fcf_row.iloc[::-1].tolist())

        # Book value growth
        bv_growth = None
        if balance is not None and not balance.empty:
            eq_row = balance.loc["Stockholders Equity"] if "Stockholders Equity" in balance.index else None
            if eq_row is not None and len(eq_row) >= 2:
                bv_growth = _compute_growth(eq_row.iloc[::-1].tolist())

        # EPS growth
        eps_growth = None
        diluted_eps = info.get("trailingEps")
        forward_eps = info.get("forwardEps")
        if diluted_eps and forward_eps and diluted_eps != 0:
            eps_growth = (forward_eps - diluted_eps) / abs(diluted_eps)

        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=end_date,
            period=period,
            currency="USD",
            market_cap=_safe_get(info, "marketCap"),
            enterprise_value=_safe_get(info, "enterpriseValue"),
            price_to_earnings_ratio=_safe_get(info, "trailingPE"),
            price_to_book_ratio=_safe_get(info, "priceToBook"),
            price_to_sales_ratio=_safe_get(info, "priceToSalesTrailing12Months"),
            enterprise_value_to_ebitda_ratio=_safe_get(info, "enterpriseToEbitda"),
            enterprise_value_to_revenue_ratio=_safe_get(info, "enterpriseToRevenue"),
            free_cash_flow_yield=_safe_get(info, "freeCashflow") / info["marketCap"]
                if info.get("freeCashflow") and info.get("marketCap") else None,
            peg_ratio=_safe_get(info, "pegRatio"),
            gross_margin=_safe_get(info, "grossMargins"),
            operating_margin=_safe_get(info, "operatingMargins"),
            net_margin=_safe_get(info, "profitMargins"),
            return_on_equity=_safe_get(info, "returnOnEquity"),
            return_on_assets=_safe_get(info, "returnOnAssets"),
            return_on_invested_capital=None,  # yfinance doesn't provide ROIC
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=_safe_get(info, "currentRatio"),
            quick_ratio=_safe_get(info, "quickRatio"),
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=_safe_get(info, "debtToEquity") / 100
                if info.get("debtToEquity") else None,  # yfinance returns as %
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=_safe_get(info, "revenueGrowth") or revenue_growth,
            earnings_growth=_safe_get(info, "earningsGrowth") or earnings_growth,
            book_value_growth=bv_growth,
            earnings_per_share_growth=eps_growth,
            free_cash_flow_growth=fcf_growth,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=_safe_get(info, "payoutRatio"),
            earnings_per_share=_safe_get(info, "trailingEps"),
            book_value_per_share=_safe_get(info, "bookValue"),
            free_cash_flow_per_share=_safe_get(info, "freeCashflow") / info["sharesOutstanding"]
                if info.get("freeCashflow") and info.get("sharesOutstanding") else None,
        )

        result = [metrics]
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in result])
        return result
    except Exception as e:
        logger.warning("yfinance get_financial_metrics failed for %s: %s", ticker, e)
        return []


# ── Line Items (Financial Statements) ────────────────────────────────────────

# Mapping from Financial Datasets field names to yfinance statement rows
_INCOME_STMT_MAP = {
    "revenue": "Total Revenue",
    "total_revenue": "Total Revenue",
    "cost_of_revenue": "Cost Of Revenue",
    "gross_profit": "Gross Profit",
    "operating_income": "Operating Income",
    "operating_expense": "Total Operating Expenses",
    "net_income": "Net Income",
    "ebitda": "EBITDA",
    "interest_expense": "Interest Expense",
    "income_tax_expense": "Tax Provision",
    "depreciation_and_amortization": "Depreciation And Amortization",
    "research_and_development": "Research Development",
    "selling_general_and_administrative": "Selling General Administrative",
    "weighted_average_shares_outstanding": "Diluted Average Shares",
    "diluted_earnings_per_share": "Diluted EPS",
    "earnings_per_share": "Basic EPS",
    "dividends_per_share": "Dividends Per Share",
}

_BALANCE_SHEET_MAP = {
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities Net Minority Interest",
    "total_equity": "Stockholders Equity",
    "shareholders_equity": "Stockholders Equity",
    "total_debt": "Total Debt",
    "long_term_debt": "Long Term Debt",
    "short_term_debt": "Short Long Term Debt",
    "cash_and_equivalents": "Cash And Cash Equivalents",
    "cash_and_short_term_investments": "Cash Cash Equivalents And Short Term Investments",
    "total_current_assets": "Current Assets",
    "total_current_liabilities": "Current Liabilities",
    "inventory": "Inventory",
    "accounts_receivable": "Net Receivables",
    "accounts_payable": "Accounts Payable",
    "goodwill": "Goodwill",
    "intangible_assets": "Intangible Assets",
    "outstanding_shares": "Share Issued",
}

_CASHFLOW_MAP = {
    "free_cash_flow": "Free Cash Flow",
    "operating_cash_flow": "Total Cash From Operating Activities",
    "capital_expenditure": "Capital Expenditures",
    "capital_expenditures": "Capital Expenditures",
    "dividends_paid": "Common Stock Dividend Paid",
    "share_repurchases": "Repurchase Of Capital Stock",
    "share_issuance": "Issuance Of Capital Stock",
    "issuance_of_debt": "Issuance Of Debt",
    "repayment_of_debt": "Repayment Of Debt",
    "depreciation_amortization": "Depreciation",
}


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from yfinance financial statements."""
    try:
        stock = yf.Ticker(ticker)

        # Get statements based on period
        if period in ("ttm", "annual"):
            income = stock.income_stmt
            balance = stock.balance_sheet
            cashflow = stock.cashflow
        else:
            income = stock.quarterly_income_stmt
            balance = stock.quarterly_balance_sheet
            cashflow = stock.quarterly_cashflow

        if income is None or income.empty:
            return []

        results = []
        # Each column is a report date
        num_periods = min(limit, len(income.columns))

        for i in range(num_periods):
            report_date = income.columns[i]
            report_period = report_date.strftime("%Y-%m-%d")

            if report_period > end_date:
                continue

            extras = {}
            for item_name in line_items:
                item_lower = item_name.lower()
                val = None

                # Search income statement
                if item_lower in _INCOME_STMT_MAP:
                    row_name = _INCOME_STMT_MAP[item_lower]
                    if row_name in income.index and i < len(income.columns):
                        val = income.iloc[income.index.get_loc(row_name), i]

                # Search balance sheet
                if val is None and item_lower in _BALANCE_SHEET_MAP:
                    row_name = _BALANCE_SHEET_MAP[item_lower]
                    if balance is not None and row_name in balance.index and i < len(balance.columns):
                        val = balance.iloc[balance.index.get_loc(row_name), i]

                # Search cashflow
                if val is None and item_lower in _CASHFLOW_MAP:
                    row_name = _CASHFLOW_MAP[item_lower]
                    if cashflow is not None and row_name in cashflow.index and i < len(cashflow.columns):
                        val = cashflow.iloc[cashflow.index.get_loc(row_name), i]

                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    extras[item_name] = float(val)
                else:
                    extras[item_name] = None

            results.append(LineItem(
                ticker=ticker,
                report_period=report_period,
                period=period,
                currency="USD",
                **extras,
            ))

        return results[:limit]
    except Exception as e:
        logger.warning("yfinance search_line_items failed for %s: %s", ticker, e)
        return []


# ── Insider Trades ───────────────────────────────────────────────────────────

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from yfinance."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**t) for t in cached]

    try:
        stock = yf.Ticker(ticker)
        transactions = stock.insider_transactions

        if transactions is None or transactions.empty:
            return []

        trades = []
        for _, row in transactions.iterrows():
            # Parse date
            tx_date = None
            filing_date = end_date
            if "Start Date" in row and pd.notna(row["Start Date"]):
                dt = pd.to_datetime(row["Start Date"])
                tx_date = dt.strftime("%Y-%m-%d")
                filing_date = tx_date

            # Filter by date range
            if tx_date:
                if tx_date > end_date:
                    continue
                if start_date and tx_date < start_date:
                    continue

            # Parse shares and value
            shares = row.get("Shares") or row.get("shares")
            if pd.notna(shares):
                shares = float(shares)
            else:
                shares = None

            value = row.get("Value") or row.get("value")
            if pd.notna(value):
                value = float(value)
            else:
                value = None

            price = value / abs(shares) if value and shares and shares != 0 else None

            # Determine transaction type
            text = str(row.get("Text", "") or row.get("Transaction", "")).lower()
            insider_name = row.get("Insider") or row.get("insider") or ""
            title = row.get("Position") or row.get("position") or ""

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=str(insider_name) if pd.notna(insider_name) else None,
                title=str(title) if pd.notna(title) else None,
                is_board_director="director" in str(title).lower() if title else None,
                transaction_date=tx_date,
                transaction_shares=shares,
                transaction_price_per_share=round(price, 2) if price else None,
                transaction_value=value,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=None,
                filing_date=filing_date,
            ))

        result = trades[:limit]
        if result:
            _cache.set_insider_trades(cache_key, [t.model_dump() for t in result])
        return result
    except Exception as e:
        logger.warning("yfinance get_insider_trades failed for %s: %s", ticker, e)
        return []


# ── Company News ─────────────────────────────────────────────────────────────

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from yfinance."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached := _cache.get_company_news(cache_key):
        return [CompanyNews(**n) for n in cached]

    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news

        if not news_items:
            return []

        articles = []
        for item in news_items:
            # yfinance v2 format: {id, content: {title, pubDate, ...}}
            content = item.get("content", item)  # fallback to item itself for old format

            # Parse publish time
            pub_ts = (content.get("pubDate") or content.get("providerPublishTime")
                      or content.get("publish_time"))
            if pub_ts:
                if isinstance(pub_ts, (int, float)):
                    dt = datetime.datetime.fromtimestamp(pub_ts)
                else:
                    dt = pd.to_datetime(pub_ts)
                date_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
                date_only = dt.strftime("%Y-%m-%d")
            else:
                date_str = end_date + "T00:00:00"
                date_only = end_date

            if date_only > end_date:
                continue
            if start_date and date_only < start_date:
                continue

            title = content.get("title", "")
            if not title:
                continue

            # Extract provider
            provider = content.get("provider", {})
            source = (provider.get("displayName") if isinstance(provider, dict)
                      else content.get("publisher", "Yahoo Finance"))

            articles.append(CompanyNews(
                ticker=ticker,
                title=title,
                author=content.get("author"),
                source=source or "Yahoo Finance",
                date=date_str,
                url=content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict)
                     else content.get("link", ""),
                sentiment=None,
            ))

        result = articles[:limit]
        if result:
            _cache.set_company_news(cache_key, [n.model_dump() for n in result])
        return result
    except Exception as e:
        logger.warning("yfinance get_company_news failed for %s: %s", ticker, e)
        return []


# ── Market Cap ───────────────────────────────────────────────────────────────

def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        return _safe_get(info, "marketCap")
    except Exception:
        return None
