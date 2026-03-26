"""API layer — routes to yfinance adapter.

Original Financial Datasets API preserved in api_original.py.
To switch back: copy api_original.py over this file.

Set DATA_SOURCE=financialdatasets in .env to use original API.
"""

import os

if os.environ.get("DATA_SOURCE", "yfinance").lower() == "financialdatasets":
    from src.tools.api_original import *  # noqa: F401,F403
else:
    from src.tools.yfinance_api import *  # noqa: F401,F403
