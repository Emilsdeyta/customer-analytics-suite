"""Load Online Retail II transaction data for CLV modeling.

Reuses the same loading/cleaning logic as the NBO module — both consume the same
transaction history, just extracting different features from it (see project plan
section 2: "same dataset, different feature sets, different questions").
"""
from __future__ import annotations

import pandas as pd

from cas.nbo.data import clean, load_raw  # noqa: F401  (re-exported for cas.clv.* callers)