"""Load and clean the Online Retail II dataset (UCI) for NBO."""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID"]


def load_raw(path: str) -> pd.DataFrame:
    """Load both sheets of the Online Retail II workbook and concatenate them."""
    sheets = pd.read_excel(path, sheet_name=["Year 2009-2010", "Year 2010-2011"])
    df = pd.concat(sheets.values(), ignore_index=True)
    return df[REQUIRED_COLUMNS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancellations (Invoice starting with 'C'), returns/negative rows, and rows with no customer id."""
    df = df.copy()
    df = df.dropna(subset=["Customer ID"])
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df