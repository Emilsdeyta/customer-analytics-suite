"""Association-rule mining (Apriori / FP-Growth) for basket-level NBO."""
from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def mine_rules(transactions: list[list[str]], min_support: float = 0.02, min_confidence: float = 0.3) -> pd.DataFrame:
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values("lift", ascending=False)
