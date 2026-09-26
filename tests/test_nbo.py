from cas.nbo.rank import hybrid_rank


def test_hybrid_rank_orders_by_combined_score():
    scores = [
        {"product": "A", "propensity": 0.5},
        {"product": "B", "propensity": 0.4},
    ]
    lifts = {"A": 1.0, "B": 2.0}
    ranked = hybrid_rank(scores, lifts, top_n=2)
    assert ranked[0]["product"] == "B"  # 0.4*2.0 = 0.8 > 0.5*1.0 = 0.5
