import pytest

from cas.scoring.priority import compute_priority_score


def test_basic_score():
    score = compute_priority_score(churn_probability=0.8, clv=500, best_offer_propensity=0.5)
    assert score == pytest.approx(0.8 * 500 * 1.5)


def test_no_offer_defaults_to_multiplier_one():
    score = compute_priority_score(churn_probability=0.5, clv=200)
    assert score == pytest.approx(0.5 * 200 * 1.0)


def test_zero_churn_gives_zero_score():
    assert compute_priority_score(churn_probability=0.0, clv=1000) == 0.0


@pytest.mark.parametrize("bad_prob", [-0.1, 1.1])
def test_invalid_churn_probability_raises(bad_prob):
    with pytest.raises(ValueError):
        compute_priority_score(churn_probability=bad_prob, clv=100)


def test_negative_clv_raises():
    with pytest.raises(ValueError):
        compute_priority_score(churn_probability=0.5, clv=-10)


@pytest.mark.parametrize("bad_prop", [-0.01, 1.5])
def test_invalid_offer_propensity_raises(bad_prop):
    with pytest.raises(ValueError):
        compute_priority_score(churn_probability=0.5, clv=100, best_offer_propensity=bad_prop)