from scripts.make_table import get_metric


def test_get_metric_exact_and_suffixed():
    metrics = {
        "IC": 0.05,
        "1day.excess_return_with_cost.annualized_return": 0.10,
        "1day.excess_return_with_cost.information_ratio": 1.2,
        "1day.excess_return_with_cost.max_drawdown": -0.09,
    }
    assert get_metric(metrics, "IC") == 0.05
    assert get_metric(metrics, "excess_return_with_cost.annualized_return") == 0.10
    assert get_metric(metrics, "excess_return_with_cost.information_ratio") == 1.2
    assert get_metric(metrics, "excess_return_with_cost.max_drawdown") == -0.09
    assert get_metric(metrics, "nonexistent") is None
