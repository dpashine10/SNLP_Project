from src.week4_evaluation import metrics_for_ranked_labels, ndcg_at_k


def test_metrics_for_perfect_ranking():
    labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    metrics = metrics_for_ranked_labels(labels)
    assert metrics["Precision@5"] == 0.4
    assert metrics["Precision@10"] == 0.2
    assert metrics["Recall@10"] == 1.0
    assert metrics["MRR"] == 1.0


def test_ndcg_is_bounded():
    assert 0.0 <= ndcg_at_k([1, 0, 1, 0], 4) <= 1.0
