import pandas as pd

from analysis import analyse_dataset, build_context_narrative, infer_task_type


def test_infer_task_type_classification_on_categorical_target():
    target = pd.Series(["cat", "dog", "cat", "dog", "cat"])
    task, reason = infer_task_type(target)
    assert task == "classification"
    assert isinstance(reason, str)


def test_infer_task_type_regression_on_continuous_target():
    target = pd.Series(range(100)) + 0.5
    task, reason = infer_task_type(target)
    assert task == "regression"


def test_analyse_dataset_runs_on_simple_dataframe():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": ["x", "y", "x", "y", "x"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    result = analyse_dataset(df, target="target", task="classification")

    assert result["shape"]["rows"] == 5
    assert result["target"] == "target"
    assert result["task"] == "classification"
    assert "feature_a" in result["columns"]["numeric"]
    assert "feature_b" in result["columns"]["categorical"]


def test_analyse_dataset_raises_on_missing_target():
    df = pd.DataFrame({"a": [1, 2, 3]})
    try:
        analyse_dataset(df, target="not_a_column", task="classification")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_context_narrative_produces_summary():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )
    analysis = analyse_dataset(df, target="target", task="classification")
    narrative = build_context_narrative(analysis)

    assert "summary" in narrative
    assert "observations" in narrative
    assert "evaluation_guidance" in narrative
    assert len(narrative["observations"]) > 0
