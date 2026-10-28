import json

from src.tune import run_search


def test_run_search_writes_best_params(tmp_path):
    summary = run_search(
        "tests/fixtures/sample_tweets.csv",
        sample_size=0,
        n_iter=2,
        cv=2,
        n_jobs=1,
        best_params_path=tmp_path / "best.json",
        reports_dir=tmp_path,
    )

    saved = json.loads((tmp_path / "best.json").read_text())
    assert saved["best_params"] == summary["best_params"]
    assert "clf__C" in saved["best_params"]
    assert (tmp_path / "tuning_results.csv").exists()


def test_choose_final_params_keeps_baseline_when_search_does_not_improve():
    from src.tune import BASELINE_PARAMS, choose_final_params

    searched = {"clf__C": 5.0}
    assert choose_final_params(searched, best_score=0.80, baseline_score=0.81) == (BASELINE_PARAMS, False)
    assert choose_final_params(searched, best_score=0.82, baseline_score=0.81) == (searched, True)
