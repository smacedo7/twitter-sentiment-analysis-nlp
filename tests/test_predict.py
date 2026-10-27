import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.predict import load_model, predict_batch, predict_detailed, predict_sentiment


def test_predict_sentiment_returns_label(model_path):
    assert predict_sentiment("I love this song, what a wonderful day!", model_path) == "Positive"
    assert predict_sentiment("I hate this, the worst service ever", model_path) == "Negative"


def test_predict_detailed_returns_label_and_confidence(model_path):
    result = predict_detailed("so happy with my new phone", model_path)

    assert set(result) == {"label", "confidence"}
    assert result["label"] == "Positive"
    assert 0.5 <= result["confidence"] <= 1.0


def test_predict_detailed_rejects_text_without_words(model_path):
    with pytest.raises(ValueError):
        predict_detailed("@user http://t.co/abc !!!", model_path)


def test_predict_batch_keeps_order_and_marks_empty_texts(model_path):
    texts = ["I love this", "@user http://t.co/abc", "I hate this"]

    results = predict_batch(texts, model_path)

    assert [r["label"] for r in results] == ["Positive", None, "Negative"]
    assert results[1]["confidence"] is None


def test_predict_batch_with_no_texts(model_path):
    assert predict_batch([], model_path) == []


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        load_model(tmp_path / "does_not_exist.pkl")


def test_models_without_predict_proba_are_supported(tmp_path):
    # Linear SVM has no predict_proba: confidence must come from the decision function.
    texts = ["love great happy", "hate awful sad", "wonderful amazing", "terrible worst bad"]
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())]).fit(texts, [1, 0, 1, 0])
    path = tmp_path / "svm.pkl"
    joblib.dump(pipeline, path)

    result = predict_detailed("love amazing", path)

    assert result["label"] == "Positive"
    assert 0.5 <= result["confidence"] <= 1.0
