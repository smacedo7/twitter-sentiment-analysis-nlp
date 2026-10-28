import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import src.config  # noqa: E402


@pytest.fixture
def app(model_path, monkeypatch):
    monkeypatch.setattr(src.config, "MODEL_PATH", model_path)
    return AppTest.from_file("streamlit_app.py", default_timeout=30)


def test_app_starts_without_errors(app):
    app.run()
    assert not app.exception


def test_single_tweet_analysis(app):
    app.run()
    app.text_area(key="tweet_text").set_value("I love this, it is great")
    app.button(key="analyze_single").click().run()
    assert not app.exception
    assert [s.value for s in app.success] == ["Positive"]


def test_tweet_without_words_shows_warning(app):
    app.run()
    app.text_area(key="tweet_text").set_value("@user http://t.co/abc")
    app.button(key="analyze_single").click().run()
    assert len(app.warning) == 1
