"""Web interface for the Twitter sentiment classifier.

Run it from the project root with:
    streamlit run streamlit_app.py
"""

import json

import pandas as pd
import streamlit as st

from src.batch import CONFIDENCE_COLUMN, LABEL_COLUMN, predict_dataframe
from src.config import METRICS_PATH, MODEL_PATH
from src.predict import load_model, predict_detailed

MAX_BATCH_ROWS = 20_000

EXAMPLES = {
    "Happy tweet": "I love how easy this application is to use!",
    "Angry tweet": "Worst customer service ever, I am never buying from them again.",
    "Negation": "I am not happy with this at all.",
}


def load_metrics() -> dict | None:
    """Read the evaluation summary saved during training, if it exists."""
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def render_sidebar() -> None:
    st.sidebar.header("About")
    st.sidebar.write(
        "Classifies tweets as **Positive**, **Negative** or **Neutral** (when the model is not sure) with a model trained on the "
        "[Sentiment140](http://help.sentiment140.com/for-students) dataset."
    )

    metrics = load_metrics()
    if metrics:
        best = metrics["models"][metrics["best_model"]]
        st.sidebar.subheader("Model")
        st.sidebar.write(metrics["best_model"])
        st.sidebar.metric("Accuracy", f"{best['accuracy']:.1%}")
        st.sidebar.metric("F1-score", f"{best['f1']:.1%}")


def show_result(result: dict) -> None:
    if result["label"] == "Positive":
        st.success("Positive")
    elif result["label"] == "Negative":
        st.error("Negative")
    else:
        st.info("Neutral: the model is not sure whether this tweet is positive or negative.")
    st.progress(result["positive_probability"], text=f"Probability of being positive: {result['positive_probability']:.1%}")


def set_example(text: str) -> None:
    # Used as a button callback: it runs before the text area is drawn, so it can change its value.
    st.session_state["tweet_text"] = text


def render_single_tweet_tab() -> None:
    st.text_area("Tweet", key="tweet_text", height=120, max_chars=1000, placeholder="Type or paste a tweet...")

    st.caption("Or try an example:")
    columns = st.columns(len(EXAMPLES))
    for column, (name, text) in zip(columns, EXAMPLES.items()):
        column.button(name, key=f"example_{name}", on_click=set_example, args=(text,))

    if st.button("Analyze", type="primary", key="analyze_single"):
        try:
            result = predict_detailed(st.session_state.get("tweet_text", ""), MODEL_PATH)
        except ValueError as error:
            st.warning(str(error))
        else:
            show_result(result)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read a CSV, falling back to latin-1 (the encoding of the original Sentiment140 file)."""
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def render_batch_tab() -> None:
    uploaded_file = st.file_uploader("Upload a CSV file with a column of tweets", type=["csv"])
    if uploaded_file is None:
        return

    try:
        df = read_uploaded_csv(uploaded_file)
    except Exception:
        st.error("Could not read this file as a CSV.")
        return

    text_columns = [name for name in df.columns if not pd.api.types.is_numeric_dtype(df[name])]
    if not text_columns:
        st.error("The file has no text column.")
        return

    default_index = text_columns.index("text") if "text" in text_columns else 0
    text_column = st.selectbox("Column with the tweets", text_columns, index=default_index)
    st.caption(f"{len(df):,} rows found.")

    if st.button("Analyze file", type="primary", key="analyze_batch"):
        if len(df) > MAX_BATCH_ROWS:
            st.warning(f"Only the first {MAX_BATCH_ROWS:,} rows will be analyzed.")
        # Results are kept in session_state because Streamlit reruns the script on every
        # interaction (for example the download button), which would otherwise erase them.
        st.session_state["batch_results"] = predict_dataframe(df.head(MAX_BATCH_ROWS), text_column, MODEL_PATH)

    results = st.session_state.get("batch_results")
    if results is None:
        return

    counts = results[LABEL_COLUMN].value_counts()
    positive, neutral, negative, skipped = st.columns(4)
    positive.metric("Positive", int(counts.get("Positive", 0)))
    neutral.metric("Neutral", int(counts.get("Neutral", 0)))
    negative.metric("Negative", int(counts.get("Negative", 0)))
    skipped.metric("Not analyzable", int(results[LABEL_COLUMN].isna().sum()))

    st.bar_chart(counts)
    st.dataframe(results)
    st.download_button(
        "Download results (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_results.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="💬")
    st.title("Twitter Sentiment Analysis")
    st.write("Find out whether a tweet sounds positive or negative.")

    render_sidebar()

    try:
        load_model(MODEL_PATH)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    single_tab, batch_tab = st.tabs(["Single tweet", "CSV file"])
    with single_tab:
        render_single_tweet_tab()
    with batch_tab:
        render_batch_tab()


main()
