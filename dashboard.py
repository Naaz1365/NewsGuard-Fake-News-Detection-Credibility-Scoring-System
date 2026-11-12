
import streamlit as st
import sys
import os

# Project root path
sys.path.insert(0, os.path.abspath("."))

from src.predict import predict_news

st.set_page_config(
    page_title="NewsGuard | Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

# Header
st.title("🛡️ NewsGuard")
st.subheader("AI-Powered Fake News Detection System")

st.caption(
    "Machine Learning • NLP • Credibility Scoring • Explainable AI"
)

st.divider()

# News input
st.markdown("### 📰 Analyze a News Statement")

news_text = st.text_area(
    "Enter your news statement:",
    height=180,
    placeholder="Paste your news statement here..."
)

if st.button(
    "🔍 Analyze News",
    type="primary",
    use_container_width=True
):

    if not news_text.strip():
        st.warning("Please enter a news statement first.")

    else:
        with st.spinner("Analyzing news..."):

            try:
                result = predict_news(news_text)

                st.divider()
                st.markdown("## Analysis Results")

                # Prediction and credibility
                col1, col2 = st.columns(2)

                with col1:
                    if result["prediction"] == "REAL":
                        st.success("Prediction: REAL")
                    else:
                        st.error("Prediction: FAKE")

                with col2:
                    st.metric(
                        "Credibility Score",
                        f'{result["credibility_score"]}/100'
                    )

                st.progress(
                    min(100, max(0, int(result["credibility_score"])))
                )

                # Probability analysis
                st.markdown("### 📊 Probability Analysis")

                p1, p2 = st.columns(2)

                p1.metric(
                    "REAL Probability",
                    f'{result["real_probability"]}%'
                )

                p2.metric(
                    "FAKE Probability",
                    f'{result["fake_probability"]}%'
                )

                st.bar_chart({
                    "Probability (%)": {
                        "REAL": result["real_probability"],
                        "FAKE": result["fake_probability"]
                    }
                })

                # Additional NLP features
                st.markdown("### 🧠 Text Analysis")

                f1, f2 = st.columns(2)

                f1.metric(
                    "Readability Score",
                    result["readability_score"]
                )

                f2.metric(
                    "Sentiment Score",
                    result["sentiment_score"]
                )

                st.caption(
                    "Readability and sentiment are supporting text features, "
                    "not direct measures of truth."
                )

                # Explanation
                st.markdown("### 💡 Model Explanation")

                st.info(result["explanation"])

                st.warning(result["disclaimer"])

            except Exception as e:
                st.error(f"Prediction error: {e}")

# Model performance
st.divider()

st.markdown("## 📊 Model Performance Evaluation")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "66.09%")
col2.metric("FAKE Precision", "60.15%")
col3.metric("FAKE Recall", "67.80%")
col4.metric("FAKE F1 Score", "63.75%")

st.caption(
    "Model: Extra Trees Classifier | "
    "Dataset: LIAR | "
    "Evaluation: Held-out test dataset"
)

# Classification report
st.markdown("### Classification Report")

st.table({
    "Class": ["FAKE", "REAL"],
    "Precision": [0.60, 0.72],
    "Recall": [0.68, 0.65],
    "F1 Score": [0.64, 0.68]
})

# Disclaimer
st.divider()

st.markdown("### ⚠️ Disclaimer")

st.write(
    "NewsGuard is an educational machine learning prototype. "
    "Predictions are based on patterns learned from the LIAR dataset. "
    "The credibility score is a model-estimated REAL probability, "
    "not proof of factual accuracy. Always verify news using "
    "reliable sources."
)

st.caption(
    "NewsGuard | Fake News Detection System | Internship Project"
)
