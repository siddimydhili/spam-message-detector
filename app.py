"""
app.py
------
Streamlit web interface for the Spam Message Detector.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import streamlit as st

from preprocessing import clean_text

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Page settings must be the first Streamlit command.
st.set_page_config(page_title="Spam Message Detector", page_icon="📩", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    """Load the trained model and vectorizer ONCE and keep them in memory.

    st.cache_resource means this function body only runs the first time;
    later runs of the app reuse the cached objects (no retraining, no reloading).
    """
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_message(message: str, model, vectorizer):
    """Return (is_spam: bool, spam_probability: float) for one message."""
    cleaned = clean_text(message)                 # 1. preprocess
    if not cleaned:
        raise ValueError("The message has no meaningful words after cleaning.")
    features = vectorizer.transform([cleaned])    # 2. TF-IDF
    spam_probability = float(model.predict_proba(features)[0][1])  # 3. model
    return spam_probability >= 0.5, spam_probability


def show_result(is_spam: bool, spam_probability: float) -> None:
    """Display the prediction with a different look for SPAM / NOT SPAM."""
    confidence = spam_probability if is_spam else 1 - spam_probability

    if is_spam:
        st.markdown(
            """
            <div style="background-color:#ffe5e5;border-left:8px solid #e53935;
                        padding:20px;border-radius:8px;margin-top:10px;">
                <h2 style="color:#b71c1c;margin:0;">🚨 SPAM MESSAGE</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color:#e6f7ea;border-left:8px solid #2e7d32;
                        padding:20px;border-radius:8px;margin-top:10px;">
                <h2 style="color:#1b5e20;margin:0;">✅ NOT SPAM</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.metric("Confidence", f"{confidence * 100:.1f}%")
    st.progress(min(max(spam_probability, 0.0), 1.0), text=f"Spam probability: {spam_probability * 100:.1f}%")


def main() -> None:
    st.title("📩 Spam Message Detector")
    st.write(
        "Enter a message below and our machine-learning model will determine "
        "whether it is spam or not spam."
    )
    st.write("")

    # Make sure the model files exist before doing anything else.
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
        st.error("Trained model not found. Please run `python train_model.py` first.")
        st.stop()

    try:
        model, vectorizer = load_artifacts()
    except Exception as error:
        st.error(f"Could not load the model: {error}")
        st.info("Try re-training with `python train_model.py`.")
        st.stop()

    message = st.text_area(
        "Your message",
        height=180,
        placeholder="Type or paste a message here...",
    )

    if st.button("Check Message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            try:
                is_spam, spam_probability = predict_message(message, model, vectorizer)
                show_result(is_spam, spam_probability)
            except ValueError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"Something went wrong while checking the message: {error}")

    st.write("")
    st.divider()
    st.caption("🤖 A machine-learning project built with Python, scikit-learn and Streamlit.")


if __name__ == "__main__":
    main()
