
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(120deg, #172554, #2563eb);
    color: white;
    text-align: center;
}

.hero h1 {
    color: white;
    font-size: 36px;
}

.hero p {
    color: #dbeafe;
    font-size: 16px;
}

.result {
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.spam {
    background-color: #fee2e2;
    color: #991b1b;
}

.ham {
    background-color: #dcfce7;
    color: #166534;
}

.small-text {
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("model/spam_model.pkl")

@st.cache_data
def load_metrics():
    with open("model/metrics.json", "r") as f:
        return json.load(f)

try:
    model = load_model()
    metrics = load_metrics()
except Exception:
    st.error(
        "Model not found! Please run train_model.py first."
    )
    st.stop()

# ---------------- SESSION HISTORY ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🛡️ SMS Spam Detection System</h1>
    <p>Intelligent SMS classification using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- NAVIGATION ----------------

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Model Performance", "📜 History", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.success("Machine Learning Model: Active")
st.sidebar.caption("Algorithm: Multinomial Naive Bayes")

# ---------------- HOME PAGE ----------------

if page == "🏠 Home":

    st.subheader("Check Your SMS")

    st.write(
        "Enter any SMS message below to check "
        "whether it is spam or a legitimate message."
    )

    message = st.text_area(
        "Enter SMS message:",
        placeholder="Example: Congratulations! You won a free prize...",
        height=150
    )

    if st.button("🔍 Detect Spam", use_container_width=True):

        if not message.strip():
            st.warning("Please enter a message first.")

        else:
            prediction = int(model.predict([message])[0])

            probabilities = model.predict_proba([message])[0]
            confidence = float(max(probabilities)) * 100

            st.markdown("---")
            st.subheader("Detection Result")

            if prediction == 1:
                st.markdown(
                    '<div class="result spam">'
                    '🚨 SPAM MESSAGE DETECTED'
                    '</div>',
                    unsafe_allow_html=True
                )
                label = "SPAM"

            else:
                st.markdown(
                    '<div class="result ham">'
                    '✅ LEGITIMATE MESSAGE (HAM)'
                    '</div>',
                    unsafe_allow_html=True
                )
                label = "HAM"

            st.write("")
            st.metric("Model Confidence", f"{confidence:.2f}%")

            st.progress(confidence / 100)

            st.session_state.history.insert(
                0,
                {
                    "Message": message,
                    "Prediction": label,
                    "Confidence": f"{confidence:.2f}%"
                }
            )

            st.caption(
                "Note: Confidence is the model's estimated probability, "
                "not a guarantee of correctness."
            )

# ---------------- PERFORMANCE PAGE ----------------

elif page == "📊 Model Performance":

    st.subheader("Model Evaluation Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    st.markdown("---")

    st.write("### Dataset Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total SMS", metrics["total_messages"])
    c2.metric("Training SMS", metrics["training_messages"])
    c3.metric("Testing SMS", metrics["testing_messages"])

    st.markdown("---")

    st.subheader("Confusion Matrix")

    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["HAM", "SPAM"],
        yticklabels=["HAM", "SPAM"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The confusion matrix shows correct and incorrect "
        "classifications for both categories."
    )

# ---------------- HISTORY PAGE ----------------

elif page == "📜 History":

    st.subheader("Message Detection History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No messages have been checked yet.")

# ---------------- ABOUT PAGE ----------------

elif page == "ℹ️ About":

    st.subheader("About This Project")

    st.write("""
    The SMS Spam Detection System is a machine learning
    application that identifies unwanted and suspicious
    text messages.

    **Technologies Used:**

    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Multinomial Naive Bayes
    - Streamlit
    - Matplotlib and Seaborn

    **Objective:**

    To classify SMS messages into SPAM and HAM
    categories using machine learning.

    **Algorithm:**

    Multinomial Naive Bayes.

    **Developer Note:**

    This is an educational project. Predictions may
    occasionally be incorrect.
    """)

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    "<center class='small-text'>"
    "SMS Spam Detection System | College Project | "
    "Powered by Python and Machine Learning"
    "</center>",
    unsafe_allow_html=True
)
