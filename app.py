from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation | K-Means",
    page_icon="👥",
    layout="wide",
)


# ---------------------------------------------------------
# Constants
# IMPORTANT: This order must match the training notebook.
# ---------------------------------------------------------
FEATURES = [
    "Age",
    "Income",
    "Total_spend",
    "NumWebPurchases",
    "NumStorePurchases",
    "Recency",
    "NumWebVisitsMonth",
]

DISPLAY_NAMES = {
    "Age": "Age",
    "Income": "Income",
    "Total_spend": "Total Spend",
    "NumWebPurchases": "Web Purchases",
    "NumStorePurchases": "Store Purchases",
    "Recency": "Recency",
    "NumWebVisitsMonth": "Web Visits / Month",
}

BASE_DIR = Path(__file__).resolve().parent
SCALER_PATH = BASE_DIR / "scaler.pkl"

# The notebook saves "k_means_model.pkl".
# The second filename is supported in case the file was renamed.
MODEL_CANDIDATES = [
    BASE_DIR / "k_means_model.pkl",
    BASE_DIR / "k_mean_model.pkl",
]


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            opacity: 0.78;
            margin-bottom: 1.2rem;
        }
        .result-box {
            padding: 1.15rem 1.25rem;
            border-radius: 0.8rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin: 0.5rem 0 1rem 0;
        }
        .small-note {
            font-size: 0.9rem;
            opacity: 0.72;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "scaler.pkl was not found. Add scaler.pkl to the same GitHub "
            "folder as app.py."
        )

    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)

    if model_path is None:
        expected = ", ".join(path.name for path in MODEL_CANDIDATES)
        raise FileNotFoundError(
            f"K-Means model was not found. Add one of these files to the "
            f"same folder as app.py: {expected}"
        )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(model_path)

    return scaler, model, model_path.name


def validate_artifacts(scaler, model):
    """Check that the saved artifacts are compatible with the 7 app features."""
    expected_count = len(FEATURES)

    scaler_n = getattr(scaler, "n_features_in_", None)
    model_n = getattr(model, "n_features_in_", None)

    if scaler_n is not None and scaler_n != expected_count:
        raise ValueError(
            f"The scaler expects {scaler_n} features, but this app is configured "
            f"for {expected_count} features."
        )

    if model_n is not None and model_n != expected_count:
        raise ValueError(
            f"The K-Means model expects {model_n} features, but this app is "
            f"configured for {expected_count} features."
        )


def build_input_dataframe(
    age,
    income,
    total_spend,
    web_purchases,
    store_purchases,
    recency,
    web_visits,
):
    """Create model input in exactly the same column order used in training."""
    return pd.DataFrame(
        [
            {
                "Age": age,
                "Income": income,
                "Total_spend": total_spend,
                "NumWebPurchases": web_purchases,
                "NumStorePurchases": store_purchases,
                "Recency": recency,
                "NumWebVisitsMonth": web_visits,
            }
        ],
        columns=FEATURES,
    )


def describe_cluster(model, cluster_id):
    """
    Describe the selected cluster relative to the StandardScaler training mean.

    K-Means was fitted on standardized features, so a positive centroid
    coordinate means above the training-set average for that feature and a
    negative value means below average.
    """
    centers = getattr(model, "cluster_centers_", None)
    if centers is None or cluster_id >= len(centers):
        return []

    center = np.asarray(centers[cluster_id], dtype=float)
    statements = []

    # Use a moderate threshold so only meaningful tendencies are described.
    for feature, z_value in zip(FEATURES, center):
        label = DISPLAY_NAMES[feature]

        if z_value >= 0.60:
            statements.append((abs(z_value), f"Higher-than-average **{label}**"))
        elif z_value <= -0.60:
            statements.append((abs(z_value), f"Lower-than-average **{label}**"))

    statements.sort(reverse=True, key=lambda item: item[0])
    return [text for _, text in statements[:4]]


def cluster_centroid_table(scaler, model, cluster_id):
    """Convert the cluster centroid back to the original feature scale."""
    centers = getattr(model, "cluster_centers_", None)
    if centers is None:
        return None

    center_scaled = np.asarray(centers[cluster_id]).reshape(1, -1)

    try:
        center_original = scaler.inverse_transform(center_scaled)[0]
    except Exception:
        return None

    table = pd.DataFrame(
        {
            "Feature": [DISPLAY_NAMES[f] for f in FEATURES],
            "Cluster centroid": center_original,
        }
    )

    # Make the values easier to read.
    table["Cluster centroid"] = table["Cluster centroid"].round(2)
    return table


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">👥 Customer Segmentation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'K-Means clustering app built from the trained customer segmentation model.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load model + scaler
# ---------------------------------------------------------
try:
    scaler, model, loaded_model_name = load_artifacts()
    validate_artifacts(scaler, model)
except Exception as exc:
    st.error("The model files could not be loaded.")
    st.code(str(exc))
    st.info(
        "Place `app.py`, `scaler.pkl`, and your K-Means `.pkl` file in the "
        "same GitHub repository folder."
    )
    st.stop()


# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("Enter customer information")

    with st.form("customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                value=40,
                step=1,
                help="Customer age. Your notebook trained the model using Age directly.",
            )

            income = st.number_input(
                "Income",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                format="%.2f",
                help="Customer income using the same unit/currency as the training dataset.",
            )

            total_spend = st.number_input(
                "Total Spend",
                min_value=0.0,
                value=500.0,
                step=50.0,
                format="%.2f",
                help=(
                    "In the notebook, Total_spend = Wines + Fruits + Meat + Fish "
                    "+ Sweets + Gold product spending."
                ),
            )

            web_purchases = st.number_input(
                "Number of Web Purchases",
                min_value=0,
                value=5,
                step=1,
            )

        with col2:
            store_purchases = st.number_input(
                "Number of Store Purchases",
                min_value=0,
                value=5,
                step=1,
            )

            recency = st.number_input(
                "Recency",
                min_value=0,
                value=30,
                step=1,
                help="Number of days since the customer's last purchase.",
            )

            web_visits = st.number_input(
                "Web Visits per Month",
                min_value=0,
                value=5,
                step=1,
            )

        submitted = st.form_submit_button(
            "Predict Customer Segment",
            type="primary",
            use_container_width=True,
        )

with right:
    st.subheader("Model information")
    st.write(
        "This application uses the same **7 features** that were selected in "
        "your notebook and applies the saved `StandardScaler` before sending "
        "the values to the saved K-Means model."
    )

    info_df = pd.DataFrame(
        {
            "Feature used by model": [DISPLAY_NAMES[f] for f in FEATURES],
            "Training column": FEATURES,
        }
    )
    st.dataframe(info_df, hide_index=True, use_container_width=True)

    n_clusters = getattr(model, "n_clusters", "Unknown")
    st.caption(
        f"Loaded model: `{loaded_model_name}`  •  Number of clusters: `{n_clusters}`"
    )


# ---------------------------------------------------------
# Prediction result
# ---------------------------------------------------------
if submitted:
    input_df = build_input_dataframe(
        age=age,
        income=income,
        total_spend=total_spend,
        web_purchases=web_purchases,
        store_purchases=store_purchases,
        recency=recency,
        web_visits=web_visits,
    )

    try:
        scaled_input = scaler.transform(input_df)
        cluster_id = int(model.predict(scaled_input)[0])
    except Exception as exc:
        st.error("Prediction failed.")
        st.code(str(exc))
        st.stop()

    st.divider()
    st.subheader("Prediction result")

    st.markdown(
        f"""
        <div class="result-box">
            <div style="font-size:0.95rem; opacity:0.72;">Predicted customer cluster</div>
            <div style="font-size:2rem; font-weight:750;">Segment {cluster_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_col1, result_col2 = st.columns([1, 1], gap="large")

    with result_col1:
        st.markdown("#### Customer values")
        display_input = input_df.rename(columns=DISPLAY_NAMES).T.reset_index()
        display_input.columns = ["Feature", "Value"]
        st.dataframe(display_input, hide_index=True, use_container_width=True)

    with result_col2:
        st.markdown("#### Segment tendencies")
        tendencies = describe_cluster(model, cluster_id)

        if tendencies:
            for statement in tendencies:
                st.markdown(f"- {statement}")
        else:
            st.write(
                "This segment is relatively close to the training-set average "
                "across the selected features."
            )

        st.caption(
            "These tendencies compare the cluster centroid with the standardized "
            "training-set average. They are descriptive, not quality scores."
        )

    centroid_df = cluster_centroid_table(scaler, model, cluster_id)
    if centroid_df is not None:
        with st.expander("View this segment's approximate centroid"):
            st.dataframe(
                centroid_df,
                hide_index=True,
                use_container_width=True,
            )

    st.info(
        "K-Means is an unsupervised clustering algorithm, so the segment number "
        "is a cluster ID rather than a probability or confidence score."
    )


# ---------------------------------------------------------
# Footer / methodology
# ---------------------------------------------------------
with st.expander("How this prediction works"):
    st.markdown(
        """
        1. The seven values entered above are arranged in the exact feature order
           used during model training.
        2. The saved `scaler.pkl` standardizes those values.
        3. The scaled values are passed to the saved K-Means model.
        4. The model assigns the customer to the nearest learned cluster centroid.

        **Important:** K-Means cluster numbers such as `0`, `1`, or `5` are
        identifiers only. Their business meaning should be determined by examining
        the feature profile of each cluster.
        """
    )
