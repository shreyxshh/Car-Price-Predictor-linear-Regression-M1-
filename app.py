
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AutoValue AI | Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM UI
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #172554 100%);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: rgba(8, 15, 31, 0.96);
        border-right: 1px solid rgba(148,163,184,.15);
    }
    .hero {
        padding: 28px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(30,41,59,.95), rgba(30,64,175,.55));
        border: 1px solid rgba(147,197,253,.22);
        box-shadow: 0 18px 50px rgba(0,0,0,.25);
        margin-bottom: 22px;
    }
    .hero h1 {
        margin: 0;
        font-size: 42px;
        letter-spacing: -1.5px;
    }
    .hero p {
        margin: 8px 0 0;
        color: #cbd5e1;
        font-size: 16px;
    }
    .card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(15,23,42,.76);
        border: 1px solid rgba(148,163,184,.16);
        box-shadow: 0 10px 30px rgba(0,0,0,.15);
    }
    .metric-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(15,23,42,.82);
        border: 1px solid rgba(96,165,250,.18);
    }
    .metric-label { color: #94a3b8; font-size: 13px; }
    .metric-value { color: #f8fafc; font-size: 28px; font-weight: 750; }
    .small-muted { color: #94a3b8; font-size: 13px; }
    div[data-testid="stMetric"] {
        background: rgba(15,23,42,.75);
        border: 1px solid rgba(148,163,184,.16);
        padding: 15px;
        border-radius: 16px;
    }
    .prediction {
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(14,116,144,.32), rgba(30,64,175,.35));
        border: 1px solid rgba(125,211,252,.25);
        text-align: center;
        margin-top: 15px;
    }
    .prediction .label { color: #bae6fd; font-size: 15px; }
    .prediction .price { font-size: 48px; font-weight: 800; margin: 4px 0; }
    .prediction .sub { color: #cbd5e1; font-size: 13px; }
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 44px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA / MODEL HELPERS
# ---------------------------------------------------------
DEFAULT_DATA_FILE = "1.04. Real-life example.csv"
MODEL_FILE = "linear_regression_model.pkl"

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    local = Path(DEFAULT_DATA_FILE)
    if local.exists():
        return pd.read_csv(local)

    # Optional fallback: download the exact dataset used in the notebook.
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "smritisingh1997/car-salescsv",
            DEFAULT_DATA_FILE,
        )
    except Exception:
        return None

def prepare_data(raw):
    df = raw.copy()

    # Match the preprocessing in the supplied notebook.
    required_target = "Price"
    required_engine = "EngineV"
    if required_target not in df.columns or required_engine not in df.columns:
        raise ValueError("The CSV must contain at least 'Price' and 'EngineV' columns.")

    df = df.dropna(subset=["Price", "EngineV"])
    df = df[df["EngineV"] <= 10].copy()
    df["Log_Price"] = np.log(df["Price"])

    if "Model" in df.columns:
        df = df.drop(columns=["Model"])

    feature_columns = [c for c in df.columns if c != "Log_Price"]
    encoders = {}

    for col in feature_columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    X = df.drop(columns=["Log_Price"])
    y = df["Log_Price"]

    return df, X, y, encoders

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)
        pred_price = np.exp(pred_log)
        actual_price = np.exp(y_test)

        results[name] = {
            "model": model,
            "r2": r2_score(y_test, pred_log),
            "mae": mean_absolute_error(actual_price, pred_price),
            "rmse": np.sqrt(mean_squared_error(actual_price, pred_price)),
            "pred_log": pred_log,
            "pred_price": pred_price,
            "actual_log": y_test,
            "actual_price": actual_price,
        }

    return results, X_train, X_test, y_train, y_test

@st.cache_resource(show_spinner=False)
def load_saved_model():
    p = Path(MODEL_FILE)
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🚗 AutoValue AI</h1>
    <p>Interactive car-price prediction powered by Linear, Ridge & Lasso Regression.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Project Controls")
    uploaded = st.file_uploader(
        "Upload your car-sales CSV",
        type=["csv"],
        help="Upload the same dataset used for training, or another dataset with compatible columns."
    )

    st.divider()

    st.markdown("### Prediction model")
    model_choice = st.selectbox(
        "Choose regression model",
        ["Linear Regression", "Ridge Regression", "Lasso Regression"],
        index=0,
    )

    st.divider()
    st.markdown("### About")
    st.caption(
        "The preprocessing follows your notebook: missing Price/EngineV rows are removed, "
        "EngineV > 10 is filtered, Price is log-transformed, Model is dropped, and "
        "categorical features are label-encoded."
    )

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
raw_df = load_csv(uploaded)

if raw_df is None:
    st.warning(
        "No dataset was found. Put `1.04. Real-life example.csv` beside this app "
        "or upload the CSV from the sidebar."
    )
    st.stop()

try:
    processed_df, X, y, encoders = prepare_data(raw_df)
except Exception as e:
    st.error(f"Dataset preprocessing failed: {e}")
    st.stop()

results, X_train, X_test, y_train, y_test = train_models(X, y)

# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------
selected = results[model_choice]

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚘 Cars in dataset", f"{len(processed_df):,}")
c2.metric("🧩 Features", f"{X.shape[1]}")
c3.metric("📈 R² Score", f"{selected['r2']:.3f}")
c4.metric("💰 Test MAE", f"${selected['mae']:,.0f}")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_predict, tab_compare, tab_data, tab_about = st.tabs(
    ["🚀 Predict Price", "📊 Model Comparison", "🔎 Explore Data", "🧠 How It Works"]
)

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
with tab_predict:
    st.markdown("### Enter vehicle details")
    st.caption("Use the controls below to create a prediction. The interface automatically adapts to the columns in your dataset.")

    values = {}
    input_cols = st.columns(2)

    # Use original raw columns excluding target/model.
    prediction_features = [c for c in X.columns if c in raw_df.columns]

    for idx, col in enumerate(prediction_features):
        with input_cols[idx % 2]:
            series = raw_df[col]

            if col in encoders:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if options:
                    values[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options,
                        help=f"Select a {col.replace('_', ' ')} from the training data."
                    )
            elif pd.api.types.is_numeric_dtype(series):
                min_v = float(series.min())
                max_v = float(series.max())
                median_v = float(series.median())

                if col == "Year":
                    values[col] = st.number_input(
                        "Year", min_value=int(min_v), max_value=int(max_v),
                        value=int(median_v), step=1
                    )
                elif col.lower() == "mileage":
                    values[col] = st.number_input(
                        "Mileage", min_value=0.0, max_value=max_v,
                        value=float(median_v), step=1000.0
                    )
                elif col == "EngineV":
                    values[col] = st.number_input(
                        "Engine Volume (L)", min_value=0.1, max_value=min(max_v, 10.0),
                        value=float(np.clip(median_v, 0.1, 10.0)), step=0.1
                    )
                else:
                    values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=min_v, max_value=max_v,
                        value=median_v
                    )

    st.markdown("---")

    if st.button("✨ Predict Car Price", type="primary", use_container_width=True):
        row = {}
        for col in X.columns:
            if col not in values:
                row[col] = X[col].median()
            elif col in encoders:
                encoder = encoders[col]
                value = str(values[col])
                try:
                    row[col] = int(encoder.transform([value])[0])
                except ValueError:
                    row[col] = int(X[col].mode()[0])
            else:
                row[col] = values[col]

        input_df = pd.DataFrame([row], columns=X.columns)
        predicted_log = float(selected["model"].predict(input_df)[0])
        predicted_price = float(np.exp(predicted_log))

        st.markdown(f"""
        <div class="prediction">
            <div class="label">Estimated Market Price</div>
            <div class="price">${predicted_price:,.0f}</div>
            <div class="sub">{model_choice} • prediction returned after inverse log transformation</div>
        </div>
        """, unsafe_allow_html=True)

        st.success("Prediction generated successfully.")

        with st.expander("🔍 See encoded model input"):
            st.dataframe(input_df, use_container_width=True)

# ---------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------
with tab_compare:
    st.markdown("### 📊 Regression model comparison")

    comparison = pd.DataFrame({
        "Model": list(results.keys()),
        "R²": [results[m]["r2"] for m in results],
        "MAE ($)": [results[m]["mae"] for m in results],
        "RMSE ($)": [results[m]["rmse"] for m in results],
    }).set_index("Model")

    st.dataframe(
        comparison.style.format({
            "R²": "{:.4f}",
            "MAE ($)": "${:,.0f}",
            "RMSE ($)": "${:,.0f}",
        }),
        use_container_width=True
    )

    st.markdown("### Actual vs Predicted")
    chart_df = pd.DataFrame({
        "Actual Price": selected["actual_price"].values,
        "Predicted Price": selected["pred_price"],
    })
    st.scatter_chart(chart_df, x="Actual Price", y="Predicted Price", use_container_width=True)

    st.caption(
        "R² is calculated on log-price, while MAE and RMSE are shown after converting "
        "predictions back to the original price scale."
    )

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
with tab_data:
    st.markdown("### 🔎 Dataset explorer")

    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(raw_df):,}")
    b.metric("Columns", f"{len(raw_df.columns)}")
    c.metric("Missing values", f"{int(raw_df.isna().sum().sum()):,}")

    st.dataframe(raw_df.head(100), use_container_width=True, height=420)

    st.markdown("### Feature summary")
    st.dataframe(raw_df.describe(include="all").T, use_container_width=True)

# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------
with tab_about:
    st.markdown("### 🧠 Project pipeline")
    st.markdown("""
    **1. Data loading** → Car-sales dataset is loaded from CSV or Kaggle.

    **2. Cleaning** → Rows missing `Price` or `EngineV` are removed and vehicles with
    `EngineV > 10` are filtered.

    **3. Target transformation** → `Price` is transformed to `Log_Price = log(Price)`.

    **4. Feature preparation** → `Model` is removed and categorical columns are
    label-encoded.

    **5. Train/test split** → 80% training and 20% testing with `random_state=42`.

    **6. Regression** → Linear, Ridge and Lasso models are trained.

    **7. Evaluation** → R², MAE and RMSE are displayed.

    **8. Prediction** → The predicted log-price is converted back using `exp()`.
    """)

    st.info(
        "This Streamlit interface is designed around the preprocessing and models "
        "present in the supplied notebook rather than introducing a different ML pipeline."
    )

st.markdown("---")
st.caption("AutoValue AI • Machine Learning Car Price Prediction • Streamlit")
