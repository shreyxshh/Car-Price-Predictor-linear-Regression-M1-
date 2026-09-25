
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AutoValue AI | Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM UI
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #172554 100%);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: rgba(8,15,31,.97);
    border-right: 1px solid rgba(148,163,184,.15);
}
.hero {
    padding: 28px 32px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(30,41,59,.96), rgba(30,64,175,.55));
    border: 1px solid rgba(147,197,253,.22);
    box-shadow: 0 18px 50px rgba(0,0,0,.25);
    margin-bottom: 22px;
}
.hero h1 { margin:0; font-size:42px; letter-spacing:-1.5px; }
.hero p { margin:8px 0 0; color:#cbd5e1; font-size:16px; }
.prediction {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(14,116,144,.32), rgba(30,64,175,.35));
    border: 1px solid rgba(125,211,252,.25);
    text-align:center;
    margin-top:15px;
}
.prediction .label { color:#bae6fd; font-size:15px; }
.prediction .price { font-size:48px; font-weight:800; margin:4px 0; }
.prediction .sub { color:#cbd5e1; font-size:13px; }
.stButton > button { border-radius:12px; font-weight:700; min-height:44px; }
div[data-testid="stMetric"] {
    background:rgba(15,23,42,.75);
    border:1px solid rgba(148,163,184,.16);
    padding:15px;
    border-radius:16px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
DEFAULT_DATA_FILE = "1.04. Real-life example.csv"
SAVED_MODEL_FILE = "linear_regression_model.pkl"

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    local_file = Path(DEFAULT_DATA_FILE)
    if local_file.exists():
        return pd.read_csv(local_file)

    # Optional fallback for the exact dataset used in the notebook.
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


# =========================================================
# PREPROCESSING
# IMPORTANT:
# This follows the notebook:
# 1. Drop missing Price/EngineV
# 2. EngineV <= 10
# 3. Log-transform Price
# 4. Drop Price and Model
# 5. Label-encode categorical columns
# 6. X = features, y = Log_Price
#
# Extra safety:
# - Remove non-positive prices before np.log()
# - Clean inf values
# - Fill remaining missing feature values so sklearn never
#   receives NaN values.
# =========================================================
def prepare_data(raw_df):
    df = raw_df.copy()

    required = {"Price", "EngineV"}
    missing_required = required - set(df.columns)

    if missing_required:
        raise ValueError(
            f"Required column(s) missing: {', '.join(sorted(missing_required))}"
        )

    # Make the two critical columns numeric.
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["EngineV"] = pd.to_numeric(df["EngineV"], errors="coerce")

    # Same core filtering as notebook.
    df = df.dropna(subset=["Price", "EngineV"]).copy()
    df = df[df["EngineV"] <= 10].copy()

    # log(0) and log(negative) are invalid.
    df = df[df["Price"] > 0].copy()

    if df.empty:
        raise ValueError(
            "No valid rows remain after cleaning Price and EngineV."
        )

    # Target transformation.
    df["Log_Price"] = np.log(df["Price"])

    # EXACT IMPORTANT FIX:
    # Price must NOT be an input feature because it is the target.
    # The original notebook drops both Price and Model.
    drop_columns = [c for c in ["Price", "Model"] if c in df.columns]
    df = df.drop(columns=drop_columns)

    # Separate target/features.
    y = df["Log_Price"].copy()
    X = df.drop(columns=["Log_Price"]).copy()

    # Convert inf to NaN.
    X = X.replace([np.inf, -np.inf], np.nan)

    encoders = {}

    # Encode categorical columns.
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in categorical_columns:
        X[col] = X[col].fillna("Unknown").astype(str)

        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder

    # Convert boolean columns to integers.
    bool_columns = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_columns:
        X[col] = X[col].astype(int)

    # Final numeric conversion.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # =====================================================
    # SECOND IMPORTANT FIX:
    # Linear/Ridge/Lasso cannot train with NaN values.
    # Fill missing numeric values with the median.
    # =====================================================
    imputer = SimpleImputer(strategy="median")
    X_values = imputer.fit_transform(X)

    X = pd.DataFrame(
        X_values,
        columns=X.columns,
        index=X.index,
    )

    # Safety check.
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix still contains invalid values.")

    return df, X, y, encoders, imputer


# =========================================================
# TRAIN MODELS
# =========================================================
@st.cache_data(show_spinner=False)
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predicted_log = model.predict(X_test)

        # Convert log-price back to normal price.
        predicted_price = np.exp(predicted_log)
        actual_price = np.exp(y_test)

        results[name] = {
            "model": model,
            "r2": r2_score(y_test, predicted_log),
            "mae": mean_absolute_error(actual_price, predicted_price),
            "rmse": np.sqrt(
                mean_squared_error(actual_price, predicted_price)
            ),
            "pred_log": predicted_log,
            "pred_price": predicted_price,
            "actual_log": y_test,
            "actual_price": actual_price,
        }

    return results, X_train, X_test, y_train, y_test


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🚗 AutoValue AI</h1>
    <p>Interactive Car Price Prediction using Linear, Ridge & Lasso Regression.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Project Controls")

    uploaded_file = st.file_uploader(
        "Upload car-sales CSV",
        type=["csv"],
        help="Upload your car-sales dataset. The app expects Price and EngineV columns.",
    )

    st.divider()

    model_choice = st.selectbox(
        "Prediction model",
        [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
        ],
    )

    st.divider()

    st.markdown("### 🛡️ Built-in fixes")
    st.caption("✓ Target leakage removed")
    st.caption("✓ Missing values handled")
    st.caption("✓ Invalid numeric values handled")
    st.caption("✓ Categorical columns encoded")
    st.caption("✓ log(Price) safety check")

# =========================================================
# LOAD DATA
# =========================================================
raw_df = load_csv(uploaded_file)

if raw_df is None:
    st.error(
        "Dataset not found.\n\n"
        "Put `1.04. Real-life example.csv` in the same folder as `app.py`, "
        "or upload the CSV from the sidebar."
    )
    st.stop()

# =========================================================
# PREPARE DATA
# =========================================================
try:
    processed_df, X, y, encoders, imputer = prepare_data(raw_df)

except Exception as error:
    st.error("❌ Dataset preprocessing failed.")
    st.exception(error)
    st.stop()

# =========================================================
# TRAIN
# =========================================================
try:
    results, X_train, X_test, y_train, y_test = train_models(X, y)

except Exception as error:
    st.error("❌ Model training failed.")
    st.exception(error)

    st.markdown("### Diagnostic information")
    st.write("Feature data types:")
    st.dataframe(X.dtypes.rename("dtype").to_frame())

    st.write("Missing values:")
    st.dataframe(X.isna().sum().rename("missing").to_frame())

    st.write("Feature preview:")
    st.dataframe(X.head())

    st.stop()

selected = results[model_choice]

# =========================================================
# METRICS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("🚘 Cars", f"{len(processed_df):,}")
c2.metric("🧩 Features", f"{X.shape[1]}")
c3.metric("📈 R² Score", f"{selected['r2']:.3f}")
c4.metric("💰 Test MAE", f"${selected['mae']:,.0f}")

# =========================================================
# TABS
# =========================================================
tab_predict, tab_compare, tab_data, tab_about = st.tabs(
    [
        "🚀 Predict Price",
        "📊 Model Comparison",
        "🔎 Explore Data",
        "🧠 How It Works",
    ]
)

# =========================================================
# PREDICTION TAB
# =========================================================
with tab_predict:
    st.markdown("### 🚗 Enter vehicle details")
    st.caption(
        "The form is generated from your dataset's feature columns. "
        "Price is intentionally excluded because it is the prediction target."
    )

    values = {}

    feature_columns = list(X.columns)

    input_columns = st.columns(2)

    for index, col in enumerate(feature_columns):
        with input_columns[index % 2]:

            # Categorical feature
            if col in encoders:
                options = sorted(
                    raw_df[col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if options:
                    values[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options,
                    )

            # Numeric feature
            else:
                original_series = pd.to_numeric(
                    raw_df[col],
                    errors="coerce"
                ).dropna()

                if original_series.empty:
                    values[col] = float(X[col].median())
                    st.number_input(
                        col.replace("_", " ").title(),
                        value=float(X[col].median()),
                        disabled=True,
                    )
                    continue

                minimum = float(original_series.min())
                maximum = float(original_series.max())
                median = float(original_series.median())

                if minimum == maximum:
                    values[col] = minimum
                    st.number_input(
                        col.replace("_", " ").title(),
                        value=minimum,
                        disabled=True,
                    )
                else:
                    values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=median,
                    )

    st.divider()

    if st.button(
        "✨ Predict Car Price",
        type="primary",
        use_container_width=True,
    ):
        input_row = {}

        for col in X.columns:

            # Categorical feature
            if col in encoders:
                encoder = encoders[col]
                selected_value = str(values.get(col, "Unknown"))

                try:
                    encoded_value = encoder.transform(
                        [selected_value]
                    )[0]

                except ValueError:
                    # Unknown category fallback.
                    encoded_value = 0

                input_row[col] = encoded_value

            # Numeric feature
            else:
                value = values.get(col, float(X[col].median()))

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = float(X[col].median())

                input_row[col] = value

        input_df = pd.DataFrame(
            [input_row],
            columns=X.columns,
        )

        # Apply exactly the same imputer used during training.
        input_array = imputer.transform(input_df)

        input_df = pd.DataFrame(
            input_array,
            columns=X.columns,
        )

        predicted_log_price = float(
            selected["model"].predict(input_df)[0]
        )

        predicted_price = float(
            np.exp(predicted_log_price)
        )

        st.markdown(
            f"""
            <div class="prediction">
                <div class="label">Estimated Car Price</div>
                <div class="price">${predicted_price:,.0f}</div>
                <div class="sub">
                    {model_choice} • log-price prediction converted back to price
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success("✅ Prediction generated successfully.")

        with st.expander("🔍 View processed input"):
            st.dataframe(
                input_df,
                use_container_width=True,
            )

# =========================================================
# MODEL COMPARISON
# =========================================================
with tab_compare:
    st.markdown("### 📊 Model Comparison")

    comparison = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "R²": [results[m]["r2"] for m in results],
            "MAE ($)": [results[m]["mae"] for m in results],
            "RMSE ($)": [results[m]["rmse"] for m in results],
        }
    )

    st.dataframe(
        comparison.style.format(
            {
                "R²": "{:.4f}",
                "MAE ($)": "${:,.0f}",
                "RMSE ($)": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Actual vs Predicted Price")

    chart_df = pd.DataFrame(
        {
            "Actual Price": selected["actual_price"],
            "Predicted Price": selected["pred_price"],
        }
    )

    st.scatter_chart(
        chart_df,
        x="Actual Price",
        y="Predicted Price",
        use_container_width=True,
    )

# =========================================================
# DATA EXPLORER
# =========================================================
with tab_data:
    st.markdown("### 🔎 Dataset Explorer")

    a, b, c = st.columns(3)

    a.metric("Rows", f"{len(raw_df):,}")
    b.metric("Columns", f"{len(raw_df.columns)}")
    c.metric(
        "Missing values",
        f"{int(raw_df.isna().sum().sum()):,}",
    )

    st.markdown("#### Raw Dataset")
    st.dataframe(
        raw_df.head(100),
        use_container_width=True,
        height=420,
    )

    st.markdown("#### Processed Features")
    st.dataframe(
        X.head(100),
        use_container_width=True,
        height=350,
    )

# =========================================================
# ABOUT
# =========================================================
with tab_about:
    st.markdown("### 🧠 ML Pipeline")

    st.markdown("""
    **1. Load dataset**  
    The car-sales CSV is loaded.

    **2. Clean critical columns**  
    Rows missing `Price` or `EngineV` are removed.

    **3. Engine filtering**  
    Rows with `EngineV > 10` are removed.

    **4. Target transformation**  
    `Price` is transformed into `Log_Price = log(Price)`.

    **5. Remove target leakage**  
    `Price` and `Model` are removed from the feature matrix.

    **6. Encode categorical variables**  
    Object/category columns are label-encoded.

    **7. Handle missing feature values**  
    Remaining numeric missing values are filled using median imputation.

    **8. Train/test split**  
    80% training and 20% testing with `random_state=42`.

    **9. Regression models**  
    Linear Regression, Ridge Regression and Lasso Regression are trained.

    **10. Prediction**  
    The model predicts log-price and `exp()` converts it back to the original price scale.
    """)

st.markdown("---")
st.caption("AutoValue AI • Car Price Prediction • Streamlit")
