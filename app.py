import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")


# PAGE CONFIG
st.set_page_config(
    page_title="AI Lead Quality Prediction",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# CUSTOM CSS
st.markdown(
    """
    <style>
    :root {
        --ink: #f8fafc;
        --muted: #a9b3c7;
        --panel: rgba(22, 31, 58, 0.82);
        --line: rgba(148, 163, 184, 0.18);
        --primary: #7c5cfc;
        --cyan: #22d3ee;
        --pink: #f472b6;
        --green: #34d399;
        --yellow: #fbbf24;
    }
    .stApp {
        background: radial-gradient(circle at 8% 4%, rgba(124,92,252,.18), transparent 28%),
                    radial-gradient(circle at 92% 8%, rgba(34,211,238,.12), transparent 24%),
                    #0b1020;
        color: var(--ink);
    }
    .block-container { max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    .hero { padding: 1.3rem 0 1.8rem; }
    .eyebrow { color: var(--cyan); font-size: .75rem; font-weight: 800; letter-spacing: .18rem; }
    .hero h1 { color: var(--ink); font-size: 3rem; line-height: 1.05; margin: .45rem 0 .6rem; }
    .hero p { color: var(--muted); font-size: 1.05rem; margin: 0; }
    .metric-card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.2rem; min-height: 7rem; }
    .metric-label { color: var(--muted); font-size: .75rem; letter-spacing: .08rem; text-transform: uppercase; }
    .metric-value { color: var(--ink); font-size: 2rem; font-weight: 800; margin-top: .4rem; }
    .section-kicker { color: var(--pink); font-size: .75rem; font-weight: 800; letter-spacing: .16rem; margin-top: 1rem; }
    .section-title { color: var(--ink); font-size: 1.8rem; font-weight: 800; margin: .25rem 0 1rem; }
    .result-card { background: rgba(52,211,153,.12); border: 1px solid rgba(52,211,153,.4); border-radius: 12px; padding: 1.4rem; text-align: center; }
    .result-card.medium { background: rgba(251,191,36,.12); border-color: rgba(251,191,36,.4); }
    .result-card.low { background: rgba(244,114,182,.12); border-color: rgba(244,114,182,.4); }
    .result-label { color: var(--muted); font-size: .75rem; letter-spacing: .14rem; }
    .result-value { color: var(--ink); font-size: 2rem; font-weight: 900; margin-top: .35rem; }
    .stButton > button { background: var(--primary); border: 0; border-radius: 8px; color: white; font-weight: 800; min-height: 3rem; }
    [data-baseweb="tab-list"] { gap: .4rem; }
    [data-baseweb="tab"] { color: var(--muted); font-weight: 700; }
    [aria-selected="true"] { color: var(--cyan) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# DATA LOADING
DATA_PATH = "lead_quality_dataset.csv"
FEATURE_COLUMNS = [
    "Industry", "Company_Size", "Website_Quality", "LinkedIn_Activity",
    "Email_Available", "Location", "Estimated_Revenue", "Number_of_Employees",
    "Previous_Response", "Meeting_Status",
]
TARGET_COLUMN = "Lead_Quality"
CATEGORICAL_COLUMNS = [
    "Industry", "Company_Size", "Website_Quality", "LinkedIn_Activity",
    "Email_Available", "Location", "Previous_Response", "Meeting_Status",
]
NUMERICAL_COLUMNS = ["Estimated_Revenue", "Number_of_Employees"]
ENGINEERED_NUMERICAL_COLUMNS = [
    "Website_Quality_Score",
    "LinkedIn_Activity_Score",
    "Email_Available_Score",
    "Company_Size_Score",
    "Previous_Response_Score",
    "Meeting_Status_Score",
    "Revenue_Log",
    "Employees_Log",
    "Lead_Signal_Total",
]
MODEL_NUMERICAL_COLUMNS = NUMERICAL_COLUMNS + ENGINEERED_NUMERICAL_COLUMNS
CLASS_LABELS = ["Low", "Medium", "High"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]


@st.cache_data(show_spinner=False)
def load_and_prepare_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    raw_data = pd.read_csv(path)
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in raw_data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    missing_before = raw_data[expected_columns].isna().sum()
    duplicates_before = int(raw_data[expected_columns].duplicated().sum())
    cleaned_data = raw_data[expected_columns].drop_duplicates().reset_index(drop=True)
    return cleaned_data, missing_before, duplicates_before


# DATA CLEANING
try:
    df, missing_before, duplicates_before = load_and_prepare_dataset(DATA_PATH)
except FileNotFoundError:
    st.error("lead_quality_dataset.csv not found. Please place the dataset in the same folder as app.py.")
    st.stop()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# PREPROCESSING
class LeadSignalFeatures(BaseEstimator, TransformerMixin):
    """Add ordered business signals without using the target column."""

    def fit(self, features, target=None):
        return self

    def transform(self, features):
        transformed = features.copy()
        signal_maps = {
            "Website_Quality_Score": ("Website_Quality", {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}),
            "LinkedIn_Activity_Score": ("LinkedIn_Activity", {"Low": 0, "Medium": 1, "High": 2}),
            "Email_Available_Score": ("Email_Available", {"No": 0, "Yes": 2}),
            "Company_Size_Score": ("Company_Size", {"Small": 0, "Medium": 1, "Large": 2}),
            "Previous_Response_Score": ("Previous_Response", {"No Response": 0, "Negative": 0, "Neutral": 1, "Positive": 3}),
            "Meeting_Status_Score": ("Meeting_Status", {"Not Contacted": 0, "Not Scheduled": 0.5, "Scheduled": 2, "Completed": 3}),
        }
        for output_column, (source_column, mapping) in signal_maps.items():
            transformed[output_column] = transformed[source_column].map(mapping)
        transformed["Revenue_Log"] = np.log1p(transformed["Estimated_Revenue"])
        transformed["Employees_Log"] = np.log1p(transformed["Number_of_Employees"])
        transformed["Lead_Signal_Total"] = (
            transformed["Website_Quality_Score"]
            + transformed["LinkedIn_Activity_Score"]
            + transformed["Email_Available_Score"]
            + transformed["Company_Size_Score"]
            + transformed["Previous_Response_Score"]
            + transformed["Meeting_Status_Score"]
            + transformed["Revenue_Log"] / 5
            + transformed["Employees_Log"] / 3
        )
        return transformed


def create_preprocessor():
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numerical", numerical_pipeline, MODEL_NUMERICAL_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


# MODEL TRAINING
@st.cache_resource(show_spinner="Training Logistic Regression, Random Forest, and XGBoost...")
def train_models(data):
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("XGBoost is not installed. Run: pip install xgboost") from error

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )
    pipelines = {
        "Logistic Regression": Pipeline([
            ("signal_features", LeadSignalFeatures()),
            ("preprocessor", create_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, C=2.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("signal_features", LeadSignalFeatures()),
            ("preprocessor", create_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    xgb_encoder = LabelEncoder()
    y_train_encoded = xgb_encoder.fit_transform(y_train)
    pipelines["XGBoost"] = Pipeline([
        ("signal_features", LeadSignalFeatures()),
        ("preprocessor", create_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.04,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            n_jobs=-1, tree_method="hist",
        )),
    ])

    pipelines["Logistic Regression"].fit(x_train, y_train)
    pipelines["Random Forest"].fit(x_train, y_train)
    pipelines["XGBoost"].fit(x_train, y_train_encoded)

    predictions = {
        "Logistic Regression": pipelines["Logistic Regression"].predict(x_test),
        "Random Forest": pipelines["Random Forest"].predict(x_test),
    }
    predictions["XGBoost"] = xgb_encoder.inverse_transform(
        pipelines["XGBoost"].predict(x_test).astype(int)
    )

    metric_rows = []
    for model_name in MODEL_NAMES:
        prediction = predictions[model_name]
        metric_rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, prediction, average="weighted", zero_division=0),
        })

    return {
        "pipelines": pipelines,
        "predictions": predictions,
        "metrics": pd.DataFrame(metric_rows),
        "x_test": x_test,
        "y_test": y_test,
        "xgb_classes": list(xgb_encoder.classes_),
        "train_size": len(x_train),
        "test_size": len(x_test),
    }


try:
    training = train_models(df)
except RuntimeError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()

models = training["pipelines"]
metrics_df = training["metrics"]


# HEADER
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">INTELLIGENT SCORING / MULTI-CLASS CLASSIFICATION</div>
        <h1>AI Lead Quality Prediction</h1>
        <p>Machine Learning Dashboard for Business Lead Classification</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# SUMMARY CARDS
summary_columns = st.columns(4)
with summary_columns[0]:
    metric_card("Total Leads", f"{len(df):,}")
with summary_columns[1]:
    metric_card("Features", len(FEATURE_COLUMNS))
with summary_columns[2]:
    metric_card("Classes", df[TARGET_COLUMN].nunique())
with summary_columns[3]:
    metric_card("Models", len(MODEL_NAMES))


def style_axes(axis):
    axis.set_facecolor("#111a31")
    axis.tick_params(colors="#a9b3c7", labelsize=8)
    axis.xaxis.label.set_color("#a9b3c7")
    axis.xaxis.label.set_fontsize(9)
    axis.yaxis.label.set_color("#a9b3c7")
    axis.yaxis.label.set_fontsize(9)
    axis.title.set_color("#f8fafc")
    axis.title.set_fontsize(11)
    legend = axis.get_legend()
    if legend is not None:
        legend.get_title().set_fontsize(9)
        for text in legend.get_texts():
            text.set_fontsize(8)
    for spine in axis.spines.values():
        spine.set_color("#263452")


def show_countplot(data, x_column, title, order=None, container=None):
    figure, axis = plt.subplots(figsize=(5.5, 3.0))
    sns.countplot(data=data, x=x_column, hue=TARGET_COLUMN, order=order, hue_order=CLASS_LABELS, ax=axis)
    axis.set_title(title)
    axis.set_xlabel(x_column.replace("_", " "))
    axis.set_ylabel("Leads")
    axis.tick_params(axis="x", rotation=25)
    style_axes(axis)
    figure.tight_layout()
    (container or st).pyplot(figure, width="content")
    plt.close(figure)


# DASHBOARD TABS
overview_tab, performance_tab, prediction_tab = st.tabs(
    ["Overview & EDA", "Model Performance", "Predict Lead"]
)


# EDA
with overview_tab:
    st.markdown('<div class="section-kicker">01 / DATA INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    info_left, info_right = st.columns(2)
    with info_left:
        st.info(f"Cleaning removed {duplicates_before:,} duplicate rows. Missing feature values remain for pipeline imputation.")
        st.write("Missing values before cleaning")
        st.dataframe(missing_before.rename("Missing Values").to_frame(), use_container_width=True)
    with info_right:
        st.write("Dataset summary")
        st.dataframe(pd.DataFrame({
            "Measure": ["Rows after cleaning", "Training samples", "Testing samples", "Classes"],
            "Value": [len(df), training["train_size"], training["test_size"], df[TARGET_COLUMN].nunique()],
        }), use_container_width=True, hide_index=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        figure, axis = plt.subplots(figsize=(5.5, 3.0))
        sns.countplot(data=df, x=TARGET_COLUMN, order=CLASS_LABELS, ax=axis)
        axis.set_title("Lead Quality Distribution")
        axis.set_xlabel("Lead Quality")
        axis.set_ylabel("Leads")
        style_axes(axis)
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)
    with chart_right:
        figure, axis = plt.subplots(figsize=(5.5, 3.0))
        df["Industry"].value_counts().sort_values().plot(kind="barh", ax=axis, color="#22d3ee")
        axis.set_title("Industry Distribution")
        axis.set_xlabel("Leads")
        axis.set_ylabel("Industry")
        style_axes(axis)
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)

    categorical_row_one = st.columns(2)
    show_countplot(df, "Company_Size", "Company Size vs Lead Quality", ["Small", "Medium", "Large"], categorical_row_one[0])
    show_countplot(df, "Website_Quality", "Website Quality vs Lead Quality", ["Poor", "Average", "Good", "Excellent"], categorical_row_one[1])

    categorical_row_two = st.columns(2)
    show_countplot(df, "LinkedIn_Activity", "LinkedIn Activity vs Lead Quality", ["Low", "Medium", "High"], categorical_row_two[0])
    show_countplot(df, "Previous_Response", "Previous Response vs Lead Quality", container=categorical_row_two[1])

    categorical_row_three = st.columns(2)
    show_countplot(df, "Meeting_Status", "Meeting Status vs Lead Quality", container=categorical_row_three[0])

    numeric_left, numeric_right = st.columns(2)
    with numeric_left:
        figure, axis = plt.subplots(figsize=(5.5, 3.0))
        sns.histplot(data=df, x="Estimated_Revenue", bins=40, kde=True, ax=axis, color="#7c5cfc")
        axis.set_title("Estimated Revenue Distribution")
        style_axes(axis)
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)
    with numeric_right:
        figure, axis = plt.subplots(figsize=(5.5, 3.0))
        sns.histplot(data=df, x="Number_of_Employees", bins=40, kde=True, ax=axis, color="#f472b6")
        axis.set_title("Number of Employees Distribution")
        style_axes(axis)
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)


# MODEL EVALUATION
with performance_tab:
    st.markdown('<div class="section-kicker">02 / MODEL ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Measured Model Performance</div>', unsafe_allow_html=True)
    display_metrics = metrics_df.copy()
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_metrics[metric] = (display_metrics[metric] * 100).map(lambda value: f"{value:.2f}%")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    selected_analysis_model = st.selectbox("Select model for confusion matrix", MODEL_NAMES, key="analysis_model")
    selected_prediction = training["predictions"][selected_analysis_model]
    matrix = confusion_matrix(training["y_test"], selected_prediction, labels=CLASS_LABELS)
    performance_left, performance_right = st.columns(2)
    with performance_left:
        figure, axis = plt.subplots(figsize=(5.5, 3.0))
        metrics_df.set_index("Model").plot(kind="bar", ax=axis, color=["#7c5cfc", "#22d3ee", "#f472b6", "#34d399"])
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score")
        axis.set_title("Model Metrics")
        axis.tick_params(axis="x", rotation=0)
        style_axes(axis)
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)
    with performance_right:
        figure, axis = plt.subplots(figsize=(5.5, 3.0))
        sns.heatmap(
            matrix,
            annot=True,
            annot_kws={"fontsize": 8},
            fmt="d",
            cmap="mako",
            cbar=False,
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            ax=axis,
        )
        axis.set_title(f"Confusion Matrix: {selected_analysis_model}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        style_axes(axis)
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)


# PREDICTION INTERFACE
with prediction_tab:
    st.markdown('<div class="section-kicker">03 / AI PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predict Lead Quality</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Select model", MODEL_NAMES, key="prediction_model")
    selected_model = models[selected_model_name]

    input_left, input_right = st.columns(2)
    with input_left:
        industry = st.selectbox("Industry", sorted(df["Industry"].dropna().unique()))
        company_size = st.selectbox("Company Size", sorted(df["Company_Size"].dropna().unique()))
        website_quality = st.selectbox("Website Quality", sorted(df["Website_Quality"].dropna().unique()))
        linkedin_activity = st.selectbox("LinkedIn Activity", sorted(df["LinkedIn_Activity"].dropna().unique()))
        email_available = st.selectbox("Email Availability", sorted(df["Email_Available"].dropna().unique()))
    with input_right:
        location = st.selectbox("Location", sorted(df["Location"].dropna().unique()))
        estimated_revenue = st.number_input("Estimated Revenue", min_value=0.0, value=1_000_000.0, step=10_000.0)
        number_of_employees = st.number_input("Number of Employees", min_value=1, value=100, step=1)
        previous_response = st.selectbox("Previous Response", sorted(df["Previous_Response"].dropna().unique()))
        meeting_status = st.selectbox("Meeting Status", sorted(df["Meeting_Status"].dropna().unique()))

    predict_clicked = st.button("Predict Lead Quality", type="primary", use_container_width=True)
    if predict_clicked:
        try:
            input_data = pd.DataFrame([{
                "Industry": industry,
                "Company_Size": company_size,
                "Website_Quality": website_quality,
                "LinkedIn_Activity": linkedin_activity,
                "Email_Available": email_available,
                "Location": location,
                "Estimated_Revenue": estimated_revenue,
                "Number_of_Employees": number_of_employees,
                "Previous_Response": previous_response,
                "Meeting_Status": meeting_status,
            }], columns=FEATURE_COLUMNS)
            prediction = selected_model.predict(input_data)[0]
            probabilities = selected_model.predict_proba(input_data)[0]
            probability_classes = training["xgb_classes"] if selected_model_name == "XGBoost" else list(selected_model.classes_)
            probability_map = dict(zip(probability_classes, probabilities))
            result_class = str(prediction).lower()
            st.markdown(
                f'<div class="result-card {result_class}"><div class="result-label">PREDICTED LEAD QUALITY</div><div class="result-value">{str(prediction).upper()}</div></div>',
                unsafe_allow_html=True,
            )
            st.subheader("Prediction Probabilities")
            probability_columns = st.columns(3)
            for column, label in zip(probability_columns, CLASS_LABELS):
                score = float(probability_map.get(label, 0.0))
                with column:
                    st.metric(label, f"{score * 100:.2f}%")
                    st.progress(score)
            with st.expander("View entered lead information"):
                st.dataframe(input_data, use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"Prediction failed: {error}")


# FOOTER
st.divider()
st.caption("AI Lead Quality Prediction • Models are trained from lead_quality_dataset.csv at application startup.")
