
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

# Page setup
st.set_page_config(
    page_title="Diabetes Health Analytics",
    page_icon="🩺",
    layout="wide"
)

# Sidebar
st.sidebar.title("🩺 Diabetes Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "EDA Explorer",
        "Model Comparison",
        "Feature Insights",
        "Evaluation",
        "Prediction Demo"
    ]
)

# Header
st.title("Diabetes Health Analytics")
st.caption("CDC Diabetes Health Indicators | Machine Learning Dashboard")

# Dataset location
DATA_PATH = Path("data/cdc_diabetes_health_indicators.csv")

# Load the dataset
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if DATA_PATH.exists():
    df = load_data(DATA_PATH)
else:
    df = None
    st.error(f"Dataset not found at: {DATA_PATH}")


if page == "Overview":
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "253,680")
    col2.metric("Input Features", "21")
    col3.metric("Leading Model", "Random Forest")

    st.markdown("""
    ### About this project

    This project explores health indicators associated with
    diabetes and compares multiple machine learning models.

    **Planned features**
    - Interactive data exploration
    - Model performance comparisons
    - Feature importance visualisation
    - Threshold and classification evaluation
    """)

    st.info(
        "This is an educational analytics project, "
        "not a medical diagnostic tool."
    )


elif page == "EDA Explorer":
    st.header("Exploratory Data Analysis")

    if df is None:
        st.error(f"Dataset not found: {DATA_PATH}")
    else:
        st.write(
            f"Dataset shape: **{df.shape[0]:,} rows × "
            f"{df.shape[1]} columns**"
        )

        with st.expander("View dataset preview"):
            st.dataframe(df.head(10), use_container_width=True)

        target_candidates = [
            c for c in df.columns
            if c.strip().lower() == "diabetes_binary"
        ]

        st.subheader("Diabetes Distribution")

        if target_candidates:
            target = target_candidates[0]

            counts = (
                df[target]
                .value_counts()
                .reindex([0, 1], fill_value=0)
                .rename(index={
                    0: "No diabetes",
                    1: "Diabetes"
                })
                .rename_axis("Group")
                .reset_index(name="Count")
            )

            fig = px.bar(
                counts,
                x="Group",
                y="Count",
                color="Group",
                title="Diabetes vs Non-Diabetes"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Target column Diabetes_binary not found.")

        st.subheader("Explore Health Indicators")

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        features = [
            c for c in numeric_columns
            if c.strip().lower() != "diabetes_binary"
        ]

        if features:
            feature = st.selectbox(
                "Select a health indicator",
                features
            )

            fig = px.histogram(
                df,
                x=feature,
                marginal="box",
                title=f"Distribution of {feature}"
            )
            st.plotly_chart(fig, use_container_width=True)

            if target_candidates:
                fig2 = px.histogram(
                    df,
                    x=feature,
                    color=target_candidates[0],
                    barmode="overlay",
                    opacity=0.7,
                    title=f"{feature} by Diabetes Status"
                )
                st.plotly_chart(fig2, use_container_width=True)



elif page == "Model Comparison":
    st.header("Model Comparison")
    st.caption("Performance comparison of five trained classifiers")

    comparison_path = Path("data/model_comparison.csv")

    if comparison_path.exists():
        results = pd.read_csv(comparison_path)

        # Summary metrics
        best_f1 = results.loc[results["F1-score"].idxmax()]
        best_auc = results.loc[results["ROC-AUC"].idxmax()]

        col1, col2 = st.columns(2)
        col1.metric(
            "Best F1-score",
            f"{best_f1['F1-score']:.3f}",
            help=str(best_f1["Model"])
        )
        col2.metric(
            "Best ROC-AUC",
            f"{best_auc['ROC-AUC']:.3f}",
            help=str(best_auc["Model"])
        )

        st.subheader("Model Performance Table")
        st.dataframe(
            results.style.format({
                "Threshold": "{:.2f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "Average Precision": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Compare Evaluation Metrics")

        metric = st.selectbox(
            "Choose a metric",
            [
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC",
                "Average Precision"
            ]
        )

        fig = px.bar(
            results,
            x="Model",
            y=metric,
            color="Model",
            title=f"{metric} Across the Five Models",
            hover_data=["Threshold"],
            text=results[metric].round(3)
        )
        fig.update_layout(
            xaxis_title="Machine Learning Model",
            yaxis_title=metric,
            showlegend=False
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Decision Thresholds")
        fig_threshold = px.bar(
            results,
            x="Model",
            y="Threshold",
            color="Model",
            title="Selected Threshold for Each Model"
        )
        fig_threshold.update_layout(showlegend=False)
        st.plotly_chart(fig_threshold, use_container_width=True)

    else:
        st.error(
            "Comparison file not found. Please add "
            "model_comparison.csv to the data folder."
        )



elif page == "Feature Insights":
    st.header("Feature Insights")
    st.caption(
        "Explore which health indicators contribute most "
        "to the trained Random Forest model."
    )

    MODEL_PATH = Path("diabetes_random_forest_compressed.joblib")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
    else:
        try:
            model_bundle = joblib.load(MODEL_PATH)
            model = model_bundle["model"]
            feature_names = model_bundle["feature_names"]

            if not hasattr(model, "feature_importances_"):
                st.error(
                    "The loaded model does not expose "
                    "feature_importances_."
                )
            else:
                importances = model.feature_importances_

                if len(feature_names) != len(importances):
                    st.error(
                        "The number of feature names does not "
                        "match the model's importance values."
                    )
                else:
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values(
                        "Importance",
                        ascending=False
                    )

                    st.success(
                        f"Loaded {model_bundle['model_name']} "
                        f"with {len(feature_names)} features."
                    )

                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Total Features",
                        len(feature_names)
                    )
                    col2.metric(
                        "Decision Threshold",
                        f"{model_bundle['threshold']:.2f}"
                    )

                    st.subheader("Top Features")

                    top_n = st.slider(
                        "Number of features to display",
                        min_value=5,
                        max_value=len(importance_df),
                        value=min(15, len(importance_df))
                    )

                    top_features = importance_df.head(top_n)

                    fig = px.bar(
                        top_features.sort_values(
                            "Importance",
                            ascending=True
                        ),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title=f"Top {top_n} Feature Importances",
                        labels={
                            "Importance": "Importance Score",
                            "Feature": "Health Indicator"
                        },
                        hover_data={
                            "Importance": ":.4f"
                        }
                    )

                    fig.update_layout(
                        yaxis_title="",
                        xaxis_title="Importance Score"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.subheader("All Feature Importance Scores")

                    st.dataframe(
                        importance_df.style.format({
                            "Importance": "{:.5f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download Feature Importance CSV",
                        data=importance_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="feature_importance.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "Feature importance measures how much "
                        "each feature contributes to the model's "
                        "tree-based decisions. It does not prove "
                        "that a feature causes diabetes or indicate "
                        "the direction of its effect."
                    )

        except Exception as e:
            st.error(
                f"Could not load or analyse the model: {e}"
            )



elif page == "Evaluation":
    st.header("Model Evaluation")
    st.caption(
        "Detailed evaluation of the Random Forest model "
        "at the selected threshold of 0.24."
    )

    data_dir = Path("data")

    required_files = [
        "confusion_matrix.csv",
        "roc_curve.csv",
        "precision_recall_curve.csv",
        "evaluation_summary.csv"
    ]

    missing_files = [
        filename for filename in required_files
        if not (data_dir / filename).exists()
    ]

    if missing_files:
        st.error(
            "The following files are missing from the data folder: "
            + ", ".join(missing_files)
        )
    else:
        cm_df = pd.read_csv(
            data_dir / "confusion_matrix.csv",
            index_col=0
        )
        roc_df = pd.read_csv(data_dir / "roc_curve.csv")
        pr_df = pd.read_csv(
            data_dir / "precision_recall_curve.csv"
        )
        summary = pd.read_csv(
            data_dir / "evaluation_summary.csv"
        )

        row = summary.iloc[0]

        st.subheader("Performance Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{row['Precision']:.3f}")
        c2.metric("Recall", f"{row['Recall']:.3f}")
        c3.metric("F1-score", f"{row['F1-score']:.3f}")
        c4.metric("ROC-AUC", f"{row['ROC-AUC']:.3f}")

        st.metric(
            "Average Precision",
            f"{row['Average Precision']:.4f}"
        )

        st.divider()

        st.subheader("Confusion Matrix")

        fig_cm = px.imshow(
            cm_df.values,
            x=cm_df.columns.tolist(),
            y=cm_df.index.tolist(),
            text_auto=",",
            color_continuous_scale="Blues",
            aspect="auto",
            title="Actual vs Predicted Class"
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.caption(
            "The matrix shows correct predictions and false "
            "positives/negatives on the held-out test set."
        )

        st.divider()

        st.subheader("ROC Curve")

        fig_roc = px.line(
            roc_df,
            x="False Positive Rate",
            y="True Positive Rate",
            title=f"ROC Curve (AUC = {row['ROC-AUC']:.3f})"
        )
        fig_roc.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="gray")
        )
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()

        st.subheader("Precision–Recall Curve")

        fig_pr = px.line(
            pr_df,
            x="Recall",
            y="Precision",
            title=(
                "Precision–Recall Curve "
                f"(Average Precision = "
                f"{row['Average Precision']:.3f})"
            )
        )
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision"
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        st.info(
            "Educational project only. These model outputs are "
            "not intended for medical diagnosis or treatment."
        )


elif page == "Prediction Demo":
    st.title("Diabetes Prediction Demo")
    st.write(
        "Explore how a machine learning model classifies health indicator "
        "data. This is an educational demonstration, not a medical diagnosis."
    )

    model_path = Path("diabetes_random_forest_compressed.joblib")

    try:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        feature_names = bundle["feature_names"]
        threshold = bundle["threshold"]

        st.subheader("Enter health indicators")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                high_bp = st.selectbox("High blood pressure?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                high_chol = st.selectbox("High cholesterol?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                chol_check = st.selectbox("Cholesterol check in the past 5 years?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                bmi = st.number_input("BMI", min_value=1, max_value=100, value=27)
                smoker = st.selectbox("Have you smoked at least 100 cigarettes in your life?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                stroke = st.selectbox("Ever told you had a stroke?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                heart_disease = st.selectbox("Heart disease or heart attack history?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                physical_activity = st.selectbox("Any physical activity in the past 30 days?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                fruits = st.selectbox("Consume fruit one or more times per day?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                veggies = st.selectbox("Consume vegetables one or more times per day?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                heavy_alcohol = st.selectbox("Heavy alcohol consumption?", [0, 1], format_func=lambda x: "Yes" if x else "No")

            with col2:
                healthcare = st.selectbox("Have any healthcare coverage?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                no_doc_cost = st.selectbox("Could not see a doctor due to cost?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                general_health = st.selectbox("General health (1 = excellent, 5 = poor)", [1, 2, 3, 4, 5], index=1)
                mental_health = st.slider("Days mental health was not good (past 30 days)", 0, 30, 0)
                physical_health = st.slider("Days physical health was not good (past 30 days)", 0, 30, 0)
                difficulty_walking = st.selectbox("Serious difficulty walking?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                sex = st.selectbox("Sex category as encoded in dataset", [0, 1])
                age = st.selectbox("Age category (1–13)", list(range(1, 14)), index=7)
                education = st.selectbox("Education category (1–6)", list(range(1, 7)), index=4)
                income = st.selectbox("Income category (1–8)", list(range(1, 9)), index=4)

            submitted = st.form_submit_button("Run prediction")

        if submitted:
            input_values = {
                "HighBP": high_bp,
                "HighChol": high_chol,
                "CholCheck": chol_check,
                "BMI": bmi,
                "Smoker": smoker,
                "Stroke": stroke,
                "HeartDiseaseorAttack": heart_disease,
                "PhysActivity": physical_activity,
                "Fruits": fruits,
                "Veggies": veggies,
                "HvyAlcoholConsump": heavy_alcohol,
                "AnyHealthcare": healthcare,
                "NoDocbcCost": no_doc_cost,
                "GenHlth": general_health,
                "MentHlth": mental_health,
                "PhysHlth": physical_health,
                "DiffWalk": difficulty_walking,
                "Sex": sex,
                "Age": age,
                "Education": education,
                "Income": income,
            }

            input_df = pd.DataFrame(
                [[input_values[name] for name in feature_names]],
                columns=feature_names,
            )

            probability = float(model.predict_proba(input_df)[0, 1])
            prediction = int(probability >= threshold)

            st.subheader("Model output")
            st.metric("Model-estimated probability", f"{probability:.1%}")
            st.write(f"Classification threshold: {threshold:.2f}")

            if prediction == 1:
                st.warning("The model classified this input as the positive class.")
            else:
                st.success("The model classified this input as the negative class.")

            st.caption(
                "This output reflects a statistical model trained on population data. "
                "It is not a diagnosis, a validated personal risk estimate, or medical advice."
            )

    except FileNotFoundError:
        st.error("Model file not found. Expected: diabetes_random_forest_compressed.joblib")
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        st.error(f"Could not load or use the model bundle: {error}")
