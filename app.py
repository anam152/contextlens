from __future__ import annotations
from analysis import analyse_dataset, build_context_narrative, infer_task_type
from modeling import available_models, train_and_compare_models
from reporting import build_markdown_report

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import (
    analyse_dataset,
    build_context_narrative,
    infer_task_type,
)

from modeling import (
    available_models,
    train_and_compare_models,
)
from reporting import build_markdown_report
APP_DIR = Path(__file__).parent
SAMPLE_DATASETS = {
    "Iris (generic)": APP_DIR / "sample_data" / "iris_contextlens.csv",
    "Heart Disease (healthcare)": APP_DIR / "sample_data" / "heart_disease.csv",
}


st.set_page_config(
    page_title="ContextLens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        .context-card {
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: .75rem;
        }
        .small-note {opacity: .75; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Create a stable fingerprint for session-state result invalidation."""
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_sample_csv() -> pd.DataFrame:
    return pd.read_csv(path)


def show_issue_cards(issues: list[dict]) -> None:
    if not issues:
        st.success("No major structural data-quality issues were detected.")
        return

    icon_map = {"high": "🔴", "medium": "🟠", "low": "🔵"}
    for issue in issues:
        icon = icon_map.get(issue["severity"], "•")
        st.markdown(
            f"""
            <div class="context-card">
                <strong>{icon} {issue["title"]}</strong><br>
                {issue["detail"]}<br>
                <span class="small-note"><strong>Suggested action:</strong> {issue["recommendation"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.title("🔎 ContextLens")
st.caption(
    "A context-aware machine-learning explorer for tabular datasets. "
    "Upload data, inspect its modelling context, compare baseline models, "
    "and export an evidence-based report."
)

with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Choose a source",
        ["Upload CSV", "Use sample dataset"],
        help="The sample dataset is an enriched Iris classification dataset.",
    )

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV, or select the sample dataset.")
            st.stop()
        try:
            df = read_uploaded_csv(uploaded.getvalue())
            source_name = uploaded.name
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()
    else:
        sample_choice = st.selectbox(
            "Sample dataset",
            options=list(SAMPLE_DATASETS.keys()),
            help="Heart Disease demonstrates ContextLens on a healthcare classification task.",
        )
        sample_path = SAMPLE_DATASETS[sample_choice]
        df = read_sample_csv(sample_path)
        source_name = sample_path.name
        

    if df.empty:
        st.error("The dataset has no rows.")
        st.stop()

    default_target = (
        "species"
        if "species" in df.columns
        else "target" if "target" in df.columns
        else df.columns[-1]
    )
    
    target = st.selectbox(
        "Target column",
        options=list(df.columns),
        index=list(df.columns).index(default_target),
        help="The variable the models should predict.",
    )

    inferred_task, inference_reason = infer_task_type(df[target])
    task_choice = st.selectbox(
        "Problem type",
        ["Auto-detect", "Classification", "Regression"],
        index=0,
    )
    task = inferred_task if task_choice == "Auto-detect" else task_choice.lower()

    st.caption(f"Detected: **{inferred_task.title()}** — {inference_reason}")

    with st.expander("Run settings"):
        test_size = st.slider("Test-set size", 0.15, 0.40, 0.25, 0.05)
        random_state = st.number_input("Random seed", 0, 100_000, 42)
        row_limit = st.number_input(
            "Maximum training rows",
            min_value=500,
            max_value=100_000,
            value=20_000,
            step=500,
            help="Larger datasets are sampled for a responsive portfolio demo.",
        )

analysis = analyse_dataset(df, target=target, task=task)
narrative = build_context_narrative(analysis)

metric_cols = st.columns(5)
metric_cols[0].metric("Rows", f"{analysis['shape']['rows']:,}")
metric_cols[1].metric("Features", f"{analysis['shape']['features']:,}")
metric_cols[2].metric("Missing cells", f"{analysis['missing']['total_cells']:,}")
metric_cols[3].metric("Duplicates", f"{analysis['duplicates']:,}")
metric_cols[4].metric("Task", task.title())

tabs = st.tabs(
    [
        "Dataset overview",
        "Context analysis",
        "Model laboratory",
        "Feature analysis",
        "Export report",
    ]
)

with tabs[0]:
    st.subheader("Dataset overview")
    st.caption(f"Source: {source_name}")

    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Column roles")
        roles_df = pd.DataFrame(
            {
                "Role": ["Numeric features", "Categorical features", "Datetime-like", "Target"],
                "Count / column": [
                    len(analysis["columns"]["numeric"]),
                    len(analysis["columns"]["categorical"]),
                    len(analysis["columns"]["datetime_like"]),
                    target,
                ],
            }
        )
        st.dataframe(roles_df, hide_index=True, use_container_width=True)

    with right:
        st.markdown("#### Missing values")
        missing_df = (
            pd.DataFrame(
                {
                    "column": list(analysis["missing"]["by_column"].keys()),
                    "missing": list(analysis["missing"]["by_column"].values()),
                }
            )
            .query("missing > 0")
            .sort_values("missing", ascending=False)
        )
        if missing_df.empty:
            st.success("No missing values.")
        else:
            st.bar_chart(missing_df.set_index("column"))

    if task == "classification":
        st.markdown("#### Target distribution")
        distribution = df[target].astype("string").fillna("<missing>").value_counts()
        st.bar_chart(distribution)
    else:
        st.markdown("#### Target summary")
        st.dataframe(
            df[target].describe().to_frame("value"),
            use_container_width=True,
        )

    with st.expander("Descriptive statistics"):
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with tabs[1]:
    st.subheader("Context analysis")
    st.write(narrative["summary"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What ContextLens inferred")
        for item in narrative["observations"]:
            st.markdown(f"- {item}")

    with c2:
        st.markdown("#### Evaluation guidance")
        for item in narrative["evaluation_guidance"]:
            st.markdown(f"- {item}")

    st.markdown("#### Data-quality and modelling issues")
    show_issue_cards(analysis["issues"])

    st.info(
        "ContextLens uses explicit, inspectable heuristics. Its recommendations "
        "support—not replace—domain expertise and proper experimental design."
    )

with tabs[2]:
    st.subheader("Model laboratory")

    model_options = available_models(task)
    selected_models = st.multiselect(
        "Choose baseline models",
        options=list(model_options.keys()),
        default=list(model_options.keys()),
        help="These are deliberately interpretable baseline families, not a full AutoML search.",
    )

    run_key = (
        dataframe_fingerprint(df),
        target,
        task,
        tuple(selected_models),
        float(test_size),
        int(random_state),
        int(row_limit),
    )

    if st.button(
        "Train and compare models",
        type="primary",
        disabled=not selected_models,
        use_container_width=True,
    ):
        with st.spinner("Training reproducible preprocessing-and-model pipelines…"):
            try:
                result = train_and_compare_models(
                    df=df,
                    target=target,
                    task=task,
                    model_names=selected_models,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    row_limit=int(row_limit),
                )
                st.session_state["contextlens_result"] = result
                st.session_state["contextlens_run_key"] = run_key
            except Exception as exc:
                st.exception(exc)

    result = st.session_state.get("contextlens_result")
    result_key = st.session_state.get("contextlens_run_key")

    if result is None or result_key != run_key:
        st.caption("Run the laboratory to produce results for the current settings.")
    else:
        st.success(
            f"Best baseline: **{result['best_model']}** "
            f"using **{result['primary_metric']}**."
        )
        results_df = pd.DataFrame(result["leaderboard"])
        st.dataframe(
            results_df.style.format(precision=4),
            hide_index=True,
            use_container_width=True,
        )

        chart_df = results_df.set_index("model")[[result["primary_metric"]]]
        st.bar_chart(chart_df)

        if result["warnings"]:
            for warning in result["warnings"]:
                st.warning(warning)

        for model_name, details in result["details"].items():
            with st.expander(f"{model_name}: diagnostics"):
                st.write(details["summary"])
                if details.get("confusion_matrix") is not None:
                    st.markdown("**Confusion matrix**")
                    st.dataframe(
                        pd.DataFrame(
                            details["confusion_matrix"],
                            index=details["class_labels"],
                            columns=details["class_labels"],
                        ),
                        use_container_width=True,
                    )
                if details.get("classification_report") is not None:
                    st.markdown("**Per-class report**")
                    report_df = pd.DataFrame(details["classification_report"]).transpose()
                    st.dataframe(report_df, use_container_width=True)

with tabs[3]:
    st.subheader("Feature analysis")
    result = st.session_state.get("contextlens_result")
    result_key = st.session_state.get("contextlens_run_key")

    if result is None or result_key != run_key:
        st.info("Train the selected models first.")
    else:
        importance = result.get("feature_importance")
        if not importance:
            st.info(
                "The best model did not expose stable coefficients or feature importances."
            )
        else:
            importance_df = pd.DataFrame(importance)
            st.caption(
                f"Global importance from the best baseline: {result['best_model']}. "
                "Encoded categories appear as separate features."
            )
            st.bar_chart(importance_df.set_index("feature")["importance"])
            st.dataframe(
                importance_df,
                hide_index=True,
                use_container_width=True,
            )
            st.warning(
                "Global importance describes model behaviour, not causality. "
                "Correlated variables can share or distort importance."
            )

with tabs[4]:
    st.subheader("Export report")
    result = st.session_state.get("contextlens_result")
    result_key = st.session_state.get("contextlens_run_key")
    valid_result = result if result is not None and result_key == run_key else None

    report = build_markdown_report(
        source_name=source_name,
        target=target,
        task=task,
        analysis=analysis,
        narrative=narrative,
        modelling_result=valid_result,
    )

    st.download_button(
        "Download Markdown report",
        data=report,
        file_name="contextlens_report.md",
        mime="text/markdown",
        on_click="ignore",
        use_container_width=True,
    )
    st.code(report, language="markdown")
