"""
Credit Risk Modeling — Streamlit Dashboard
Run: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure src/ is on path so project modules are importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import io
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resource loaders ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_scoring_engine():
    from credit_risk_modeling.scoring import load_model_and_preprocessor
    return load_model_and_preprocessor()


@st.cache_resource(show_spinner=False)
def get_rule_engine():
    from credit_risk_modeling.decision_rules import ApprovalRuleEngine
    return ApprovalRuleEngine()


@st.cache_resource(show_spinner=False)
def get_explainability_engine():
    from credit_risk_modeling.explainibility import ExplainabilityEngine
    return ExplainabilityEngine()


# ── Sidebar navigation ─────────────────────────────────────────────────────────
PAGES = [
    "📋 Dashboard",
    "👤 Score Applicant",
    "📦 Batch Scoring",
    "📈 Portfolio Analytics",
    "🔍 Model Explainability",
    "🔔 Model Monitor",
]

with st.sidebar:
    st.title("Credit Risk Platform")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Powered by LightGBM + scikit-learn")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📋 Dashboard":
    st.title("📋 Credit Risk Dashboard")
    st.markdown("Overview of the credit risk modeling platform and portfolio health.")

    # Try to load existing processed results
    from credit_risk_modeling.config import PROCESSED_DATA_DIR, MODELS_DIR

    results_path = PROCESSED_DATA_DIR / "test_results.csv"

    if results_path.exists():
        df = pd.read_csv(results_path)

        total = len(df)
        approved = (df["decision"] == "APPROVE").sum() if "decision" in df.columns else None
        denied = (df["decision"] == "DENY").sum() if "decision" in df.columns else None
        avg_pd = df["pd"].mean() if "pd" in df.columns else None
        total_el = df["expected_loss"].sum() if "expected_loss" in df.columns else None
        total_ead = df["ead"].sum() if "ead" in df.columns else None

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Applicants", f"{total:,}")
        if approved is not None:
            col2.metric("Approved", f"{approved:,}", f"{approved/total*100:.1f}%")
            col3.metric("Denied", f"{denied:,}", f"-{denied/total*100:.1f}%", delta_color="inverse")
        if avg_pd is not None:
            col4.metric("Avg PD", f"{avg_pd*100:.2f}%")
        if total_el is not None:
            col5.metric("Total Expected Loss", f"${total_el:,.0f}")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        if "risk_tier" in df.columns:
            with col_a:
                st.subheader("Risk Tier Distribution")
                tier_counts = df["risk_tier"].value_counts()
                fig, ax = plt.subplots(figsize=(4, 4))
                colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
                ax.pie(
                    tier_counts.values,
                    labels=tier_counts.index,
                    autopct="%1.1f%%",
                    colors=[colors.get(t, "#95a5a6") for t in tier_counts.index],
                    startangle=90,
                )
                ax.axis("equal")
                st.pyplot(fig)
                plt.close(fig)

        if "loan_intent" in df.columns and "pd" in df.columns:
            with col_b:
                st.subheader("Avg PD by Loan Intent")
                intent_pd = df.groupby("loan_intent")["pd"].mean().sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(5, 4))
                bars = ax.barh(intent_pd.index, intent_pd.values * 100, color="#3498db")
                ax.set_xlabel("Average PD (%)")
                ax.bar_label(bars, fmt="%.1f%%", padding=3)
                ax.set_xlim(0, intent_pd.values.max() * 130)
                st.pyplot(fig)
                plt.close(fig)

    else:
        st.info(
            "No processed results found. Run **Batch Scoring** to generate portfolio data, "
            "or use **Score Applicant** to score individual applicants."
        )

    # Model status
    st.markdown("---")
    st.subheader("Model Status")
    model_file = MODELS_DIR / "best_tree_model.pkl"
    preprocessor_file = MODELS_DIR / "tree_preprocessed_pipeline.pkl"
    lgd_file = MODELS_DIR / "lgd_by_segment.json"

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Classifier",
        "✅ Loaded" if model_file.exists() else "❌ Missing",
        "best_tree_model.pkl",
    )
    col2.metric(
        "Preprocessor",
        "✅ Loaded" if preprocessor_file.exists() else "❌ Missing",
        "tree_preprocessed_pipeline.pkl",
    )
    col3.metric(
        "LGD Estimates",
        "✅ Loaded" if lgd_file.exists() else "❌ Missing",
        "lgd_by_segment.json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SCORE APPLICANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Score Applicant":
    st.title("👤 Score Individual Applicant")
    st.markdown("Enter applicant details to generate a real-time credit risk assessment.")

    with st.form("applicant_form"):
        st.subheader("Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=90, value=30)
        with col2:
            person_income = st.number_input(
                "Annual Income ($)", min_value=0, max_value=10_000_000, value=60_000, step=1_000
            )
        with col3:
            person_home_ownership = st.selectbox(
                "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
            )

        col4, col5 = st.columns(2)
        with col4:
            person_emp_length = st.number_input(
                "Employment Length (years)", min_value=0.0, max_value=50.0, value=3.0, step=0.5
            )
        with col5:
            cb_person_default_on_file = st.selectbox(
                "Prior Default on File", ["N", "Y"]
            )

        cb_person_cred_hist_length = st.slider(
            "Credit History Length (years)", min_value=0, max_value=30, value=5
        )

        st.subheader("Loan Details")
        col6, col7, col8 = st.columns(3)
        with col6:
            loan_intent = st.selectbox(
                "Loan Intent",
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            )
        with col7:
            loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        with col8:
            loan_amnt = st.number_input(
                "Loan Amount ($)", min_value=100, max_value=100_000, value=10_000, step=500
            )

        col9, col10 = st.columns(2)
        with col9:
            loan_int_rate = st.number_input(
                "Interest Rate (%)", min_value=0.0, max_value=35.0, value=10.0, step=0.1
            )
        with col10:
            loan_percent_income = st.number_input(
                "Loan / Income Ratio", min_value=0.0, max_value=1.0,
                value=round(loan_amnt / max(person_income, 1), 2), step=0.01
            )

        phase = st.radio("LGD Phase", [2, 3], horizontal=True,
                         help="Phase 2: simplified 100% LGD | Phase 3: segment-based LGD from JSON")

        submitted = st.form_submit_button("🔍 Score Applicant", use_container_width=True)

    if submitted:
        features = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }

        try:
            from credit_risk_modeling.scoring import score_applicant
            from credit_risk_modeling.decision_rules import ApprovalRuleEngine

            with st.spinner("Scoring applicant…"):
                result = score_applicant(features, phase=phase)
                engine = get_rule_engine()
                decision = engine.decide(
                    pd=result["pd"],
                    lgd=result["lgd"],
                    ead=result["ead"],
                    expected_loss=result["expected_loss"],
                    loan_intent=loan_intent,
                )

            st.markdown("---")
            st.subheader("Assessment Results")

            # Decision banner
            decision_val = decision["decision"]
            if decision_val == "APPROVE":
                st.success(f"✅ **APPROVED** — {decision['reason']}")
            elif decision_val == "MANUAL_REVIEW":
                st.warning(f"⚠️ **MANUAL REVIEW** — {decision['reason']}")
            else:
                st.error(f"❌ **DENIED** — {decision['reason']}")

            # Risk score gauge row
            col1, col2, col3, col4, col5 = st.columns(5)
            risk_tier = result["risk_tier"]
            tier_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(risk_tier, "gray")

            col1.metric("Risk Score", f"{result['risk_score']} / 100")
            col2.metric("Risk Tier", risk_tier)
            col3.metric("Probability of Default", f"{result['pd']*100:.2f}%")
            col4.metric("Confidence", f"{result['confidence']*100:.1f}%")
            col5.metric("EL / EAD", f"{decision['el_pct']*100:.2f}%")

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Loss Metrics")
                loss_df = pd.DataFrame(
                    {
                        "Metric": ["Probability of Default (PD)", "Loss Given Default (LGD)", "Exposure at Default (EAD)", "Expected Loss (EL)"],
                        "Value": [
                            f"{result['pd']*100:.3f}%",
                            f"{result['lgd']*100:.1f}%",
                            f"${result['ead']:,.2f}",
                            f"${result['expected_loss']:,.2f}",
                        ],
                    }
                )
                st.dataframe(loss_df, use_container_width=True, hide_index=True)

            with col_b:
                st.subheader("PD Gauge")
                fig, ax = plt.subplots(figsize=(4, 2.5))
                pd_val = result["pd"]
                bar_color = "#e74c3c" if pd_val > 0.5 else "#f39c12" if pd_val > 0.2 else "#2ecc71"
                ax.barh(["PD"], [pd_val], color=bar_color, height=0.4)
                ax.barh(["PD"], [1 - pd_val], left=[pd_val], color="#ecf0f1", height=0.4)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability of Default")
                ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
                ax.set_title(f"PD = {pd_val*100:.2f}%")
                ax.legend(fontsize=8)
                st.pyplot(fig)
                plt.close(fig)

            # Explainability
            st.subheader("Key Risk Drivers")
            try:
                engine_expl = get_explainability_engine()
                explanation = engine_expl.explain_prediction(features)
                factors_df = pd.DataFrame(explanation["top_factors"])
                if not factors_df.empty:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.barh(
                        factors_df["feature"].head(10),
                        factors_df["importance"].head(10),
                        color="#3498db",
                    )
                    ax.set_xlabel("Feature Importance")
                    ax.set_title("Top Contributing Features")
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception:
                st.info("Explainability details unavailable. See Model Explainability page.")

        except FileNotFoundError:
            st.error(
                "Model files not found. Ensure `best_tree_model.pkl` and "
                "`tree_preprocessed_pipeline.pkl` exist in the `models/` directory."
            )
        except Exception as e:
            st.error(f"Scoring error: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Scoring":
    st.title("📦 Batch Scoring")
    st.markdown(
        "Upload a CSV of applicants to score in bulk. "
        "Download a sample template to see the expected format."
    )

    # Sample template download
    sample_data = {
        "person_age": [25, 35, 45],
        "person_income": [60000, 80000, 30000],
        "person_home_ownership": ["RENT", "OWN", "RENT"],
        "person_emp_length": [3.0, 5.0, 0.5],
        "loan_intent": ["EDUCATION", "HOMEIMPROVEMENT", "PERSONAL"],
        "loan_grade": ["B", "C", "F"],
        "loan_amnt": [5000, 15000, 25000],
        "loan_int_rate": [9.5, 12.5, 20.0],
        "loan_percent_income": [0.08, 0.19, 0.83],
        "cb_person_default_on_file": ["N", "N", "Y"],
        "cb_person_cred_hist_length": [5, 8, 2],
    }
    sample_csv = pd.DataFrame(sample_data).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Sample Template",
        data=sample_csv,
        file_name="applicants_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Applicants CSV", type=["csv"])
    phase = st.radio("LGD Phase", [2, 3], horizontal=True,
                     help="Phase 2: 100% LGD | Phase 3: segment-based LGD")

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.write(f"**{len(df_input)} applicants loaded.** Preview:")
            st.dataframe(df_input.head(), use_container_width=True)

            if st.button("▶️ Run Batch Scoring", use_container_width=True):
                from credit_risk_modeling.scoring import score_batch
                from credit_risk_modeling.decision_rules import ApprovalRuleEngine

                with st.spinner(f"Scoring {len(df_input)} applicants…"):
                    applicants_list = df_input.to_dict(orient="records")
                    scoring_results = score_batch(applicants_list, phase=phase)
                    scores_df = pd.DataFrame(scoring_results)

                    engine = get_rule_engine()
                    decisions = engine.batch_decide(scoring_results)
                    decisions_df = pd.DataFrame(decisions)

                # Merge input + scores + decisions
                results_df = pd.concat(
                    [df_input.reset_index(drop=True), decisions_df.reset_index(drop=True)],
                    axis=1,
                )

                # Drop duplicate columns from merge
                results_df = results_df.loc[:, ~results_df.columns.duplicated()]

                st.success(f"✅ Scoring complete for {len(results_df)} applicants.")
                st.session_state["batch_results"] = results_df

                # Quick stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Approved", int((results_df["decision"] == "APPROVE").sum()))
                col2.metric("Manual Review", int((results_df["decision"] == "MANUAL_REVIEW").sum()))
                col3.metric("Denied", int((results_df["decision"] == "DENY").sum()))
                col4.metric("Avg PD", f"{results_df['pd'].mean()*100:.2f}%")

                st.dataframe(results_df, use_container_width=True)

                # Download
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Scored Results",
                    data=csv_bytes,
                    file_name="scored_applicants.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error during batch scoring: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Portfolio Analytics":
    st.title("📈 Portfolio Analytics")

    from credit_risk_modeling.config import PROCESSED_DATA_DIR

    # Source selector
    source = st.radio(
        "Data Source",
        ["Use batch scoring results (from this session)", "Upload scored CSV", "Load test_results.csv"],
        horizontal=True,
    )

    df = None
    if source == "Use batch scoring results (from this session)":
        if "batch_results" in st.session_state:
            df = st.session_state["batch_results"]
            st.success(f"Using {len(df)} records from current session batch scoring.")
        else:
            st.info("No batch results in session yet. Run Batch Scoring first, or choose another source.")

    elif source == "Upload scored CSV":
        uploaded = st.file_uploader("Upload scored CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)

    else:
        results_path = PROCESSED_DATA_DIR / "test_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            st.success(f"Loaded {len(df)} records from test_results.csv.")
        else:
            st.warning("test_results.csv not found in data/processed/.")

    if df is not None and not df.empty:
        required_cols = {"decision", "pd", "lgd", "ead", "expected_loss", "risk_tier"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"CSV is missing required columns: {missing}")
        else:
            from credit_risk_modeling.portfolio_analytics import PortfolioAnalyzer

            analyzer = PortfolioAnalyzer()
            metrics = analyzer.analyze(df)

            # KPI tiles
            st.subheader("Portfolio KPIs")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Applicants", f"{metrics.total_applicants:,}")
            col2.metric("Approval Rate", f"{metrics.approval_rate*100:.1f}%")
            col3.metric("Denial Rate", f"{metrics.denial_rate*100:.1f}%")
            col4.metric("Avg PD", f"{metrics.avg_pd*100:.2f}%")
            col5.metric("Total Expected Loss", f"${metrics.total_expected_loss:,.0f}")

            col6, col7, col8 = st.columns(3)
            col6.metric("Avg LGD", f"{metrics.avg_lgd*100:.1f}%")
            col7.metric("Avg EAD", f"${metrics.avg_ead:,.0f}")
            col8.metric("Total EAD", f"${metrics.total_ead:,.0f}")

            st.markdown("---")

            col_a, col_b = st.columns(2)

            # Risk tier breakdown
            with col_a:
                st.subheader("Risk Tier Breakdown")
                tier_data = metrics.by_risk_tier
                tier_df = pd.DataFrame(tier_data).T
                if "count" in tier_df.columns:
                    tier_df["count"] = tier_df["count"].astype(int)
                st.dataframe(tier_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(4, 4))
                colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
                counts = {k: v.get("count", 0) for k, v in tier_data.items()}
                ax.pie(
                    list(counts.values()),
                    labels=list(counts.keys()),
                    autopct="%1.1f%%",
                    colors=[colors.get(k, "#95a5a6") for k in counts.keys()],
                    startangle=90,
                )
                ax.axis("equal")
                st.pyplot(fig)
                plt.close(fig)

            # Loan intent breakdown
            with col_b:
                st.subheader("By Loan Intent")
                if metrics.by_loan_intent:
                    intent_df = pd.DataFrame(metrics.by_loan_intent).T
                    if "avg_pd" in intent_df.columns:
                        intent_df["avg_pd_pct"] = (intent_df["avg_pd"].astype(float) * 100).round(2)
                    st.dataframe(intent_df, use_container_width=True)

                    if "avg_pd" in intent_df.columns:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sorted_df = intent_df["avg_pd_pct"].sort_values()
                        ax.barh(sorted_df.index, sorted_df.values, color="#3498db")
                        ax.set_xlabel("Avg PD (%)")
                        ax.set_title("Average PD by Loan Intent")
                        st.pyplot(fig)
                        plt.close(fig)

            st.markdown("---")
            st.subheader("Decision Breakdown")
            if metrics.by_decision:
                dec_df = pd.DataFrame(metrics.by_decision).T
                st.dataframe(dec_df, use_container_width=True)

            # Raw text report
            with st.expander("📄 Full ASCII Report"):
                st.code(analyzer.report(metrics))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Model Explainability":
    st.title("🔍 Model Explainability")

    try:
        engine = get_explainability_engine()

        st.subheader("Global Feature Importance")
        top_n = st.slider("Top N features to display", min_value=5, max_value=30, value=15)

        with st.spinner("Computing feature importance…"):
            fi_df = engine.get_feature_importance()

        if fi_df is not None and not fi_df.empty:
            fi_top = fi_df.head(top_n)
            fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
            ax.barh(fi_top["feature"][::-1], fi_top["importance"][::-1], color="#3498db")
            ax.set_xlabel("Importance Score")
            ax.set_title(f"Top {top_n} Feature Importances")
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(fi_df, use_container_width=True, hide_index=True)

            # Download
            csv_bytes = fi_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Feature Importance CSV",
                data=csv_bytes,
                file_name="feature_importance.csv",
                mime="text/csv",
            )
        else:
            st.warning("Feature importance data unavailable.")

        st.markdown("---")
        st.subheader("Explainability Report")
        with st.spinner("Generating summary report…"):
            report_text = engine.generate_summary_report()
        st.text_area("Report", report_text, height=300)

    except FileNotFoundError:
        st.error(
            "Model files not found. Ensure `best_tree_model.pkl` and "
            "`tree_preprocessed_pipeline.pkl` exist in the `models/` directory."
        )
    except Exception as e:
        st.error(f"Error loading explainability engine: {e}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔔 Model Monitor":
    st.title("🔔 Model Monitor")
    st.markdown(
        "Detect data drift and data quality issues in a batch of scored applicants. "
        "Upload a scored CSV or run monitoring on the current session's batch results."
    )

    from credit_risk_modeling.config import PROCESSED_DATA_DIR
    from credit_risk_modeling.model_monitor import ModelMonitor

    st.subheader("Baseline Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        baseline_approval = st.number_input(
            "Baseline Approval Rate", min_value=0.0, max_value=1.0, value=0.65, step=0.01
        )
    with col2:
        baseline_pd = st.number_input(
            "Baseline Avg PD", min_value=0.0, max_value=1.0, value=0.15, step=0.01
        )
    with col3:
        baseline_confidence = st.number_input(
            "Baseline Avg Confidence", min_value=0.0, max_value=1.0, value=0.85, step=0.01
        )

    source = st.radio(
        "Data Source",
        ["Current session batch results", "Upload scored CSV", "Load test_results.csv"],
        horizontal=True,
    )

    df = None
    if source == "Current session batch results":
        if "batch_results" in st.session_state:
            df = st.session_state["batch_results"]
        else:
            st.info("No batch results in session. Run Batch Scoring first.")
    elif source == "Upload scored CSV":
        uploaded = st.file_uploader("Upload scored CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
    else:
        results_path = PROCESSED_DATA_DIR / "test_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)

    if df is not None:
        monitor = ModelMonitor(
            baseline_metrics={
                "approval_rate": baseline_approval,
                "avg_pd": baseline_pd,
                "avg_confidence": baseline_confidence,
            }
        )

        if st.button("▶️ Run Monitoring Check", use_container_width=True):
            with st.spinner("Running monitoring checks…"):
                try:
                    monitoring = monitor.monitor_batch(df)
                except Exception as e:
                    st.error(f"Monitoring error: {e}")
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc())
                    st.stop()

            st.markdown("---")
            st.subheader("Monitoring Results")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Predictions", f"{monitoring.total_predictions:,}")
            col2.metric("Approval Rate", f"{monitoring.approval_rate*100:.1f}%",
                        f"{(monitoring.approval_rate - baseline_approval)*100:+.1f}%")
            col3.metric("Avg PD", f"{monitoring.avg_pd*100:.2f}%",
                        f"{(monitoring.avg_pd - baseline_pd)*100:+.2f}%")
            col4.metric("Avg Confidence", f"{monitoring.avg_confidence*100:.1f}%",
                        f"{(monitoring.avg_confidence - baseline_confidence)*100:+.1f}%")

            col5, col6 = st.columns(2)
            col5.metric("Missing Values %", f"{monitoring.missing_values_pct*100:.2f}%")
            col6.metric("Outlier Count", monitoring.outlier_count)

            st.markdown("---")
            st.subheader("Drift Alerts")
            if monitoring.alerts:
                for alert in monitoring.alerts:
                    st.warning(f"⚠️ {alert}")
            else:
                st.success("✅ No significant drift detected.")

            # Drift summary table
            drift_df = pd.DataFrame(
                {
                    "Metric": ["Approval Rate", "Avg PD", "Avg Confidence"],
                    "Baseline": [
                        f"{baseline_approval*100:.1f}%",
                        f"{baseline_pd*100:.2f}%",
                        f"{baseline_confidence*100:.1f}%",
                    ],
                    "Current": [
                        f"{monitoring.approval_rate*100:.1f}%",
                        f"{monitoring.avg_pd*100:.2f}%",
                        f"{monitoring.avg_confidence*100:.1f}%",
                    ],
                    "Drift": [
                        f"{monitoring.approval_rate_drift*100:+.1f}%",
                        f"{monitoring.pd_drift*100:+.2f}%",
                        f"{monitoring.confidence_drift*100:+.1f}%",
                    ],
                }
            )
            st.dataframe(drift_df, use_container_width=True, hide_index=True)

            st.caption(f"Monitoring timestamp: {monitoring.timestamp}")
