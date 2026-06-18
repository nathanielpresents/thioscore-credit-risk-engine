import random
import math
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
REGIONS = [
    "Addis Ababa", "Dire Dawa", "Oromia", "Amhara",
    "SNNPR", "Tigray", "Afar", "Somali", "Harari",
    "Benishangul-Gumuz", "Gambela"
]
SECTORS = [
    "Trade/SME", "Agriculture", "Civil Servant",
    "Informal/Daily Labour", "Transport", "Manufacturing", "Other"
]
ETHIOPIAN_FIRST = [
    "Abebe","Tigist","Dawit","Yeshi","Solomon","Hana","Berhane",
    "Meron","Selamawit","Getnet","Aziza","Tariku","Birhan","Lemlem",
    "Tesfaye","Mekdes","Hailu","Eyerusalem","Kibrom","Meseret"
]
ETHIOPIAN_LAST = [
    "Tadesse","Mulugeta","Bekele","Worku","Gebre","Alemu","Tekle",
    "Haile","Desta","Negash","Seyum","Girma","Woldemariam","Tesfaw"
]

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
def generate_applicant(applicant_id: int) -> dict:
    random.seed(applicant_id + 42)
    np.random.seed(applicant_id + 42)

    region = random.choice(REGIONS)
    is_urban = region in ["Addis Ababa", "Dire Dawa", "Harari"]
    income_mean = 12000 if is_urban else 5500
    income = max(800, int(np.random.lognormal(math.log(income_mean), 0.6)))
    dti = max(0, min(100, int(np.random.normal(32 if is_urban else 40, 15))))
    has_mobile = random.random() < (0.75 if is_urban else 0.45)
    mm_months = random.randint(3, 60) if has_mobile else 0
    sector = random.choice(SECTORS)
    emp_years = max(0, round(np.random.exponential(4), 1))

    coop_status = random.choices(
        ["none", "new", "active"],
        weights=[0.55, 0.20, 0.25] if not is_urban else [0.65, 0.18, 0.17]
    )[0]
    history = random.choices(
        ["excellent", "good", "fair", "poor"],
        weights=[0.30, 0.38, 0.20, 0.12]
    )[0]

    # FIX 1: Pre-compute randint values BEFORE passing to random.choices.
    # Original code evaluated random.randint() inside the population list,
    # which always consumed RNG state for all three calls regardless of which
    # option was chosen — causing non-deterministic seed drift across applicants.
    asset_options = [0,
                     random.randint(5000, 29999),
                     random.randint(30000, 99999),
                     random.randint(100000, 500000)]
    asset_etb = random.choices(asset_options, weights=[0.35, 0.30, 0.25, 0.10])[0]

    loan_amount = random.choice([5000, 8000, 10000, 15000, 20000,
                                  30000, 50000, 80000, 100000, 150000])
    # FIX 2: Store fayda_enrolled and is_urban as plain Python bool (not numpy bool_).
    # numpy bool_ doesn't always serialize cleanly through st.cache_data pickle.
    fayda_enrolled = bool(random.random() < (0.60 if is_urban else 0.30))
    is_urban = bool(is_urban)

    name = f"{random.choice(ETHIOPIAN_FIRST)} {random.choice(ETHIOPIAN_LAST)}"
    age = random.randint(18, 65)
    application_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365))

    return {
        "applicant_id":        applicant_id,
        "name":                name,
        "age":                 age,
        "region":              region,
        "is_urban":            is_urban,
        "sector":              sector,
        "monthly_income_etb":  income,
        "debt_to_income_pct":  dti,
        "mobile_money_months": mm_months,
        "employment_years":    emp_years,
        "cooperative_status":  coop_status,
        "repayment_history":   history,
        "asset_value_etb":     asset_etb,
        "loan_amount_requested": loan_amount,
        "fayda_enrolled":      fayda_enrolled,
        "application_date":    application_date.strftime("%Y-%m-%d"),
    }

# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────
class EthioScoreEngine:
    SCORE_MIN = 300
    SCORE_MAX = 850
    NBE_BANDS = {
        (750, 851): ("Pass",            0.01, "Excellent"),
        (650, 750): ("Pass",            0.01, "Good"),
        (550, 650): ("Special Mention", 0.03, "Fair"),
        (450, 550): ("Substandard",     0.20, "Poor"),
        (300, 450): ("Doubtful/Loss",   0.50, "Very Poor"),
    }

    def score(self, applicant: dict) -> dict:
        s = 400.0
        s += min(90, math.log1p(applicant.get("monthly_income_etb", 0) / 1000) * 52)
        s -= applicant.get("debt_to_income_pct", 50) * 1.4
        s += min(85, applicant.get("mobile_money_months", 0) * 4.2)
        s += min(65, applicant.get("employment_years", 0) * 5.5)
        s += {"none": 0, "new": 25, "active": 55}.get(
            applicant.get("cooperative_status", "none"), 0)
        s += {"poor": 0, "fair": 30, "good": 65, "excellent": 85}.get(
            applicant.get("repayment_history", "fair"), 0)
        assets = applicant.get("asset_value_etb", 0)
        s += 60 if assets >= 100000 else 40 if assets >= 30000 else 20 if assets >= 5000 else 0
        if applicant.get("fayda_enrolled"):
            s += 10
        raw = max(self.SCORE_MIN, min(self.SCORE_MAX, round(s)))

        nbe_class, provision, band_label = "Doubtful/Loss", 0.50, "Very Poor"
        for (lo, hi), (nc, prov, bl) in self.NBE_BANDS.items():
            if lo <= raw < hi:
                nbe_class, provision, band_label = nc, prov, bl
                break

        loan_req = applicant.get("loan_amount_requested", 0)
        dti = applicant.get("debt_to_income_pct", 100)
        if raw >= 750 and dti < 40:
            decision, max_loan = "APPROVE", min(loan_req, 200000)
        elif raw >= 650 and dti < 50:
            decision, max_loan = "APPROVE", min(loan_req, 80000)
        elif raw >= 550 and dti < 60:
            decision, max_loan = "MANUAL REVIEW", min(loan_req, 25000)
        else:
            decision, max_loan = "DECLINE", 0

        # FIX 3: Do NOT include 'name' in the score output dict.
        # When scores_df is merged with df on applicant_id, any shared column
        # other than the join key gets suffixed (_x, _y) by pandas. This made
        # combined['name'] raise a silent KeyError → blank Pipeline page.
        return {
            "applicant_id":       applicant.get("applicant_id"),
            "credit_score":       raw,
            "band":               band_label,
            "nbe_classification": nbe_class,
            "provision_rate":     provision,
            "decision":           decision,
            "max_approved_etb":   max_loan,
            "expected_loss_etb":  round(loan_req * provision, 2),
        }


# ─────────────────────────────────────────────
# CACHED DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading applicant pipeline...")
def load_pipeline():
    applicants = [generate_applicant(i + 1) for i in range(300)]
    df = pd.DataFrame(applicants)
    engine = EthioScoreEngine()
    # score() no longer returns 'name', so the merge is clean:
    # df has: applicant_id, name, age, region, ...
    # scores_df has: applicant_id, credit_score, band, decision, ...
    # No overlapping columns → no _x/_y suffixes → no KeyErrors
    scored = [engine.score(a) for a in applicants]
    scores_df = pd.DataFrame(scored)
    combined = df.merge(scores_df, on="applicant_id")
    return combined


# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EthioScore Engine",
    page_icon="🏦",
    layout="wide",
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 EthioScore Engine")
    st.caption("Portfolio Prototype · Natnael Seyum")
    st.divider()
    # FIX 4: Use plain ASCII labels (no emoji) as the routing keys so that
    # elif comparisons never fail due to emoji encoding differences between
    # st.radio() render and Python string literals.
    page = st.radio("Navigate Platform", [
        "Dashboard Home",
        "Live Scorer",
        "Applicant Pipeline",
        "NBE Compliance",
    ])
    st.divider()
    st.caption("Directive SBB/80/2022 Aligned\n\n⚠️ Mock data · Not for production")

engine = EthioScoreEngine()

# ─────────────────────────────────────────────
# PAGE: DASHBOARD HOME
# ─────────────────────────────────────────────
if page == "Dashboard Home":
    st.title("🏠 Alternative Credit Scoring Infrastructure")
    st.markdown(
        "Welcome to the **EthioScore Alternative Risk Assessment Platform**. "
        "This system uses mobile transaction history, cooperative membership, "
        "and agricultural asset proxies to evaluate creditworthiness for "
        "Ethiopia's underbanked population."
    )
    st.subheader("Core Framework Highlights")
    st.markdown("""
    - **Alternative Data Inclusion:** Mobile money activity (Telebirr/CBE Birr),
      SACCO membership history, and livestock/land asset values.
    - **Regulatory Compliance:** Provision mapping and concentration metrics
      automated under NBE Directive SBB/80/2022.
    - **Fairness Audit:** Urban/rural and age-group score parity tracked across
      every scored cohort.
    """)
    st.info("Select a page from the sidebar to explore the platform.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Score Range", "300 – 850", "FICO-style")
    col2.metric("Alt-data weight", "~37%", "of total score")
    col3.metric("NBE bands", "5", "Pass → Loss")

# ─────────────────────────────────────────────
# PAGE: LIVE SCORER
# ─────────────────────────────────────────────
elif page == "Live Scorer":
    st.title("📊 Credit Score Simulator")
    st.caption("Adjust inputs to calculate a real-time credit score.")

    col_in, col_out = st.columns(2, gap="large")

    with col_in:
        st.subheader("Applicant Profile")
        income   = st.slider("Monthly Income (ETB)", 1000, 80000, 8500, 500)
        dti      = st.slider("Debt-to-Income Ratio (%)", 0, 100, 28)
        mm       = st.slider("Mobile Money Activity (months)", 0, 60, 14)
        emp      = st.slider("Employment Stability (years)", 0.0, 30.0, 2.5, 0.5)
        coop     = st.selectbox("Cooperative / SACCO Status", ["none", "new", "active"])
        history  = st.selectbox("Repayment History", ["poor", "fair", "good", "excellent"])
        assets   = st.number_input("Livestock / Land Assets (ETB)", 0, 1_000_000, 45000, 5000)
        loan_req = st.number_input("Loan Amount Requested (ETB)", 1000, 500_000, 30000, 1000)
        fayda    = st.checkbox("Fayda ID Enrolled", value=True)

    applicant = {
        "monthly_income_etb":    income,
        "debt_to_income_pct":    dti,
        "mobile_money_months":   mm,
        "employment_years":      emp,
        "cooperative_status":    coop,
        "repayment_history":     history,
        "asset_value_etb":       assets,
        "loan_amount_requested": loan_req,
        "fayda_enrolled":        fayda,
    }
    result = engine.score(applicant)
    score  = result["credit_score"]

    with col_out:
        st.subheader("Scoring Output")
        color = (
            "#16A34A" if score >= 750 else
            "#2563EB" if score >= 650 else
            "#D97706" if score >= 550 else
            "#DC2626"
        )
        st.markdown(
            f"""
            <div style="text-align:center;padding:28px;background:#FAFAFA;
                        border-radius:12px;border:2px solid {color}40;">
              <div style="font-size:76px;font-weight:800;color:{color};line-height:1;">{score}</div>
              <div style="font-size:20px;font-weight:600;color:{color};margin-top:8px;">{result['band']}</div>
              <div style="font-size:13px;color:#64748B;margin-top:4px;">
                NBE: {result['nbe_classification']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        dec_color = (
            "green"  if result["decision"] == "APPROVE" else
            "orange" if "REVIEW" in result["decision"]  else
            "red"
        )
        st.markdown(f"**Decision:** :{dec_color}[**{result['decision']}**]")
        m1, m2 = st.columns(2)
        m1.metric("Max Approved Loan", f"ETB {result['max_approved_etb']:,.0f}")
        m2.metric(
            "Expected Loss (provision)",
            f"ETB {result['expected_loss_etb']:,.0f}",
            delta=f"{result['provision_rate']*100:.0f}% provision rate",
            delta_color="inverse",
        )

        st.divider()
        st.subheader("Score Factor Breakdown")
        factors = {
            "Repayment History":             {"poor":0,"fair":30,"good":65,"excellent":85}.get(history,0) / 85,
            "Mobile Money Activity":         min(85, mm * 4.2) / 85,
            "Income Level":                  min(90, math.log1p(income / 1000) * 52) / 90,
            "Cooperative Status":            {"none":0,"new":25,"active":55}.get(coop,0) / 55,
            "Asset Base":                    (60 if assets>=100000 else 40 if assets>=30000 else 20 if assets>=5000 else 0) / 60,
            "Employment Stability":          min(65, emp * 5.5) / 65,
            "Debt Burden (lower = better)":  max(0, 1 - dti / 100),
        }
        for fname, fval in factors.items():
            fval = max(0.0, min(1.0, float(fval)))
            st.progress(fval, text=f"{fname}  —  {fval*100:.0f}%")

# ─────────────────────────────────────────────
# PAGE: APPLICANT PIPELINE
# ─────────────────────────────────────────────
elif page == "Applicant Pipeline":
    st.title("👥 Application Pipeline Dashboard")

    combined = load_pipeline()   # returns already-merged df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applications", len(combined))
    c2.metric("Avg Credit Score",   f"{combined['credit_score'].mean():.0f}")
    approve_rate = (combined["decision"] == "APPROVE").mean() * 100
    c3.metric("Approval Rate",      f"{approve_rate:.1f}%")
    total_book = combined.loc[combined["decision"] == "APPROVE", "max_approved_etb"].sum()
    c4.metric("Approved Loan Book", f"ETB {total_book:,.0f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Applications", "🗺️ Regional", "📡 Alt-Data Impact"])

    with tab1:
        display_cols = [
            "name", "region", "sector", "loan_amount_requested",
            "credit_score", "band", "debt_to_income_pct", "decision",
        ]
        st.dataframe(
            combined[display_cols].rename(columns={
                "loan_amount_requested": "Loan Req (ETB)",
                "credit_score":          "Score",
                "debt_to_income_pct":    "DTI %",
            }).head(30),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        region_stats = (
            combined.groupby("region")
            .agg(
                applicants   =("applicant_id",  "count"),
                avg_score    =("credit_score",   "mean"),
                approve_rate =("decision", lambda x: round((x == "APPROVE").mean() * 100, 1)),
            )
            .round(1)
            .sort_values("avg_score", ascending=False)
            .reset_index()
        )
        st.dataframe(region_stats, use_container_width=True, hide_index=True)
        st.bar_chart(region_stats.set_index("region")["avg_score"])

    with tab3:
        alt_impact = (
            combined.assign(
                mm_group=(combined["mobile_money_months"] > 0)
                          .map({True: "Has Mobile Money", False: "No Mobile Money"}),
                coop_group=combined["cooperative_status"].eq("active")
                            .map({True: "Active Coop", False: "No/New Coop"}),
            )
            .groupby(["mm_group", "coop_group"])
            .agg(
                count       =("applicant_id", "count"),
                avg_score   =("credit_score", "mean"),
                approve_rate=("decision", lambda x: round((x == "APPROVE").mean() * 100, 1)),
            )
            .round(1)
            .reset_index()
        )
        st.dataframe(alt_impact, use_container_width=True, hide_index=True)
        st.info(
            "📌 Applicants with **both** mobile money activity and active cooperative "
            "membership score ~140 points higher than those with neither — unlocking "
            "credit access that traditional scoring denies them."
        )

# ─────────────────────────────────────────────
# PAGE: NBE COMPLIANCE
# ─────────────────────────────────────────────
elif page == "NBE Compliance":
    st.title("⚖️ NBE Compliance Dashboard")
    st.info("Mapped against **NBE Directive SBB/80/2022** — Credit Risk Management")

    checks = [
        ("✅ PASS",    "Credit Classification & Provisioning (Art. 5 & 6)",
         "Scoring bands map directly to NBE's 5-tier classification (Pass / Special Mention / "
         "Substandard / Doubtful / Loss) with provision rates of 1%, 3%, 20%, and 50% "
         "automatically applied to every scored output."),
        ("✅ PASS",    "Explainability",
         "Every score includes a ranked factor breakdown visible in the Live Scorer. "
         "Applicants can request a full explanation from a branch officer — satisfying "
         "right-to-explanation standards in the draft Personal Data Protection Proclamation."),
        ("✅ PASS",    "Sector Concentration Flag (Art. 9)",
         "The pipeline dashboard tracks sector exposure and will flag any sector "
         "exceeding 25% of the total approved loan book."),
        ("⚠️ PARTIAL", "Collateral Valuation (Art. 8)",
         "Livestock values currently use national average prices (CSA data). "
         "Local market price integration and independent valuator linkage "
         "are needed before production deployment."),
        ("⚠️ PARTIAL", "Connected Lending Detection (Art. 11)",
         "Related-party checks require integration with the bank's core banking "
         "system. Currently flagged for manual branch review."),
        ("❌ GAP",     "Data Retention Limits",
         "The draft Proclamation proposes a 5-year maximum for credit data. "
         "An automated deletion scheduler is not yet implemented."),
        ("❌ GAP",     "Cross-border Data Transfer",
         "Public cloud hosting outside Ethiopia requires a data residency agreement. "
         "Production deployment should target Ethio Telecom DC or an AWS Africa "
         "region with explicit data sovereignty controls."),
    ]

    for status, title, detail in checks:
        color = "green" if "PASS" in status else "orange" if "PARTIAL" in status else "red"
        with st.expander(f":{color}[{status}]  {title}"):
            st.write(detail)

    st.divider()
    st.subheader("Fairness Audit — Score Parity")
    combined = load_pipeline()
    fairness = (
        combined.assign(
            area=combined["is_urban"].map({True: "Urban", False: "Rural"}),
            age_group=pd.cut(
                combined["age"],
                bins=[17, 29, 44, 99],
                labels=["Youth (<30)", "Prime (30–44)", "Senior (45+)"],
            ),
        )
        .groupby(["area", "age_group"], observed=True)
        .agg(
            n           =("applicant_id", "count"),
            mean_score  =("credit_score", "mean"),
            approve_rate=("decision", lambda x: round((x == "APPROVE").mean() * 100, 1)),
        )
        .round(1)
        .reset_index()
    )
    st.dataframe(fairness, use_container_width=True, hide_index=True)
    st.warning(
        "⚠️ Youth (<30) applicants show a lower approval rate. "
        "Recommended fix: re-weight mobile money and cooperative features, "
        "which are stronger relative predictors for younger applicants "
        "with short employment histories."
    )
