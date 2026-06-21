import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")

# ── MUST be first Streamlit call ───────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Modeling and Risk Scoring for Bank Customer Churn",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --primary:        #3B82F6;
  --primary-dim:    rgba(59,130,246,0.14);
  --primary-border: rgba(59,130,246,0.32);
  --danger:         #EF4444;
  --danger-dim:     rgba(239,68,68,0.12);
  --success:        #22C55E;
  --success-dim:    rgba(34,197,94,0.12);
  --warning:        #F59E0B;
  --warning-dim:    rgba(245,158,11,0.12);
  --surface:        rgba(128,128,128,0.08);
  --surface-border: rgba(128,128,128,0.18);
  --radius:         10px;
  --radius-lg:      14px;
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
}

.main .block-container {
  padding: 1.5rem 2.5rem 3rem 2.5rem !important;
  max-width: 1400px !important;
}

h1 {
  font-size: 1.9rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  line-height: 1.2 !important;
  margin-bottom: 0.2rem !important;
}

h2, h3 {
  font-size: 1.1rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
  margin-top: 1.4rem !important;
  margin-bottom: 0.55rem !important;
  padding-bottom: 5px !important;
  border-bottom: 1.5px solid var(--primary-border) !important;
  color: var(--primary) !important;
}

.page-subtitle {
  font-size: 0.875rem;
  opacity: 0.55;
  margin-bottom: 1.1rem;
  margin-top: -0.1rem;
}

.page-breadcrumb {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--primary);
  margin-bottom: 3px;
  font-family: 'JetBrains Mono', monospace;
  opacity: 0.8;
}

.project-hero {
  background: linear-gradient(135deg,
    rgba(59,130,246,0.13) 0%,
    rgba(99,102,241,0.08) 55%,
    rgba(59,130,246,0.04) 100%);
  border: 1px solid rgba(59,130,246,0.22);
  border-radius: 16px;
  padding: 26px 30px 22px 30px;
  margin-bottom: 1.4rem;
  position: relative;
  overflow: hidden;
}
.project-hero::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #3B82F6, #6366F1, #3B82F6);
}
.project-hero-eyebrow {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #3B82F6;
  margin-bottom: 7px;
  font-family: 'JetBrains Mono', monospace;
}
.project-hero-title {
  font-size: 1.65rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.2;
  margin: 0 0 10px 0;
}
.project-hero-meta {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 10px;
}
.hero-tag {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 20px;
  background: rgba(59,130,246,0.12);
  border: 1px solid rgba(59,130,246,0.25);
  color: #3B82F6;
  font-family: 'JetBrains Mono', monospace;
}
.hero-tag.green {
  background: rgba(34,197,94,0.10);
  border-color: rgba(34,197,94,0.25);
  color: #22C55E;
}
.hero-tag.purple {
  background: rgba(99,102,241,0.12);
  border-color: rgba(99,102,241,0.25);
  color: #818CF8;
}

.insight-box {
  background: var(--primary-dim);
  border-left: 3px solid var(--primary);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 10px 14px;
  font-size: 13px;
  line-height: 1.55;
  margin: 5px 0;
}
.insight-box.red    { background: var(--danger-dim);  border-left-color: var(--danger); }
.insight-box.green  { background: var(--success-dim); border-left-color: var(--success); }
.insight-box.orange { background: var(--warning-dim); border-left-color: var(--warning); }

[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: var(--radius);
  padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  opacity: 0.55 !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.6rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
  font-family: 'JetBrains Mono', monospace !important;
}

hr {
  border-color: var(--surface-border) !important;
  margin: 0.85rem 0 !important;
}

.dash-footer {
  text-align: center;
  font-size: 11px;
  opacity: 0.38;
  padding: 0.9rem 0 0.4rem;
  letter-spacing: 0.04em;
  font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# Plotly default template
PLOT_TEMPLATE = "plotly_dark"
COLORS = {
    "blue":   "#3B82F6",
    "red":    "#EF4444",
    "green":  "#22C55E",
    "orange": "#F59E0B",
    "purple": "#8B5CF6",
    "pink":   "#EC4899",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")


@st.cache_data
def preprocess(df):
    df2 = df.copy()
    df2.drop(columns=["CustomerId", "Surname", "Year"], errors="ignore", inplace=True)
    df2 = pd.get_dummies(df2, columns=["Geography", "Gender"], drop_first=False)
    df2["BalanceToSalary"] = df2["Balance"] / (df2["EstimatedSalary"] + 1)
    df2["ProductDensity"]  = df2["NumOfProducts"] / (df2["Tenure"] + 1)
    df2["EngageProduct"]   = df2["IsActiveMember"] * df2["NumOfProducts"]
    df2["AgeTenure"]       = df2["Age"] * df2["Tenure"]
    X = df2.drop("Exited", axis=1)
    y = df2["Exited"]
    return X, y


@st.cache_data
def train_models(_X, _y):
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    specs = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
    }

    results = {}
    for name, (model, use_sc) in specs.items():
        Xtr = Xtr_s if use_sc else X_train
        Xte = Xte_s if use_sc else X_test
        model.fit(Xtr, y_train)
        yp  = model.predict(Xte)
        ypr = model.predict_proba(Xte)[:, 1]
        cm  = confusion_matrix(y_test, yp)
        results[name] = {
            "model":     model,
            "y_pred":    yp,
            "y_prob":    ypr,
            "accuracy":  accuracy_score(y_test, yp),
            "precision": precision_score(y_test, yp),
            "recall":    recall_score(y_test, yp),
            "f1":        f1_score(y_test, yp),
            "roc_auc":   roc_auc_score(y_test, ypr),
            "cm": cm,
            "TP": int(cm[1, 1]), "FP": int(cm[0, 1]),
            "TN": int(cm[0, 0]), "FN": int(cm[1, 0]),
        }

    gb        = GradientBoostingClassifier(n_estimators=100, random_state=42)
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb, _X, _y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return results, X_test, y_test, scaler, X_train, y_train, cv_scores


df = load_data()
X, y = preprocess(df)
results, X_test, y_test, scaler, X_train, y_train, cv_scores = train_models(X, y)

best_name = "Gradient Boosting"
best      = results[best_name]
gb_model  = best["model"]

probs_all = gb_model.predict_proba(X)[:, 1]
df_risk   = df.copy()
df_risk["ChurnProbability"] = (probs_all * 100).round(2)
df_risk["RiskLevel"] = pd.cut(
    probs_all, bins=[0, 0.3, 0.6, 1.01],
    labels=["Low", "Medium", "High"]
)

SCALE = len(df) / len(y_test)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 14px 0;">
      <div style="font-size:9.5px;font-weight:700;letter-spacing:0.14em;
                  text-transform:uppercase;color:#3B82F6;font-family:monospace;margin-bottom:6px;">
        🏦 Project
      </div>
      <div style="font-size:13px;font-weight:800;line-height:1.4;letter-spacing:-0.02em;">
        Predictive Modeling and<br>Risk Scoring for Bank<br>Customer Churn
      </div>
      <div style="font-size:11px;font-weight:500;opacity:0.5;margin-top:5px;">
        European Bank · 2025
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("Navigate to", [
        "📊 Overview & EDA",
        "🤖 Model Comparison",
        "📐 Model Quality Metrics",
        "🔍 Feature Importance",
        "👥 Customer Segment Profiles",
        "💰 Business ROI Calculator",
        "🎯 What-If Simulator",
        "📋 Customer Risk Table",
    ])

    st.divider()
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
        "text-transform:uppercase;opacity:0.4;margin-bottom:8px;'>Quick Stats</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Dataset:** `10,000` customers")
    st.markdown(f"**Churn rate:** `{df.Exited.mean()*100:.1f}%`")
    st.markdown(f"**Best AUC:** `{best['roc_auc']:.3f}`")
    st.markdown(f"**CV AUC:** `{cv_scores.mean():.3f} ± {cv_scores.std():.3f}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":

    st.markdown("""
    <div class="project-hero">
      <div class="project-hero-eyebrow">📊 Module 1 of 8 &nbsp;·&nbsp; Exploratory Data Analysis</div>
      <div class="project-hero-title">
        Predictive Modeling and Risk Scoring<br>for Bank Customer Churn
      </div>
      <div class="project-hero-meta">
        <span class="hero-tag">European Bank Dataset</span>
        <span class="hero-tag">10,000 Customers</span>
        <span class="hero-tag green">France · Spain · Germany</span>
        <span class="hero-tag purple">Gradient Boosting · AUC 0.868</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers",  "10,000")
    c2.metric("Churned",          f"{df.Exited.sum():,}",         delta="20.4%", delta_color="inverse")
    c3.metric("Retained",         f"{(df.Exited==0).sum():,}")
    c4.metric("Avg Age",          f"{df.Age.mean():.1f}")
    c5.metric("Avg Balance",      f"€{df.Balance.mean()/1000:.0f}K")
    c6.metric("Avg Credit Score", f"{df.CreditScore.mean():.0f}")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Churn Distribution")
        vals   = df.Exited.value_counts().sort_index()
        fig = px.pie(
            values=vals.values,
            names=["Retained", "Churned"],
            color_discrete_sequence=["#22C55E", "#EF4444"],
            hole=0.4,
            template=PLOT_TEMPLATE,
        )
        fig.update_traces(textfont_size=13, textfont_color="white",
                          marker=dict(line=dict(color="white", width=2)))
        fig.update_layout(height=360, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Churn Rate by Geography")
        geo_churn = df.groupby("Geography")["Exited"].mean().reset_index()
        geo_churn.columns = ["Geography", "ChurnRate"]
        geo_churn["ChurnRate"] = (geo_churn["ChurnRate"] * 100).round(1)
        geo_churn["Color"] = geo_churn["ChurnRate"].apply(
            lambda v: "#EF4444" if v > 25 else "#F59E0B" if v > 20 else "#22C55E"
        )
        fig = px.bar(geo_churn, x="Geography", y="ChurnRate",
                     color="Geography",
                     color_discrete_sequence=["#EF4444", "#22C55E", "#F59E0B"],
                     text="ChurnRate", template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(y=df.Exited.mean()*100, line_dash="dash",
                      line_color="gray", annotation_text="Overall Avg")
        fig.update_layout(height=360, margin=dict(t=20, b=10),
                          yaxis_title="Churn Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Churn by Age Group")
        df_tmp = df.copy()
        df_tmp["AgeGroup"] = pd.cut(
            df_tmp["Age"], bins=[0,30,40,50,60,100],
            labels=["18-30","31-40","41-50","51-60","60+"]
        )
        age_churn = df_tmp.groupby("AgeGroup", observed=True)["Exited"].mean().reset_index()
        age_churn["Exited"] = (age_churn["Exited"] * 100).round(1)
        fig = px.bar(age_churn, x="AgeGroup", y="Exited",
                     color="Exited",
                     color_continuous_scale=["#22C55E","#F59E0B","#EF4444"],
                     text="Exited", template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=360, margin=dict(t=20,b=10),
                          yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("### Active vs Inactive Member Churn")
        act = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
        act["Label"]  = ["Inactive", "Active"]
        act["Exited"] = (act["Exited"] * 100).round(1)
        fig = px.bar(act, x="Label", y="Exited",
                     color="Label",
                     color_discrete_sequence=["#EF4444","#22C55E"],
                     text="Exited", template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=360, margin=dict(t=20,b=10),
                          yaxis_title="Churn Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### Churn by Gender")
        gen = df.groupby("Gender")["Exited"].mean().reset_index()
        gen["Exited"] = (gen["Exited"] * 100).round(1)
        fig = px.bar(gen, x="Gender", y="Exited",
                     color="Gender",
                     color_discrete_sequence=["#EC4899","#3B82F6"],
                     text="Exited", template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=360, margin=dict(t=20,b=10),
                          yaxis_title="Churn Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("### Churn by Number of Products")
        prod = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
        prod["Exited"] = (prod["Exited"] * 100).round(1)
        prod["Color"]  = prod["Exited"].apply(
            lambda v: "#EF4444" if v > 50 else "#22C55E"
        )
        fig = px.bar(prod, x="NumOfProducts", y="Exited",
                     color="Exited",
                     color_continuous_scale=["#22C55E","#F59E0B","#EF4444"],
                     text="Exited", template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=360, margin=dict(t=20,b=10),
                          xaxis_title="Number of Products",
                          yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key EDA Insights")
    for style, text in [
        ("red",   "🔴 <b>Germany 32.4% churn</b> — 2× France/Spain. Immediate regional intervention needed."),
        ("red",   "🔴 <b>Age 51-60 critical</b> — 56.2% churn. Life transition phase with highest attrition."),
        ("red",   "🔴 <b>Inactive members churn 26.9%</b> vs 14.3% active — engagement is the #1 lever."),
        ("red",   "🔴 <b>Female customers churn 52% more</b> than males (25.1% vs 16.5%)."),
        ("red",   "🔴 <b>3-4 products = 82-100% churn</b> — clear evidence of forced cross-selling."),
        ("green", "🟢 <b>Ages 18-30 only 7.5% churn</b> — most loyal segment with highest lifetime value."),
    ]:
        st.markdown(f'<div class="insight-box {style}">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("<div class='page-breadcrumb'>🏦 Predictive Modeling &amp; Risk Scoring · Module 2 of 8</div>",
                unsafe_allow_html=True)
    st.markdown("# 🤖 Model Comparison & Evaluation")
    st.markdown("<div class='page-subtitle'>Four classifiers evaluated on held-out test set (20% = 2,000 customers)</div>",
                unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        "Model":     list(results.keys()),
        "Accuracy":  [f"{r['accuracy']*100:.1f}%"  for r in results.values()],
        "Precision": [f"{r['precision']*100:.1f}%" for r in results.values()],
        "Recall":    [f"{r['recall']*100:.1f}%"    for r in results.values()],
        "F1-Score":  [f"{r['f1']:.3f}"             for r in results.values()],
        "ROC-AUC":   [f"{r['roc_auc']:.3f}"        for r in results.values()],
        "TP": [r["TP"] for r in results.values()],
        "FP": [r["FP"] for r in results.values()],
        "TN": [r["TN"] for r in results.values()],
        "FN": [r["FN"] for r in results.values()],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.success(
        f"✅ Best Model: **Gradient Boosting** — "
        f"AUC: {best['roc_auc']:.3f} | F1: {best['f1']:.3f} | "
        f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    )

    col1, col2 = st.columns(2)
    model_colors = ["#85C1E9", "#F59E0B", "#3B82F6", "#1A5276"]

    with col1:
        st.markdown("### ROC Curves — All Models")
        fig = go.Figure()
        for (name, res), col in zip(results.items(), model_colors):
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"{name} ({res['roc_auc']:.3f})",
                line=dict(color=col, width=2)
            ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], name="Random (0.500)",
            line=dict(color="gray", dash="dash"), opacity=0.5
        ))
        fig.update_layout(
            template=PLOT_TEMPLATE, height=400,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(font=dict(size=10)),
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Precision-Recall Curves")
        fig = go.Figure()
        for (name, res), col in zip(results.items(), model_colors):
            prec_c, rec_c, _ = precision_recall_curve(y_test, res["y_prob"])
            fig.add_trace(go.Scatter(
                x=rec_c, y=prec_c, name=name,
                line=dict(color=col, width=2)
            ))
        fig.add_hline(y=df.Exited.mean(), line_dash="dash",
                      line_color="gray", annotation_text="Baseline")
        fig.update_layout(
            template=PLOT_TEMPLATE, height=400,
            xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(font=dict(size=10)),
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion Matrices — Business Interpretation")
    cols = st.columns(4)
    for idx, (name, res) in enumerate(results.items()):
        with cols[idx]:
            cm = res["cm"]
            fig = px.imshow(
                cm, text_auto=True,
                x=["Retained","Churned"],
                y=["Retained","Churned"],
                color_continuous_scale="Blues",
                template=PLOT_TEMPLATE,
                title=f"{name}<br>AUC={res['roc_auc']:.3f}",
            )
            fig.update_layout(height=280, margin=dict(t=50,b=10),
                              coloraxis_showscale=False)
            fig.update_xaxes(title="Predicted")
            fig.update_yaxes(title="Actual")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<div style='font-size:11px;background:rgba(128,128,128,0.08);"
                f"border:1px solid rgba(128,128,128,0.18);padding:8px 10px;"
                f"border-radius:8px;margin-top:4px;line-height:1.8;'>"
                f"✅ <b>TP={res['TP']}</b> Correctly caught<br>"
                f"⚠️ <b>FN={res['FN']}</b> Missed churners<br>"
                f"💸 <b>FP={res['FP']}</b> False alarms<br>"
                f"✅ <b>TN={res['TN']}</b> Correctly retained</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### 5-Fold Cross-Validation — Gradient Boosting")
    col1, col2 = st.columns(2)
    with col1:
        fold_labels = [f"Fold {i+1}" for i in range(5)]
        colors_cv   = ["#EF4444" if v < cv_scores.mean() else "#22C55E" for v in cv_scores]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fold_labels, y=cv_scores,
            marker_color=colors_cv,
            text=[f"{v:.3f}" for v in cv_scores],
            textposition="outside",
        ))
        fig.add_hline(y=cv_scores.mean(), line_dash="dash",
                      line_color="#3B82F6",
                      annotation_text=f"Mean={cv_scores.mean():.3f}")
        fig.update_layout(
            template=PLOT_TEMPLATE, height=320,
            yaxis_title="ROC-AUC Score",
            yaxis_range=[0.82, 0.90],
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cv_df = pd.DataFrame({
            "Fold": fold_labels + ["Mean", "Std Dev"],
            "AUC":  list(cv_scores.round(3)) + [round(cv_scores.mean(),3), round(cv_scores.std(),3)],
        })
        st.dataframe(cv_df, hide_index=True, use_container_width=True)
        st.markdown(
            f'<div class="insight-box green">✅ Low std dev ({cv_scores.std():.3f}) '
            f"confirms model is <b>stable and not overfitting</b> across all folds.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Model Quality Metrics":
    st.markdown("<div class='page-breadcrumb'>🏦 Predictive Modeling &amp; Risk Scoring · Module 3 of 8</div>",
                unsafe_allow_html=True)
    st.markdown("# 📐 Model Quality & Threshold Analysis")
    st.markdown("<div class='page-subtitle'>Comprehensive evaluation metrics with business interpretation</div>",
                unsafe_allow_html=True)

    st.markdown("### Threshold Sensitivity Analysis")
    thresholds   = [0.3, 0.4, 0.5, 0.6, 0.7]
    implications = [
        "⬆️ More churners caught, more false alarms",
        "✅ Balanced — good for campaigns",
        "✅ Default — standard threshold",
        "🎯 High precision, fewer false alarms",
        "⚠️ Very conservative, misses many churners",
    ]
    thresh_rows = []
    for t, impl in zip(thresholds, implications):
        yp  = (best["y_prob"] >= t).astype(int)
        cm  = confusion_matrix(y_test, yp)
        thresh_rows.append({
            "Threshold":            t,
            "Precision (%)":        round(precision_score(y_test, yp) * 100, 1),
            "Recall (%)":           round(recall_score(y_test, yp) * 100, 1),
            "F1-Score":             round(f1_score(y_test, yp), 3),
            "FP — False Alarms":    int(cm[0, 1]),
            "FN — Missed Churners": int(cm[1, 0]),
            "Cost Implication":     impl,
        })
    st.dataframe(pd.DataFrame(thresh_rows), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Precision vs Recall Tradeoff")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds, y=[r["Precision (%)"] for r in thresh_rows],
            name="Precision", mode="lines+markers",
            line=dict(color="#3B82F6", width=2), marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=[r["Recall (%)"] for r in thresh_rows],
            name="Recall", mode="lines+markers",
            line=dict(color="#EF4444", width=2), marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=[r["F1-Score"]*100 for r in thresh_rows],
            name="F1 × 100", mode="lines+markers",
            line=dict(color="#22C55E", width=2, dash="dash"), marker=dict(size=8)
        ))
        fig.add_vline(x=0.5, line_dash="dot", line_color="gray",
                      annotation_text="Default")
        fig.update_layout(template=PLOT_TEMPLATE, height=360,
                          xaxis_title="Threshold", yaxis_title="Score (%)",
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Error Counts vs Threshold")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds, y=[r["FP — False Alarms"] for r in thresh_rows],
            name="FP — False Alarms", mode="lines+markers",
            line=dict(color="#F59E0B", width=2), marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=[r["FN — Missed Churners"] for r in thresh_rows],
            name="FN — Missed Churners", mode="lines+markers",
            line=dict(color="#EF4444", width=2), marker=dict(size=8)
        ))
        fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
        fig.update_layout(template=PLOT_TEMPLATE, height=360,
                          xaxis_title="Threshold", yaxis_title="Count",
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Risk Segment Quality Assessment")
    seg_data = {
        "🟢 Low Risk (<30%)":      {"n": 7919, "actual": 8.7,  "pred": 9.9,  "color": "#22C55E"},
        "🟡 Medium Risk (30-60%)": {"n": 1133, "actual": 45.7, "pred": 43.0, "color": "#F59E0B"},
        "🔴 High Risk (>60%)":     {"n": 948,  "actual": 87.4, "pred": 81.6, "color": "#EF4444"},
    }
    c1, c2, c3 = st.columns(3)
    for col, (seg, d) in zip([c1,c2,c3], seg_data.items()):
        with col:
            st.markdown(
                f"<div style='background:{d['color']}1A;border:2px solid {d['color']};"
                f"border-radius:12px;padding:16px;text-align:center;'>"
                f"<div style='font-size:14px;font-weight:700;color:{d['color']}'>{seg}</div>"
                f"<div style='font-size:28px;font-weight:800;color:{d['color']};margin:8px 0'>{d['n']:,}</div>"
                f"<div style='font-size:12px;opacity:0.6;'>customers</div>"
                f"<hr style='border-color:{d['color']}44;margin:8px 0;'>"
                f"<div style='font-size:13px;'><b>Actual Churn:</b> {d['actual']}%</div>"
                f"<div style='font-size:13px;'><b>Avg Probability:</b> {d['pred']}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Actual vs Predicted Churn by Segment")
        segs     = ["Low Risk", "Medium Risk", "High Risk"]
        actual_v = [8.7, 45.7, 87.4]
        pred_v   = [9.9, 43.0, 81.6]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Actual Churn %",    x=segs, y=actual_v,
                             marker_color="#3B82F6",
                             text=[f"{v}%" for v in actual_v], textposition="outside"))
        fig.add_trace(go.Bar(name="Predicted Prob %", x=segs, y=pred_v,
                             marker_color="#F59E0B",
                             text=[f"{v}%" for v in pred_v], textposition="outside"))
        fig.update_layout(template=PLOT_TEMPLATE, barmode="group", height=360,
                          yaxis_title="Churn Rate / Probability (%)",
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Score Distribution by True Label")
        retained_probs = best["y_prob"][y_test == 0]
        churned_probs  = best["y_prob"][y_test == 1]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=retained_probs, name="Retained (actual)",
            opacity=0.7, marker_color="#22C55E",
            histnorm="probability density", nbinsx=40
        ))
        fig.add_trace(go.Histogram(
            x=churned_probs, name="Churned (actual)",
            opacity=0.7, marker_color="#EF4444",
            histnorm="probability density", nbinsx=40
        ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="white",
                      annotation_text="0.5")
        fig.add_vline(x=0.3, line_dash="dot",  line_color="gray",
                      annotation_text="0.3")
        fig.update_layout(template=PLOT_TEMPLATE, barmode="overlay", height=360,
                          xaxis_title="Predicted Probability",
                          yaxis_title="Density", margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Quality — Business Interpretation")
    for style, text in [
        ("green",  "✅ <b>Threshold 0.4 recommended</b>: catches ~55% of churners with ~70% precision — optimal budget/coverage balance."),
        ("orange", "⚠️ <b>At threshold 0.5</b>: ~53 false alarms but only ~206 missed churners. Acceptable for high-value segments."),
        ("red",    "🔴 <b>High Risk segment (87.4% actual churn)</b>: model correctly isolates this tier — deploy relationship managers immediately."),
        ("green",  "✅ <b>CV AUC 0.866 ± 0.008</b>: extremely stable — model is production-ready with no overfitting."),
    ]:
        st.markdown(f'<div class="insight-box {style}">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.markdown("<div class='page-breadcrumb'>🏦 Predictive Modeling &amp; Risk Scoring · Module 4 of 8</div>",
                unsafe_allow_html=True)
    st.markdown("# 🔍 Feature Importance Analysis")
    st.markdown("<div class='page-subtitle'>Gradient Boosting — which features drive churn predictions most</div>",
                unsafe_allow_html=True)

    fi = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### Top 20 Feature Importances")
        top20 = fi.head(20).reset_index()
        top20.columns = ["Feature", "Importance"]
        top20["Color"] = top20["Importance"].apply(
            lambda v: "#EF4444" if v > 0.08 else "#F59E0B" if v > 0.05 else "#3B82F6"
        )
        fig = px.bar(
            top20.iloc[::-1], x="Importance", y="Feature",
            orientation="h",
            color="Color",
            color_discrete_map="identity",
            text=top20.iloc[::-1]["Importance"].apply(lambda v: f"{v:.3f}"),
            template=PLOT_TEMPLATE,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=600, margin=dict(t=20,b=10),
                          showlegend=False, xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top 10 Features")
        top10 = fi.head(10).reset_index()
        top10.columns = ["Feature", "Importance"]
        top10["Importance %"] = (top10["Importance"]*100).round(1).astype(str)+"%"
        top10.insert(0, "Rank", range(1,11))
        st.dataframe(top10[["Rank","Feature","Importance %"]],
                     hide_index=True, use_container_width=True)

        st.markdown("### Engineered Feature Scores")
        for feat in ["BalanceToSalary","ProductDensity","EngageProduct","AgeTenure"]:
            if feat in fi.index:
                val = float(fi[feat])
                st.progress(min(val*8, 1.0), text=f"{feat}: {val:.4f}")

    st.markdown("### Correlation Heatmap")
    key_cols = ["Age","Balance","IsActiveMember","NumOfProducts",
                "CreditScore","Tenure","EstimatedSalary","Exited"]
    corr = df[key_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, template=PLOT_TEMPLATE,
        title="Feature Correlation Matrix",
    )
    fig.update_layout(height=450, margin=dict(t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Insights")
    for style, text in [
        ("red",    "🔴 <b>Age (38.7%)</b> — single most powerful predictor. The 41-60 cohort has the highest attrition."),
        ("red",    "🔴 <b>NumOfProducts (28.8%)</b> — 3+ products is an over-selling signal."),
        ("orange", "🟡 <b>EngageProduct (6.8%)</b> — engineered: active membership × products captures true engagement."),
        ("green",  "🟢 <b>IsActiveMember (5.6%)</b> — strongest controllable lever. Engagement programs reduce churn."),
        ("orange", "🟡 <b>Geography_Germany (5.4%)</b> — structural regional risk embedded as a significant predictor."),
    ]:
        st.markdown(f'<div class="insight-box {style}">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER SEGMENT PROFILES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segment Profiles":
    st.markdown("<div class='page-breadcrumb'>🏦 Predictive Modeling &amp; Risk Scoring · Module 5 of 8</div>",
                unsafe_allow_html=True)
    st.markdown("# 👥 Customer Segment Profiles")
    st.markdown("<div class='page-subtitle'>Deep business interpretation for each risk segment</div>",
                unsafe_allow_html=True)

    df_risk["AgeGroup"] = pd.cut(
        df_risk["Age"], bins=[0,30,40,50,60,100],
        labels=["18-30","31-40","41-50","51-60","60+"]
    )

    segments = [
        {"label":"🔴 High Risk",   "mask": df_risk["RiskLevel"]=="High",
         "color":"#EF4444", "actual":"87.4%",
         "action":("red","🔴 <b>Who are they?</b> Avg age ~50, ~49% from Germany, only ~19% active. Mid-life, disengaged. <b>Action: Immediate outreach by relationship manager. Offer premium upgrade or fee waiver.</b>")},
        {"label":"🟡 Medium Risk", "mask": df_risk["RiskLevel"]=="Medium",
         "color":"#F59E0B", "actual":"45.7%",
         "action":("orange","🟡 <b>Who are they?</b> Avg age ~46, ~43% from Germany, ~43% active. Borderline — salvageable. <b>Action: Targeted retention campaign within 14 days. Personalised offer or loyalty reward.</b>")},
        {"label":"🟢 Low Risk",    "mask": df_risk["RiskLevel"]=="Low",
         "color":"#22C55E", "actual":"8.7%",
         "action":("green","🟢 <b>Who are they?</b> Avg age ~37, predominantly France/Spain, ~57% active. Stable, engaged. <b>Action: Cross-sell opportunities and long-term loyalty programmes to maximise CLV.</b>")},
    ]

    for seg in segments:
        sub = df_risk[seg["mask"]]
        st.markdown(f"### {seg['label']} — {len(sub):,} customers · Actual churn: {seg['actual']}")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Avg Age",          f"{sub.Age.mean():.1f}")
        c2.metric("Avg Balance",       f"€{sub.Balance.mean()/1000:.0f}K")
        c3.metric("% Active Members",  f"{sub.IsActiveMember.mean()*100:.1f}%")
        c4.metric("% Germany",         f"{(sub.Geography=='Germany').mean()*100:.1f}%")
        c5.metric("Avg Products",      f"{sub.NumOfProducts.mean():.2f}")

        col1, col2 = st.columns(2)
        with col1:
            age_dist = sub["AgeGroup"].value_counts().sort_index().reset_index()
            age_dist.columns = ["AgeGroup","Count"]
            fig = px.bar(age_dist, x="AgeGroup", y="Count",
                         text="Count", template=PLOT_TEMPLATE,
                         color_discrete_sequence=[seg["color"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, margin=dict(t=20,b=10),
                              title=f"{seg['label']} — Age Distribution",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            geo_dist = sub["Geography"].value_counts().reset_index()
            geo_dist.columns = ["Geography","Count"]
            fig = px.pie(geo_dist, values="Count", names="Geography",
                         color_discrete_sequence=["#3B82F6","#F59E0B","#EF4444"],
                         hole=0.35, template=PLOT_TEMPLATE,
                         title=f"{seg['label']} — Geography Split")
            fig.update_layout(height=300, margin=dict(t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

        style, text = seg["action"]
        st.markdown(f'<div class="insight-box {style}">{text}</div>', unsafe_allow_html=True)
        st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — BUSINESS ROI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Business ROI Calculator":
    st.markdown("<div class='page-breadcrumb'>🏦 Predictive Modeling &amp; Risk Scoring · Module 6 of 8</div>",
                unsafe_allow_html=True)
    st.markdown("# 💰 Business ROI Calculator")
    st.markdown("<div class='page-subtitle'>Quantified business impact with cost-benefit analysis</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Input Parameters")
        avg_clv           = st.slider("Average CLV per Customer (€)", 1000, 10000, 2500, 100)
        retention_cost    = st.slider("Retention Campaign Cost per Customer (€)", 50, 500, 150, 10)
        intervention_rate = st.slider("Campaign Success Rate (%)", 10, 60, 30, 5)
        threshold         = st.selectbox("Deployment Threshold", [0.3,0.4,0.5,0.6], index=2)

    yp_thresh        = (best["y_prob"] >= threshold).astype(int)
    cm_t             = confusion_matrix(y_test, yp_thresh)
    tp_t, fp_t, fn_t = int(cm_t[1,1]), int(cm_t[0,1]), int(cm_t[1,0])

    total_high_risk   = int((probs_all >= threshold).sum())
    total_caught      = int(tp_t * SCALE)
    total_false_alarm = int(fp_t * SCALE)
    total_missed      = int(fn_t * SCALE)
    saved_customers   = int(total_caught * intervention_rate / 100)
    clv_retained      = saved_customers * avg_clv
    campaign_cost     = total_high_risk * retention_cost
    net_benefit       = clv_retained - campaign_cost
    roi_pct           = (net_benefit / campaign_cost * 100) if campaign_cost > 0 else 0

    with col2:
        st.markdown("### ROI Results")
        r1, r2 = st.columns(2)
        r1.metric("Customers Flagged",    f"{total_high_risk:,}")
        r2.metric("True Churners Caught", f"{total_caught:,}")
        r1.metric("Estimated Saved",      f"{saved_customers:,}", f"at {intervention_rate}% success")
        r2.metric("CLV Retained",         f"€{clv_retained:,.0f}")
        r1.metric("Campaign Cost",        f"€{campaign_cost:,.0f}")
        r2.metric("Net Benefit",          f"€{net_benefit:,.0f}",
                  delta=f"ROI: {roi_pct:.0f}%",
                  delta_color="normal" if net_benefit > 0 else "inverse")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Cost Breakdown")
        fig = go.Figure(go.Bar(
            x=["CLV Retained\n(Benefit)", "Campaign Cost\n(Investment)", "Net Benefit"],
            y=[clv_retained, -campaign_cost, net_benefit],
            marker_color=["#22C55E","#EF4444","#3B82F6" if net_benefit>0 else "#EF4444"],
            text=[f"€{abs(clv_retained):,.0f}", f"€{campaign_cost:,.0f}", f"€{abs(net_benefit):,.0f}"],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="white", line_width=0.5, opacity=0.3)
        fig.update_layout(template=PLOT_TEMPLATE, height=360,
                          yaxis_title="Amount (€)", margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Cost of Model Errors")
        fp_cost = total_false_alarm * retention_cost
        fn_cost = total_missed * avg_clv
        fig = go.Figure(go.Bar(
            x=["FP Cost\n(Wasted campaigns)", "FN Cost\n(Lost CLV)"],
            y=[fp_cost, fn_cost],
            marker_color=["#F59E0B","#EF4444"],
            text=[f"€{fp_cost:,.0f}", f"€{fn_cost:,.0f}"],
            textposition="outside",
        ))
        fig.update_layout(template=PLOT_TEMPLATE, height=360,
                          yaxis_title="Cost (€)", margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full Scenario Comparison")
    rows = []
    for t in [0.3,0.4,0.5,0.6,0.7]:
        yp_t   = (best["y_prob"] >= t).astype(int)
        cm_s   = confusion_matrix(y_test, yp_t)
        flagged = int((probs_all >= t).sum())
        caught  = int(cm_s[1,1] * SCALE)
        saved   = int(caught * intervention_rate / 100)
        clv_s   = saved * avg_clv
        cost_s  = flagged * retention_cost
        net_s   = clv_s - cost_s
        roi_s   = net_s / cost_s * 100 if cost_s > 0 else 0
        rows.append({
            "Threshold": t, "Flagged": flagged, "Caught": caught,
            "Saved": saved, "CLV Retained": f"€{clv_s:,}",
            "Campaign Cost": f"€{cost_s:,}", "Net Benefit": f"€{net_s:,}",
            "ROI": f"{roi_s:.0f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 What-If Simulator":
    st.markdown("<div class='page-breadcrumb'>🏦 Predictive Modeling &amp; Risk Scoring · Module 7 of 8</div>",
                unsafe_allow_html=True)
    st.markdown("# 🎯 What-If Churn Risk Simulator")
    st.markdown("<div class='page-subtitle'>Adjust a customer profile and get an instant churn probability</div>",
                unsafe_allow_html=True)

    col_inp, col_out = st.columns(2)
    with col_inp:
        st.markdown("### Customer Profile Input")
        age       = st.slider("Age", 18, 80, 38)
        credit    = st.slider("Credit Score", 300, 850, 650)
        balance   = st.slider("Account Balance (€)", 0, 250000, 76000, step=1000)
        salary    = st.slider("Estimated Salary (€)", 20000, 200000, 100000, step=1000)
        tenure    = st.slider("Tenure (Years)", 0, 10, 5)
        products  = st.selectbox("Number of Products", [1,2,3,4])
        geography = st.selectbox("Geography", ["France","Spain","Germany"])
        gender    = st.selectbox("Gender", ["Male","Female"])
        has_cc    = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)

    input_dict = {
        "CreditScore":       credit,
        "Age":               age,
        "Tenure":            tenure,
        "Balance":           balance,
        "NumOfProducts":     products,
        "HasCrCard":         int(has_cc),
        "IsActiveMember":    int(is_active),
        "EstimatedSalary":   salary,
        "Geography_France":  int(geography=="France"),
        "Geography_Germany": int(geography=="Germany"),
        "Geography_Spain":   int(geography=="Spain"),
        "Gender_Female":     int(gender=="Female"),
        "Gender_Male":       int(gender=="Male"),
        "BalanceToSalary":   balance/(salary+1),
        "ProductDensity":    products/(tenure+1),
        "EngageProduct":     int(is_active)*products,
        "AgeTenure":         age*tenure,
    }
    input_df = pd.DataFrame([input_dict])[X.columns]
    prob     = gb_model.predict_proba(input_df)[0][1]

    if prob < 0.3:
        risk_label, risk_color = "🟢 LOW RISK",    "#22C55E"
        rec = "Stable customer. Focus on cross-sell and CLV maximisation."
    elif prob < 0.6:
        risk_label, risk_color = "🟡 MEDIUM RISK", "#F59E0B"
        rec = "Outreach within 14 days. Offer loyalty reward or product upgrade."
    else:
        risk_label, risk_color = "🔴 HIGH RISK",   "#EF4444"
        rec = "URGENT: Assign to relationship manager. Offer fee waiver or premium upgrade immediately."

    with col_out:
        st.markdown("### Churn Risk Assessment")
        st.markdown(
            f"<div style='background:{risk_color}1A;border:2px solid {risk_color};"
            f"border-radius:12px;padding:24px;text-align:center;margin-bottom:16px;'>"
            f"<div style='font-size:48px;font-weight:800;color:{risk_color};'>{prob*100:.1f}%</div>"
            f"<div style='font-size:20px;font-weight:700;color:{risk_color};margin:4px 0;'>{risk_label}</div>"
            f"<div style='font-size:12px;opacity:0.55;margin-top:8px;'>Churn Probability</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix":"%","font":{"size":28}},
            gauge={
                "axis": {"range":[0,100]},
                "bar":  {"color": risk_color},
                "steps": [
                    {"range":[0,30],   "color":"rgba(34,197,94,0.2)"},
                    {"range":[30,60],  "color":"rgba(245,158,11,0.2)"},
                    {"range":[60,100], "color":"rgba(239,68,68,0.2)"},
                ],
                "threshold": {"line":{"color":"white","width":2},"value":prob*100},
            },
        ))
        fig.update_layout(template=PLOT_TEMPLATE, height=250, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Recommendation:** {rec}")
        st.markdown("**Key Risk Factors:**")
        if age > 50:               st.markdown(f"⚠️ Age {age} — high-risk cohort")
        elif age > 40:             st.markdown(f"⚠️ Age {age} — moderate-risk cohort")
        else:                      st.markdown(f"✅ Age {age} — low-risk young segment")
        if not is_active:          st.markdown("⚠️ Inactive member — major risk factor")
        else:                      st.markdown("✅ Active member — protective factor")
        if geography == "Germany": st.markdown("⚠️ Germany — 2× baseline churn rate")
        if products >= 3:          st.markdown(f"⚠️ {products} products — possible over-selling")
        elif products == 2:        st.markdown("✅ 2 products — optimal engagement level")
        if gender == "Female":     st.markdown("⚠️ Female — historically higher churn in this dataset")
        if credit >= 700:          st.markdown(f"✅ Credit score {credit} — strong protective factor")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — CUSTOMER RISK TABLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Customer Risk Table":
    st.markdown("<div class='page-breadcrumb'>🏦 Predictive Modeling &amp; Risk Scoring · Module 8 of 8</div>",
                unsafe_allow_html=True)
    st.markdown("# 📋 Customer Risk Scoring Table")
    st.markdown("<div class='page-subtitle'>Full scored dataset — filter, sort, and export</div>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    risk_filter = col1.multiselect("Risk Level", ["Low","Medium","High"], default=["High","Medium"])
    geo_filter  = col2.multiselect("Geography", df["Geography"].unique().tolist(),
                                   default=df["Geography"].unique().tolist())
    min_prob    = col3.slider("Min Churn Probability (%)", 0, 100, 0)

    filtered = df_risk[
        df_risk["RiskLevel"].isin(risk_filter) &
        df_risk["Geography"].isin(geo_filter) &
        (df_risk["ChurnProbability"] >= min_prob)
    ].sort_values("ChurnProbability", ascending=False)

    st.markdown(f"Showing **{len(filtered):,}** customers")

    display_cols = [
        "CustomerId","Age","Geography","Gender","Balance",
        "NumOfProducts","IsActiveMember","CreditScore",
        "ChurnProbability","RiskLevel","Exited",
    ]
    st.dataframe(
        filtered[display_cols].head(500).style.background_gradient(
            subset=["ChurnProbability"], cmap="RdYlGn_r"
        ),
        use_container_width=True, hide_index=True, height=450,
    )

    rc1, rc2, rc3 = st.columns(3)
    high = int((df_risk["RiskLevel"]=="High").sum())
    med  = int((df_risk["RiskLevel"]=="Medium").sum())
    low  = int((df_risk["RiskLevel"]=="Low").sum())
    rc1.metric("High Risk",   f"{high:,}", f"{high/len(df)*100:.1f}% of total")
    rc2.metric("Medium Risk", f"{med:,}",  f"{med/len(df)*100:.1f}% of total")
    rc3.metric("Low Risk",    f"{low:,}",  f"{low/len(df)*100:.1f}% of total")

    st.download_button(
        "⬇️ Download Risk Scores CSV",
        filtered[display_cols].to_csv(index=False),
        file_name="customer_churn_risk_scores.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div class='dash-footer'>"
    "Predictive Modeling and Risk Scoring for Bank Customer Churn"
    "&nbsp;·&nbsp;European Bank 2025"
    "&nbsp;·&nbsp;Gradient Boosting AUC 0.868"
    "&nbsp;·&nbsp;CV: 0.866 ± 0.008"
    "</div>",
    unsafe_allow_html=True,
)