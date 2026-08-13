"""
Employee Advocacy Dashboard — Glassdoor Employee Sentiment & Culture
People Analytics portfolio | Zikra Aditia

Run:  streamlit run dashboard.py
Deploy: push to GitHub -> share.streamlit.io

Two views (radio in sidebar):
  1. Executive View   — CEO/CHRO: situation in 5 seconds
  2. Risk & Voice View — operational: risk calculator, segments, voice-of-employee

Honest to the data: Glassdoor is anonymous and review-level, so there is no
individual "employee action table" (that needs identified data — see IBM project).
This dashboard works at the aggregate + model level instead.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# CONFIG + BRANDING
# ----------------------------------------------------------------------
st.set_page_config(page_title="Employee Advocacy Dashboard", page_icon="📊", layout="wide")

NAVY, ORANGE, RED, GREEN = "#1E2761", "#E67E22", "#E74C3C", "#27AE60"
ICE, LGRAY, MGRAY, DARK = "#CADCFC", "#F4F6F9", "#7F8C8D", "#1A1A2E"

st.markdown(f"""
<style>
    .main {{ background-color:#FFFFFF; }}
    h1,h2,h3,h4 {{ font-family:Georgia,serif; color:{NAVY}; }}
    .stMarkdown, p, label {{ font-family:Calibri,'Segoe UI',sans-serif; }}
    .dashboard-header {{
        background:{NAVY}; color:white; padding:18px 24px; border-radius:10px;
        margin-bottom:18px;
    }}
    .dashboard-header h2 {{ color:white; margin:0; font-family:Georgia,serif; }}
    .dashboard-header p {{ color:{ICE}; margin:4px 0 0 0; font-size:14px; }}
    .kpi-card {{
        background:{LGRAY}; border-radius:10px; padding:16px 14px; text-align:center;
    }}
    .kpi-number {{ font-family:Georgia,serif; font-size:34px; font-weight:bold; line-height:1.1; }}
    .kpi-label {{ font-size:12.5px; color:{MGRAY}; margin-top:6px; }}
    [data-testid="stSidebar"] {{ background:{LGRAY}; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# DATA — small aggregate frame reconstructed from notebook results.
# Honest: these are the segment-level numbers we actually have.
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    # Segment-level advocacy (from notebook: recommend rates by status)
    seg = pd.DataFrame({
        "Status": ["Current", "Former"],
        "Reviews": [327711, 221440],
        "RecommendRate": [76.8, 62.5],
        "HighRiskPct": [23.2, 39.6],
        "MediumRiskPct": [13.8, 15.5],
        "LowRiskPct": [63.0, 44.9],
    })
    # Driver odds ratios (Section 9)
    drivers = pd.DataFrame({
        "Driver": ["Culture & Values", "Senior Management", "Career Opportunities",
                   "Work-Life Balance", "Compensation"],
        "OddsRatio": [2.46, 2.27, 2.19, 1.54, 1.48],
        "Gap": [77, 68, 68, 50, 56],
    })
    # NLP contrast (Module F)
    voice = pd.DataFrame({
        "Theme": ["Management", "Work conditions"],
        "Detractors": [14158, 18893],
        "Promoters": [5663, 49004],
    })
    return seg, drivers, voice

seg, drivers, voice = load_data()

# Model coefficients (Section 9) for the calculator
COEF = {"culture_values":0.897,"senior_mgmt":0.825,"career_opp":0.777,
        "work_life_balance":0.393,"comp_benefits":0.447,"is_current":0.107,"both_low":-0.075}
INTERCEPT = 0.90
MEAN = {"culture_values":3.2,"senior_mgmt":3.0,"career_opp":3.1,"work_life_balance":3.2,"comp_benefits":3.1}
STD  = {"culture_values":1.15,"senior_mgmt":1.20,"career_opp":1.15,"work_life_balance":1.10,"comp_benefits":1.10}

def predict_risk(c, se, ca, w, co, ic):
    bl = 1 if (c<=2 and se<=2) else 0
    z = INTERCEPT
    z += COEF["culture_values"]*(c-MEAN["culture_values"])/STD["culture_values"]
    z += COEF["senior_mgmt"]*(se-MEAN["senior_mgmt"])/STD["senior_mgmt"]
    z += COEF["career_opp"]*(ca-MEAN["career_opp"])/STD["career_opp"]
    z += COEF["work_life_balance"]*(w-MEAN["work_life_balance"])/STD["work_life_balance"]
    z += COEF["comp_benefits"]*(co-MEAN["comp_benefits"])/STD["comp_benefits"]
    z += COEF["is_current"]*ic + COEF["both_low"]*bl
    return 1 - 1/(1+np.exp(-z))

def band(r):
    if r>=0.60: return "High", RED
    if r>=0.30: return "Medium", ORANGE
    return "Low", GREEN

def kpi(col, number, label, color=NAVY):
    col.markdown(f"<div class='kpi-card'><div class='kpi-number' style='color:{color};'>"
                 f"{number}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SIDEBAR — view switch + filters (max 3, Big4 rule)
# ----------------------------------------------------------------------
st.sidebar.markdown(f"<h3 style='color:{NAVY};'>Employee Advocacy</h3>", unsafe_allow_html=True)
view = st.sidebar.radio("View", ["Executive View", "Risk & Voice View"])
st.sidebar.divider()
st.sidebar.markdown("**Filters**")
status_filter = st.sidebar.multiselect("Employment status", ["Current", "Former"],
                                        default=["Current", "Former"])
st.sidebar.caption("Filters affect the Executive View segment figures. "
                   "Glassdoor is anonymous — no per-employee filtering is possible.")

# Apply filter honestly
segf = seg[seg["Status"].isin(status_filter)] if status_filter else seg
if len(segf) == 0:
    segf = seg

# Weighted aggregate helper
def wavg(frame, col):
    return np.average(frame[col], weights=frame["Reviews"])

# ======================================================================
# EXECUTIVE VIEW
# ======================================================================
if view == "Executive View":
    st.markdown("<div class='dashboard-header'><h2>Employee Advocacy — Executive View</h2>"
                "<p>People Analytics · 838K UK Glassdoor reviews (2008–2021)</p></div>",
                unsafe_allow_html=True)

    total_rev = int(segf["Reviews"].sum())
    not_rec = round(100 - wavg(segf, "RecommendRate"), 1)
    high_risk = round(wavg(segf, "HighRiskPct"), 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, f"{not_rec}%", "Would not recommend", RED)
    kpi(c2, f"{total_rev/1000:.0f}K", "Reviews in view", NAVY)
    kpi(c3, f"{high_risk:.0f}%", "High-risk share", ORANGE)
    kpi(c4, "£1.15–2.69M", "Cost exposure /1,000", NAVY)
    kpi(c5, "2.46×", "Top driver (Culture)", GREEN)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    # Chart 1 — driver ranking
    with left:
        st.markdown("##### What drives advocacy (odds ratio, controlled)")
        colors = [ORANGE if d=="Culture & Values" else NAVY for d in drivers["Driver"]]
        f1 = go.Figure(go.Bar(x=drivers["OddsRatio"], y=drivers["Driver"], orientation="h",
                              marker_color=colors, text=[f"{v:.2f}×" for v in drivers["OddsRatio"]],
                              textposition="outside", textfont=dict(family="Calibri", color=NAVY)))
        f1.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                         showlegend=False, margin=dict(t=10,b=10,l=10,r=30),
                         xaxis=dict(range=[0,3], showgrid=True, gridcolor="#EEE"),
                         yaxis=dict(categoryorder="total ascending"), font=dict(family="Calibri"))
        st.plotly_chart(f1, use_container_width=True)

    # Chart 2 — risk distribution donut
    with right:
        st.markdown("##### Risk distribution")
        hi = wavg(segf,"HighRiskPct"); me = wavg(segf,"MediumRiskPct"); lo = wavg(segf,"LowRiskPct")
        f2 = go.Figure(go.Pie(labels=["High","Medium","Low"], values=[hi,me,lo], hole=0.55,
                              marker_colors=[RED,ORANGE,GREEN],
                              textinfo="label+percent", textfont=dict(family="Calibri", size=13)))
        f2.update_layout(height=320, showlegend=False, paper_bgcolor="white",
                         margin=dict(t=10,b=10,l=10,r=10), font=dict(family="Calibri"))
        st.plotly_chart(f2, use_container_width=True)

    # Chart 3 — compounding effect (full width)
    st.markdown("##### The critical finding: culture x leadership compound")
    f3 = go.Figure(go.Bar(
        x=["Both strong","Culture strong /\nLeadership weak","Culture weak /\nLeadership strong","Both weak"],
        y=[98, 65, 51, 12], marker_color=[GREEN, ORANGE, ORANGE, RED],
        text=["98%","65%","51%","12%"], textposition="outside", textfont=dict(family="Calibri", color=NAVY)))
    f3.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                     margin=dict(t=10,b=10,l=10,r=10), yaxis=dict(range=[0,110], title="% recommend"),
                     font=dict(family="Calibri"))
    st.plotly_chart(f3, use_container_width=True)
    st.markdown(f"<div style='background:{NAVY}; color:white; padding:14px 20px; border-radius:8px;'>"
                "<b>Investment → Return:</b> ~£1.15M annual exposure sits against a low-capital lever — "
                "leadership and culture change is mostly behaviour and process, not payroll. "
                "<i>Illustrative exposure under stated assumptions, not guaranteed ROI.</i></div>",
                unsafe_allow_html=True)

# ======================================================================
# RISK & VOICE VIEW
# ======================================================================
else:
    st.markdown("<div class='dashboard-header'><h2>Employee Advocacy — Risk &amp; Voice View</h2>"
                "<p>Operational · score a profile, read the segments, hear the employee voice</p></div>",
                unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "30%", "Flagged High-risk", RED)
    kpi(c2, "0.86", "Model recall (detractors)", GREEN)
    kpi(c3, "45%", "High+Medium coverage", NAVY)
    kpi(c4, "Former", "Highest-risk segment", ORANGE)

    st.markdown("<br>", unsafe_allow_html=True)
    calc, seg_col = st.columns([1, 1])

    # --- Risk calculator ---
    with calc:
        st.markdown("##### Risk calculator — score a profile")
        cu = st.slider("Culture & Values", 1, 5, 3)
        se = st.slider("Senior Management", 1, 5, 3)
        ca = st.slider("Career Opportunities", 1, 5, 3)
        wl = st.slider("Work-Life Balance", 1, 5, 3)
        co = st.slider("Compensation", 1, 5, 3)
        stt = st.radio("Status", ["Current","Former"], horizontal=True)
        risk = predict_risk(cu, se, ca, wl, co, 1 if stt=="Current" else 0)
        lab, colr = band(risk)
        g = go.Figure(go.Indicator(mode="gauge+number", value=risk*100,
            number={"suffix":"%","font":{"size":40,"color":colr,"family":"Georgia"}},
            title={"text":f"<b>{lab} Risk</b>","font":{"size":18,"color":colr,"family":"Georgia"}},
            gauge={"axis":{"range":[0,100]}, "bar":{"color":colr},
                   "steps":[{"range":[0,30],"color":"#E8F6EE"},
                            {"range":[30,60],"color":"#FDF0E3"},
                            {"range":[60,100],"color":"#FBE4E1"}]}))
        g.update_layout(height=240, margin=dict(t=40,b=10,l=20,r=20), paper_bgcolor="white")
        st.plotly_chart(g, use_container_width=True)
        if cu<=2 and se<=2:
            st.warning("⚠️ Compounding effect: both Culture and Senior Management weak → "
                       "advocacy collapses to ~12% in the data. Fix them together.")

    # --- Segment breakdown (honest: what we have) ---
    with seg_col:
        st.markdown("##### Risk by segment (current vs former)")
        f = go.Figure()
        f.add_trace(go.Bar(name="High", x=seg["Status"], y=seg["HighRiskPct"], marker_color=RED,
                           text=[f"{v}%" for v in seg["HighRiskPct"]], textposition="inside"))
        f.add_trace(go.Bar(name="Medium", x=seg["Status"], y=seg["MediumRiskPct"], marker_color=ORANGE,
                           text=[f"{v}%" for v in seg["MediumRiskPct"]], textposition="inside"))
        f.add_trace(go.Bar(name="Low", x=seg["Status"], y=seg["LowRiskPct"], marker_color=GREEN,
                           text=[f"{v}%" for v in seg["LowRiskPct"]], textposition="inside"))
        f.update_layout(barmode="stack", height=300, plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(t=10,b=10,l=10,r=10), font=dict(family="Calibri"),
                        legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(f, use_container_width=True)
        st.caption("Former employees concentrate risk (39.6% High vs 23.2%), but they have already "
                   "left. Target current employees by their weak driver — not by demographic.")

    st.divider()

    # --- Voice of employee ---
    st.markdown("##### Voice of employee — what detractors vs promoters complain about")
    fv = go.Figure()
    fv.add_trace(go.Bar(name="Detractors", x=voice["Theme"], y=voice["Detractors"],
                        marker_color=RED, text=voice["Detractors"], textposition="outside"))
    fv.add_trace(go.Bar(name="Promoters", x=voice["Theme"], y=voice["Promoters"],
                        marker_color=GREEN, text=voice["Promoters"], textposition="outside"))
    fv.update_layout(barmode="group", height=300, plot_bgcolor="white", paper_bgcolor="white",
                     margin=dict(t=10,b=10,l=10,r=10), font=dict(family="Calibri"),
                     legend=dict(orientation="h", y=-0.15), yaxis=dict(title="Mentions"))
    st.plotly_chart(fv, use_container_width=True)
    st.markdown(f"<div style='background:{LGRAY}; padding:14px 20px; border-radius:8px;'>"
                "Detractors mention management <b>2.5× more</b> than promoters. Promoters complain about "
                "work conditions <b>2.6× more</b> — yet still recommend. Rough conditions are tolerated; "
                "bad management is the dealbreaker.</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# FOOTER + caveats (always visible)
# ----------------------------------------------------------------------
st.divider()
st.caption("⚠️ Association, not causation — the model shows where to act; a randomized pilot proves it works.  ·  "
           "'Recommend' is a proxy for advocacy (no attrition/salary data).  ·  "
           "Zikra Aditia — People Analytics & Data Science")
