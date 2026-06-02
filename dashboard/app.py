"""
dashboard/app.py
================
Live Streamlit dashboard for Store Intelligence.

Shows real-time metrics updating as events flow in from the pipeline.
Auto-refreshes every 10 seconds.

Metrics displayed:
  - Live visitor count
  - Conversion rate (gauge)
  - Queue depth (alert indicator)
  - Active anomalies (colour-coded)
  - Zone heatmap (bar chart)
  - Conversion funnel (waterfall)
  - Store health status

To run standalone (without Docker):
    API_URL=http://localhost:8000 streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
STORE_IDS = os.getenv("STORE_IDS", "STORE_BLR_002").split(",")
REFRESH_INTERVAL = 10  # seconds

st.set_page_config(
    page_title="Store Intelligence — Apex Retail",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .anomaly-critical {
        background: #3d1a1a;
        border-left: 4px solid #f38ba8;
        padding: 12px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .anomaly-warn {
        background: #3d2e1a;
        border-left: 4px solid #fab387;
        padding: 12px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .anomaly-info {
        background: #1a2e3d;
        border-left: 4px solid #89b4fa;
        padding: 12px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .stale-badge {
        background: #f38ba8;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
    }
    .ok-badge {
        background: #a6e3a1;
        color: #1e1e2e;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
    }
</style>
""", unsafe_allow_html=True)


# ── API Helpers ───────────────────────────────────────────────────────────────

def fetch(endpoint: str, timeout: int = 5) -> dict | None:
    """Fetch JSON from API endpoint. Returns None on failure."""
    try:
        resp = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None


def severity_color(severity: str) -> str:
    return {"CRITICAL": "#f38ba8", "WARN": "#fab387", "INFO": "#89b4fa"}.get(severity, "#cdd6f4")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏪 Store Intelligence")
    st.caption("Apex Retail — Live Analytics")
    st.divider()

    selected_store = st.selectbox("Store", STORE_IDS, key="store_select")
    auto_refresh = st.toggle("Auto-refresh (10s)", value=True)
    refresh_btn = st.button("🔄 Refresh Now")

    st.divider()
    health_data = fetch("/health")
    if health_data:
        st.markdown(f"**API Status:** {'🟢 Healthy' if health_data['status'] == 'healthy' else '🔴 Degraded'}")
        st.caption(f"Uptime: {health_data.get('uptime_seconds', 0):.0f}s")

        for store in health_data.get("stores", []):
            badge = "ok-badge" if store["status"] == "OK" else "stale-badge"
            st.markdown(
                f"{store['store_id']}: "
                f'<span class="{badge}">{store["status"]}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.error("API unreachable")

    st.divider()
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


# ── Auto-refresh logic ────────────────────────────────────────────────────────
if auto_refresh:
    st.empty()   # placeholder for refresh trigger

# Fetch all data for selected store
metrics = fetch(f"/stores/{selected_store}/metrics")
funnel = fetch(f"/stores/{selected_store}/funnel")
heatmap = fetch(f"/stores/{selected_store}/heatmap")
anomalies = fetch(f"/stores/{selected_store}/anomalies")


# ── Main Header ───────────────────────────────────────────────────────────────
st.title(f"📊 {selected_store}")
if metrics:
    st.caption(f"As of {metrics['as_of'][:19].replace('T', ' ')} UTC")
else:
    st.warning("⚠️ Could not load metrics from API")

st.divider()


# ── KPI Row ───────────────────────────────────────────────────────────────────
if metrics:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "👥 Unique Visitors",
            metrics["unique_visitors"],
            help="Non-staff visitors with at least one ENTRY event today",
        )

    with col2:
        conv_pct = round(metrics["conversion_rate"] * 100, 1)
        st.metric(
            "💳 Conversion Rate",
            f"{conv_pct}%",
            help="Visitors who completed a purchase ÷ total unique visitors",
        )

    with col3:
        dwell = round(metrics["avg_dwell_seconds"])
        m, s = divmod(dwell, 60)
        st.metric(
            "⏱ Avg Dwell",
            f"{m}m {s}s",
            help="Mean dwell time across all zone visits",
        )

    with col4:
        queue = metrics["queue_depth"]
        queue_delta = None
        st.metric(
            "🧾 Queue Depth",
            queue,
            delta=None,
            delta_color="inverse",
            help="Current estimated billing queue depth",
        )

    with col5:
        abandon_pct = round(metrics["abandonment_rate"] * 100, 1)
        st.metric(
            "🚪 Abandon Rate",
            f"{abandon_pct}%",
            help="Billing queue abandonments ÷ total queue joins",
        )
else:
    st.info("No metrics data available yet. Run the detection pipeline and ingest events.")


# ── Anomalies ─────────────────────────────────────────────────────────────────
st.subheader("🚨 Active Anomalies")
if anomalies and anomalies.get("active_anomalies"):
    for a in anomalies["active_anomalies"]:
        css_class = f"anomaly-{a['severity'].lower()}"
        icon = {"CRITICAL": "🔴", "WARN": "🟡", "INFO": "🔵"}.get(a["severity"], "⚪")
        st.markdown(
            f'<div class="{css_class}">'
            f'<strong>{icon} {a["anomaly_type"]}</strong> '
            f'<span style="color: #cdd6f4; font-size: 0.85em;">[{a["severity"]}]</span><br>'
            f'{a["description"]}<br>'
            f'<em>→ {a["suggested_action"]}</em>'
            f"</div>",
            unsafe_allow_html=True,
        )
else:
    st.success("✅ No active anomalies")


st.divider()


# ── Two-column section: Funnel + Heatmap ──────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📉 Conversion Funnel")
    if funnel and funnel.get("stages"):
        stages_df = pd.DataFrame(funnel["stages"])

        # Waterfall-style funnel chart
        fig_funnel = go.Figure(go.Funnel(
            y=stages_df["stage"],
            x=stages_df["count"],
            textinfo="value+percent initial",
            marker=dict(color=["#89b4fa", "#74c7ec", "#a6e3a1", "#cba6f7"]),
        ))
        fig_funnel.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdd6f4"),
            height=320,
        )
        st.plotly_chart(fig_funnel, width="stretch")

        # Drop-off table
        drop_df = stages_df[["stage", "count", "drop_off_pct"]].copy()
        drop_df.columns = ["Stage", "Count", "Drop-off %"]
        st.dataframe(drop_df, hide_index=True, width="stretch")
    else:
        st.info("No funnel data yet.")

with col_right:
    st.subheader("🗺 Zone Heatmap")
    if heatmap and heatmap.get("zones"):
        zones_df = pd.DataFrame(heatmap["zones"])

        fig_heat = px.bar(
            zones_df,
            x="normalised_score",
            y="zone_id",
            orientation="h",
            color="normalised_score",
            color_continuous_scale="Viridis",
            text=zones_df["visit_count"].apply(lambda x: f"{x} visits"),
            labels={"normalised_score": "Heat Score (0–100)", "zone_id": "Zone"},
        )
        fig_heat.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdd6f4"),
            coloraxis_showscale=False,
            height=320,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, width="stretch")

        # Low confidence warning
        low_conf = [z for z in heatmap["zones"] if not z.get("data_confidence", True)]
        if low_conf:
            st.warning(
                f"⚠️ Low data confidence on {len(low_conf)} zone(s) — fewer than 20 sessions."
            )
    else:
        st.info("No zone data yet.")


# ── Zone Dwell Table ──────────────────────────────────────────────────────────
st.divider()
st.subheader("⏱ Zone Dwell Times")
if metrics and metrics.get("zone_metrics"):
    zm_df = pd.DataFrame(metrics["zone_metrics"])
    zm_df["avg_dwell_seconds"] = zm_df["avg_dwell_seconds"].apply(
        lambda s: f"{int(s//60)}m {int(s%60)}s"
    )
    zm_df.columns = ["Zone", "Avg Dwell", "Visit Count"]
    zm_df = zm_df.sort_values("Visit Count", ascending=False)
    st.dataframe(zm_df, hide_index=True, width="stretch")
else:
    st.info("No zone dwell data yet.")


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()