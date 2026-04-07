"""
Spanish Electricity Mix — Analysis & Projections 2025-2030
Developed for EREN Investment Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
from plotly.subplots import make_subplots
from datetime import date as dt_date, timedelta as dt_timedelta

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Spanish Electricity Mix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; border-radius: 6px 6px 0 0; }
    .quality-banner { border-radius: 6px; padding: 8px 12px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

# Estimated installed capacities — Spain end-2025 (GW)
BASE_SOLAR_GW   = 32.0
BASE_WIND_GW    = 31.0
BASE_BESS_GW    = 0.5   # near-zero in 2025

MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",
    5:"May", 6:"Jun", 7:"Jul", 8:"Aug",
    9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
}

SEASON_MAP = {
    1:"Winter",  2:"Winter",  3:"Spring",
    4:"Spring",  5:"Spring",  6:"Summer",
    7:"Summer",  8:"Summer",  9:"Autumn",
    10:"Autumn", 11:"Autumn", 12:"Winter"
}

SEASON_COLORS = {
    "Winter": "#4169E1",
    "Spring": "#2E8B57",
    "Summer": "#FFD700",
    "Autumn": "#D2691E",
}

# Technologies stacked in the dispatch chart
DISPATCH_TECHS = [
    "Solar PV", "Wind", "Nuclear", "Hydro (on-flow)",
    "Combined Cycle GT", "Cogeneration", "Solar Thermal",
    "Coal", "Other Renewables", "Natural Gas - Steam Turbine",
    "Pumping Turbine", "Hybridization", "Non-renewable Waste",
]

RENEWABLES = ["Solar PV", "Wind", "Hydro (on-flow)", "Solar Thermal", "Other Renewables"]
FOSSIL     = ["Combined Cycle GT", "Coal", "Natural Gas - Steam Turbine", "Fuel Oil"]

COLORS = {
    "Solar PV":                    "#FFD700",
    "Wind":                        "#87CEEB",
    "Nuclear":                     "#9370DB",
    "Hydro (on-flow)":             "#4169E1",
    "Combined Cycle GT":           "#FF6347",
    "Cogeneration":                "#D2691E",
    "Solar Thermal":               "#FFA500",
    "Coal":                        "#696969",
    "Other Renewables":            "#90EE90",
    "Natural Gas - Steam Turbine": "#FF8C00",
    "Pumping Turbine":             "#00CED1",
    "Hybridization":               "#20B2AA",
    "Non-renewable Waste":         "#A0522D",
    "Battery Discharge":           "#00FA9A",
    "Battery Charging":            "#FF69B4",
    "⚡ Curtailment":              "#FF000044",
}

# ============================================================================
# DATA LOADING & IMPUTATION
# ============================================================================

@st.cache_data
def _load_and_impute():
    """
    Load REE 2025 hourly data and impute missing hours.

    Strategy:
      1. Load raw CSV.
      2. Build the complete 8,760-hour 2025 calendar.
      3. Merge valid data (Demand > 0) onto the full grid.
      4. For every missing hour: fill each column with the mean of
         (same month, same hour) across all valid days of that month.
      5. Return the imputed DataFrame + data-quality metadata.
    """
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "spain_mix_data.csv")
    df_raw = pd.read_csv(csv_path, parse_dates=["Date"])

    # Columns to impute (everything except the datetime metadata)
    exclude = {"Date", "Year", "Month", "Day", "Hour"}
    numeric_cols = [c for c in df_raw.columns if c not in exclude]

    # Valid rows only (Demand > 0)
    df_valid = df_raw[df_raw["Demand"] > 0].copy()

    # ── Build complete 2025 hourly grid ──────────────────────────────────────
    full_dt = pd.date_range("2025-01-01 00:00", "2025-12-31 23:00", freq="h")
    full_df = pd.DataFrame({
        "Date":  full_dt,
        "Year":  full_dt.year,
        "Month": full_dt.month,
        "Day":   full_dt.day,
        "Hour":  full_dt.hour,
    })

    # Merge valid data onto the complete grid (left join → NaN for missing hours)
    merged = full_df.merge(
        df_valid[["Date"] + numeric_cols], on="Date", how="left"
    )

    # ── Compute monthly-hourly averages from valid data ───────────────────────
    monthly_avg = df_valid.groupby(["Month", "Hour"])[numeric_cols].mean()

    # Fallback: months with zero valid data (e.g. April) inherit the average
    # of their nearest available neighbours (March + May → April).
    available_months = set(monthly_avg.index.get_level_values("Month"))
    missing_months   = set(range(1, 13)) - available_months
    for m in sorted(missing_months):
        prev_m = max((x for x in available_months if x < m), default=None)
        next_m = min((x for x in available_months if x > m), default=None)
        for h in range(24):
            if prev_m is not None and next_m is not None:
                row = (monthly_avg.loc[(prev_m, h)] + monthly_avg.loc[(next_m, h)]) / 2
            elif prev_m is not None:
                row = monthly_avg.loc[(prev_m, h)]
            else:
                row = monthly_avg.loc[(next_m, h)]
            monthly_avg.loc[(m, h), :] = row
    monthly_avg = monthly_avg.sort_index()

    # ── Fill missing rows ─────────────────────────────────────────────────────
    missing_mask = merged["Demand"].isna()

    if missing_mask.any():
        # For each missing hour, look up (Month, Hour) in monthly_avg
        fill_keys = (
            merged.loc[missing_mask, ["Month", "Hour"]]
            .reset_index()                                  # preserve original index
            .merge(monthly_avg.reset_index(), on=["Month", "Hour"], how="left")
            .set_index("index")
        )
        for col in numeric_cols:
            merged.loc[missing_mask, col] = fill_keys[col].values

    # ── Derived columns ───────────────────────────────────────────────────────
    merged["MonthName"] = merged["Month"].map(MONTH_NAMES)
    merged["Season"]    = merged["Month"].map(SEASON_MAP)
    merged["DayOfYear"] = merged["Date"].dt.dayofyear
    merged["VRE_share"] = (merged["Solar PV"] + merged["Wind"]) / merged["Demand"]
    merged["_imputed"]  = missing_mask.values  # flag for chart annotations

    # ── Quality metadata ──────────────────────────────────────────────────────
    imputed_dates_set = set(merged.loc[missing_mask, "Date"].dt.date)
    raw_valid         = len(df_valid)

    metadata = {
        "raw_hours":         raw_valid,
        "expected_hours":    8760,
        "imputed_hours":     int(missing_mask.sum()),
        "imputed_days":      len(imputed_dates_set),
        "imputed_date_list": sorted(str(d) for d in imputed_dates_set),
        "completeness_pct":  raw_valid / 8760 * 100,
        "date_min":          df_valid["Date"].min(),
        "date_max":          df_valid["Date"].max(),
    }

    return merged, metadata


def load_data() -> pd.DataFrame:
    """Return the complete, imputed 8,760-hour DataFrame."""
    return _load_and_impute()[0]


def get_data_quality() -> dict:
    """Return data-quality metadata (raw hours, imputed days, completeness %)."""
    return _load_and_impute()[1]


# ============================================================================
# BATTERY DISPATCH SIMULATION
# ============================================================================

def simulate_battery(surplus_arr: np.ndarray, cap_mw: float,
                     duration_h: float, efficiency: float = 0.90):
    """
    Simplified dispatch: charge on VRE surplus, discharge on deficit.
    Returns (charge, discharge, soc) in MW / MWh.
    """
    capacity_mwh = cap_mw * duration_h
    n = len(surplus_arr)
    soc      = np.zeros(n + 1)
    charge   = np.zeros(n)
    discharge= np.zeros(n)

    for i in range(n):
        s = surplus_arr[i]
        if s > 0:
            max_ch    = min(cap_mw, s, (capacity_mwh - soc[i]) / max(efficiency, 0.01))
            charge[i] = max(0.0, max_ch)
            soc[i+1]  = soc[i] + charge[i] * efficiency
        elif s < 0 and soc[i] > 0:
            max_dis     = min(cap_mw, -s, soc[i])
            discharge[i]= max(0.0, max_dis)
            soc[i+1]    = soc[i] - discharge[i]
        else:
            soc[i+1] = soc[i]

    return charge, discharge, soc[:-1]


# ============================================================================
# PROJECTION ENGINE
# ============================================================================

@st.cache_data
def project_year(delta_solar_gw: float, delta_wind_gw: float,
                 bess_total_gw: float, bess_dur_h: float,
                 cum_demand_growth_pct: float):
    """
    Project the energy mix for a given scenario.

    Simplified modelling assumptions:
    - PV and wind production are scaled proportionally to capacity additions.
    - Nuclear, hydro and cogeneration remain constant.
    - CCGT fills residual demand (capped at 2025 observed maximum).
    - BESS charges on VRE surplus, discharges on deficit.
    """
    df = load_data().copy()

    solar_scale  = 1 + delta_solar_gw / BASE_SOLAR_GW
    wind_scale   = 1 + delta_wind_gw  / BASE_WIND_GW
    demand_scale = 1 + cum_demand_growth_pct / 100.0

    df["Solar PV"]      = df["Solar PV"]      * solar_scale
    df["Wind"]          = df["Wind"]          * wind_scale
    df["Solar Thermal"] = df["Solar Thermal"] * solar_scale
    df["Demand"]        = df["Demand"]        * demand_scale

    # "Predictable" generation: VRE + must-run
    must_run = (df["Nuclear"] + df["Hydro (on-flow)"] + df["Cogeneration"]
                + df["Solar Thermal"] + df["Other Renewables"])
    surplus  = df["Solar PV"] + df["Wind"] + must_run - df["Demand"]

    # BESS simulation
    if bess_total_gw > 0:
        cap_mw  = bess_total_gw * 1000
        ch, dis, soc = simulate_battery(surplus.values, cap_mw, bess_dur_h)
        batt_charge    = pd.Series(ch,  index=df.index)
        batt_discharge = pd.Series(dis, index=df.index)
        batt_soc       = pd.Series(soc, index=df.index)
    else:
        batt_charge    = pd.Series(0.0, index=df.index)
        batt_discharge = pd.Series(0.0, index=df.index)
        batt_soc       = pd.Series(0.0, index=df.index)

    net_surplus   = surplus + batt_discharge - batt_charge
    curtailment   = net_surplus.clip(lower=0)
    deficit       = (-net_surplus).clip(lower=0)
    ccgt_cap      = df["Combined Cycle GT"].max()
    ccgt_dispatch = deficit.clip(upper=ccgt_cap)

    # Metrics
    total_demand = df["Demand"].sum()
    ren_gen      = (df["Solar PV"] + df["Wind"] + df["Hydro (on-flow)"]
                    + df["Solar Thermal"] + df["Other Renewables"]).sum()
    vre_share_s  = (df["Solar PV"] + df["Wind"]) / df["Demand"]
    bess_energy  = bess_total_gw * 1000 * bess_dur_h if bess_total_gw > 0 else 1

    return {
        "df":              df,
        "batt_charge":     batt_charge,
        "batt_discharge":  batt_discharge,
        "batt_soc":        batt_soc,
        "curtailment":     curtailment,
        "ccgt_dispatch":   ccgt_dispatch,
        "solar_twh":       df["Solar PV"].sum() / 1e6,
        "wind_twh":        df["Wind"].sum()     / 1e6,
        "demand_twh":      total_demand          / 1e6,
        "renewable_pct":   ren_gen / total_demand * 100,
        "curtailment_twh": curtailment.sum()    / 1e6,
        "ccgt_twh":        ccgt_dispatch.sum()  / 1e6,
        "batt_cycles":     batt_discharge.sum() / bess_energy,
        "cannib_70":       int((vre_share_s > 0.70).sum()),
        "cannib_90":       int((vre_share_s > 0.90).sum()),
        "surplus_hours":   int((net_surplus > 0).sum()),
    }


# ============================================================================
# OMIE DAY-AHEAD PRICES — LOAD OR GENERATE
# ============================================================================

@st.cache_data
def load_or_generate_omie_prices(_df_mix):
    """
    Load omie_prices_2025.csv if available; otherwise generate synthetic
    Day-Ahead prices calibrated against real Spanish 2025 observations.

    Price model: simplified merit-order
      Price = Seasonal_base + Hour_effect + β_PV × PV_penetration
              + β_VRE × VRE_share + noise
    Calibrated to reproduce:
      - Annual base price ~55-60 EUR/MWh
      - Spring midday trough: 5-15 EUR/MWh
      - Evening peak: 70-90 EUR/MWh
      - ~200-300 negative hours/year (mainly spring/summer midday)
    """
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "omie_prices_2025.csv")
    if os.path.exists(csv_path):
        df_p = pd.read_csv(csv_path, parse_dates=["datetime"])
        return df_p

    # --- Synthetic calibrated generation ---
    # Calibrated against OMIE 2025 observations:
    #   DA avg price ~51 EUR/MWh | PV Capture Price ~32 EUR/MWh (cannib ~62%)
    #   Midday/evening spread ~60 EUR/MWh | ~240 hours < 10 EUR/MWh
    np.random.seed(42)
    df = _df_mix.copy()
    n  = len(df)

    pv_pen = df["Solar PV"] / df["Demand"]

    # Seasonal base (EUR/MWh) — winter expensive (gas), spring low (VRE + low demand)
    season_base = df["Month"].map({
        1: 78, 2: 74, 3: 62, 4: 57, 5: 52, 6: 60,
        7: 67, 8: 69, 9: 65, 10: 63, 11: 71, 12: 76,
    }).astype(float)

    # Hourly profile (evening peak 19-21h, overnight trough 2-4h)
    hour_effect = df["Hour"].map({
        0: -10, 1: -13, 2: -15, 3: -16, 4: -14, 5: -8,
        6: -2,  7: 3,   8: 6,   9: 8,   10: 6,  11: 3,
        12: 0,  13: -1, 14: -2, 15: -1, 16: 2,  17: 6,
        18: 14, 19: 20, 20: 24, 21: 22, 22: 12, 23: 0,
    }).astype(float)

    # PV cannibalisation effect — non-linear (convex)
    pv_effect   = -80 * pv_pen**1.25

    # Complementary wind effect
    wind_pen    = df["Wind"] / df["Demand"]
    wind_effect = -20 * wind_pen

    # Stochastic noise (base volatility + occasional price spikes)
    noise  = np.random.normal(0, 5, n)
    spikes = np.random.choice([0, 1], size=n, p=[0.97, 0.03])
    noise += spikes * np.random.uniform(15, 60, n)

    # Final price (floor at -30 EUR/MWh)
    price = (season_base + hour_effect + pv_effect + wind_effect + noise).clip(lower=-30)

    df_omie = pd.DataFrame({
        "datetime":      df["Date"],
        "date":          df["Date"].dt.date,
        "hour":          df["Hour"],
        "month":         df["Month"],
        "price_eur_mwh": price.round(2),
    })

    return df_omie


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # ── Session-state defaults for projection parameters ─────────────────────
    for _k, _v in [("proj_solar", 4.0), ("proj_wind", 2.0), ("proj_bess", 0.5),
                   ("proj_bess_dur", 4), ("proj_demand", 1.5)]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    df  = load_data()
    dq  = get_data_quality()

    # ── OMIE prices — loaded once, shared across all tabs ────────────────────
    _df_omie = load_or_generate_omie_prices(df)
    _df_omie["_date"] = _df_omie["datetime"].dt.date.astype(str)
    _df_omie["_hour"] = _df_omie["hour"].astype(int)
    _dfw = df.copy()
    _dfw["_date"] = _dfw["Date"].dt.date.astype(str)
    _dfw["_hour"] = _dfw["Hour"].astype(int)
    df_merged = _dfw.merge(
        _df_omie[["_date", "_hour", "price_eur_mwh"]],
        on=["_date", "_hour"], how="left"
    )
    if df_merged["price_eur_mwh"].isna().any():
        _mfill = df_merged.groupby(["Month", "Hour"])["price_eur_mwh"].transform("median")
        df_merged["price_eur_mwh"] = df_merged["price_eur_mwh"].fillna(_mfill)
    df_merged["PV_penetration"]  = df_merged["Solar PV"] / df_merged["Demand"]
    df_merged["VRE_penetration"] = (df_merged["Solar PV"] + df_merged["Wind"]) / df_merged["Demand"]

    # ── TITLE ────────────────────────────────────────────────────────────────
    st.title("⚡ Spanish Electricity Mix — Analysis & Projections 2025-2030")

    # ── DATA QUALITY BANNER ──────────────────────────────────────────────────
    pct = dq["completeness_pct"]
    if pct >= 98:
        indicator = "🟢"
        quality_label = "Excellent"
        banner_type   = "success"
    elif pct >= 90:
        indicator = "🟡"
        quality_label = "Good"
        banner_type   = "warning"
    else:
        indicator = "🔴"
        quality_label = "Incomplete"
        banner_type   = "error"

    banner_text = (
        f"{indicator} **Data Quality: {quality_label}** — "
        f"**{dq['raw_hours']:,}** raw hours loaded out of **8,760** expected "
        f"({pct:.1f}% raw coverage) · "
        f"**{dq['imputed_days']} days imputed** using monthly hourly averages "
        f"(March 19/25/30, all April, May 30) · "
        f"Date range: {dq['date_min'].strftime('%d %b %Y')} → "
        f"{dq['date_max'].strftime('%d %b %Y')} · "
        f"Dataset after imputation: **{len(df):,} hours** (complete 2025)"
    )

    if banner_type == "success":
        st.success(banner_text)
    elif banner_type == "warning":
        st.warning(banner_text)
    else:
        st.error(banner_text)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 2025 Reference")
        st.markdown("**Installed Capacities (end-2025)**")
        st.markdown(f"☀️ Solar PV: **{BASE_SOLAR_GW:.0f} GW**")
        st.markdown(f"💨 Wind: **{BASE_WIND_GW:.0f} GW**")
        st.markdown(f"🔋 BESS: **{BASE_BESS_GW:.1f} GW** (near-zero)")
        st.markdown("💧 Pumped Hydro: **~3.3 GW**")
        st.markdown("⚛️ Nuclear: **~7.4 GW**")
        st.divider()
        st.info(
            "**Sources:** REE 2025 hourly generation data · "
            "OMIE 2025 Day-Ahead prices · "
            "PNIEC 2021-2030 capacity targets"
        )
        st.divider()
        st.markdown("### 📥 Export Data")
        _export_clicked = st.button("📊 Export to Excel", use_container_width=True,
                                     help="Download KPIs + monthly data + price summary as .xlsx")
        if _export_clicked:
            import io, openpyxl
            from openpyxl.styles import Font, PatternFill
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as _xw:
                # Sheet 1: Annual KPIs
                _kpi_data = {
                    "Metric": [
                        "Total Demand (TWh)", "Total Generation (TWh)",
                        "Renewable Share (%)", "VRE Share (PV+Wind, %)",
                        "Avg DA Price (€/MWh)", "PV Capture Price (€/MWh)",
                        "Hours VRE > 70%", "Hours PV > 50%",
                        "Hours Negative Price", "Hours < 10 €/MWh",
                    ],
                    "Value": [
                        round(df_merged["Demand"].sum()/1e6, 1),
                        round(df_merged[[t for t in DISPATCH_TECHS if t in df_merged.columns and df_merged[t].sum()>0]].sum().sum()/1e6, 1),
                        round(sum(df_merged[t].sum() for t in RENEWABLES if t in df_merged.columns)/df_merged["Demand"].sum()*100, 1),
                        round((df_merged["Solar PV"]+df_merged["Wind"]).sum()/df_merged["Demand"].sum()*100, 1),
                        round(df_merged["price_eur_mwh"].mean(), 1),
                        round((df_merged.loc[df_merged["Solar PV"]>0,"price_eur_mwh"]*df_merged.loc[df_merged["Solar PV"]>0,"Solar PV"]).sum()/df_merged.loc[df_merged["Solar PV"]>0,"Solar PV"].sum(), 1),
                        int((df_merged["VRE_share"]>0.70).sum()),
                        int((df_merged["Solar PV"]/df_merged["Demand"]>0.50).sum()),
                        int((df_merged["price_eur_mwh"]<0).sum()),
                        int((df_merged["price_eur_mwh"]<10).sum()),
                    ]
                }
                pd.DataFrame(_kpi_data).to_excel(_xw, sheet_name="Annual KPIs", index=False)
                # Sheet 2: Monthly generation
                _techs_ex = [t for t in DISPATCH_TECHS if t in df_merged.columns and df_merged[t].sum()>0]
                _mo_ex = df_merged.groupby("Month")[_techs_ex].sum()/1e6
                _mo_ex.index = [MONTH_NAMES[m] for m in _mo_ex.index]
                _mo_ex.columns = [f"{c} (TWh)" for c in _mo_ex.columns]
                _mo_ex.to_excel(_xw, sheet_name="Monthly Generation")
                # Sheet 3: Monthly prices
                _px_mo = df_merged.groupby("Month").agg(
                    Avg_DA_Price=("price_eur_mwh", "mean"),
                    PV_Capture_Price=("price_eur_mwh", lambda x: (x * df_merged.loc[x.index,"Solar PV"]).sum()/max(df_merged.loc[x.index,"Solar PV"].sum(),1)),
                    Hours_Negative=("price_eur_mwh", lambda x: int((x<0).sum())),
                    Hours_Below_10=("price_eur_mwh", lambda x: int((x<10).sum())),
                )
                _px_mo.index = [MONTH_NAMES[m] for m in _px_mo.index]
                _px_mo.to_excel(_xw, sheet_name="Monthly Prices")
            buf.seek(0)
            st.download_button(
                "⬇️ Download Excel file",
                data=buf,
                file_name="spain_energy_mix_2025.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dispatch 2025",
        "📅 Monthly Analysis",
        "📈 Mix KPIs",
        "💶 OMIE Prices & PV Cannibalisation",
        "🌡️ Heatmaps",
        "🏭 Flexibility Assets",
        "🔭 Projections 2026-2030",
    ])

    # Reverse month-name → month-number lookup
    month_num = {v: k for k, v in MONTH_NAMES.items()}

    # Pre-compute imputed dates as Timestamps for chart annotations
    imputed_ts = [pd.Timestamp(d) for d in dq["imputed_date_list"]]

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — DISPATCH 2025
    # ════════════════════════════════════════════════════════════════════════
    with tab1:

        # ── Local helpers ─────────────────────────────────────────────────────
        def _hex_rgba(hex_color: str, alpha: float) -> str:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        def _tech_vals(df_sub: pd.DataFrame, tech: str) -> pd.Series:
            """Return series for real or computed technology aliases."""
            if tech == "VRE":
                return df_sub["Solar PV"] + df_sub["Wind"]
            if tech == "Residual Load":
                return (df_sub["Demand"] - df_sub["Solar PV"] - df_sub["Wind"]).clip(lower=0)
            return df_sub[tech]

        techs_nz = [t for t in DISPATCH_TECHS if df[t].sum() > 0]

        INSTALLED_CAP_MW = {
            "Solar PV":          BASE_SOLAR_GW * 1000,
            "Wind":              BASE_WIND_GW  * 1000,
            "Nuclear":           7400.0,
            "Hydro (on-flow)":   13000.0,
            "Combined Cycle GT": float(df["Combined Cycle GT"].max()),
            "Cogeneration":      float(df["Cogeneration"].max()),
            "Solar Thermal":     2300.0,
            "Pumping Turbine":   3300.0,
        }

        # ════════════════════════════════════════════════════════════════════
        # SECTION 0 — ANNUAL OVERVIEW  (5.1.1)
        # ════════════════════════════════════════════════════════════════════
        st.subheader("2025 Annual Overview")

        # Five headline KPIs
        ann_dem_twh  = df_merged["Demand"].sum() / 1e6
        ann_gen_twh  = df_merged[techs_nz].sum().sum() / 1e6
        ann_avg_px   = df_merged["price_eur_mwh"].mean()
        _pv_mask_ann = df_merged["Solar PV"] > 0
        ann_pv_cap   = (
            (df_merged.loc[_pv_mask_ann, "price_eur_mwh"]
             * df_merged.loc[_pv_mask_ann, "Solar PV"]).sum()
            / df_merged.loc[_pv_mask_ann, "Solar PV"].sum()
        )
        ann_h_neg = int((df_merged["price_eur_mwh"] < 0).sum())

        _ov1, _ov2, _ov3, _ov4, _ov5 = st.columns(5)
        _ov1.metric("Total Demand",          f"{ann_dem_twh:.1f} TWh")
        _ov2.metric("Total Generation",      f"{ann_gen_twh:.1f} TWh")
        _ov3.metric("Avg DA Price",          f"{ann_avg_px:.1f} €/MWh")
        _ov4.metric("PV Capture Price",      f"{ann_pv_cap:.1f} €/MWh",
                    delta=f"{ann_pv_cap - ann_avg_px:.1f} €", delta_color="inverse")
        _ov5.metric("Negative-Price Hours",  f"{ann_h_neg:,} h")

        # Dual-axis monthly chart ─────────────────────────────────────────────
        _mo_grp   = df_merged.groupby("Month")
        _mo_gen   = _mo_grp[techs_nz].sum() / 1e6        # TWh per month
        _mo_dem   = _mo_grp["Demand"].sum()  / 1e6
        _mo_px    = _mo_grp["price_eur_mwh"].mean()
        _mo_pvcap = {}
        for _mid, _grp in _mo_grp:
            _m_pv = _grp["Solar PV"] > 0
            _mo_pvcap[_mid] = (
                (_grp.loc[_m_pv, "price_eur_mwh"] * _grp.loc[_m_pv, "Solar PV"]).sum()
                / _grp.loc[_m_pv, "Solar PV"].sum()
            ) if _m_pv.sum() > 0 else 0.0

        _x_mo = [MONTH_NAMES[m] for m in sorted(_mo_gen.index)]

        fig_ov = go.Figure()
        for _t in techs_nz:
            fig_ov.add_trace(go.Bar(
                name=_t,
                x=_x_mo,
                y=[_mo_gen.loc[_m, _t] for _m in sorted(_mo_gen.index)],
                marker_color=COLORS.get(_t, "#AAA"), yaxis="y",
                hovertemplate=f"<b>{_t}</b>: %{{y:.1f}} TWh<extra></extra>",
            ))
        fig_ov.add_trace(go.Scatter(
            x=_x_mo, y=[_mo_dem.loc[_m] for _m in sorted(_mo_gen.index)],
            mode="lines+markers", name="Demand",
            line=dict(color="black", width=2), yaxis="y",
            hovertemplate="<b>Demand</b>: %{y:.1f} TWh<extra></extra>",
        ))
        fig_ov.add_trace(go.Scatter(
            x=_x_mo, y=[_mo_px.loc[_m] for _m in sorted(_mo_gen.index)],
            mode="lines+markers", name="Avg DA Price",
            line=dict(color="darkorange", width=2.5, dash="dash"),
            yaxis="y2",
            hovertemplate="<b>Avg DA Price</b>: %{y:.1f} €/MWh<extra></extra>",
        ))
        fig_ov.add_trace(go.Scatter(
            x=_x_mo, y=[_mo_pvcap[_m] for _m in sorted(_mo_gen.index)],
            mode="lines+markers", name="PV Capture Price",
            line=dict(color="#DAA520", width=2.5, dash="dot"),
            marker=dict(symbol="star", size=8),
            yaxis="y2",
            hovertemplate="<b>PV Capture Price</b>: %{y:.1f} €/MWh<extra></extra>",
        ))
        _px_max = max(_mo_px.max(), ann_avg_px * 1.5)
        fig_ov.update_layout(
            barmode="stack", height=440,
            title="Monthly Generation by Technology (TWh) with Day-Ahead Prices",
            xaxis_title="Month",
            yaxis=dict(title="Generation / Demand (TWh)", gridcolor="#eee"),
            yaxis2=dict(title="Price (€/MWh)", overlaying="y", side="right",
                        showgrid=False, range=[0, _px_max * 1.3]),
            legend=dict(orientation="h", y=-0.32, font=dict(size=10)),
            plot_bgcolor="white", hovermode="x unified",
        )
        st.plotly_chart(fig_ov, use_container_width=True)

        st.info(
            "**What this shows:** Full-year 2025 generation by technology (stacked bars, left "
            "axis) with monthly average Day-Ahead price and production-weighted PV Capture Price "
            "(lines, right axis). "
            "**How to read:** When the PV Capture Price (gold dotted) diverges below the DA "
            "Average (orange dashed), solar cannibalisation is active — PV generators receive "
            "less than the market average. The gap is widest in spring/summer. "
            "**Limitation:** April data is imputed from March/May averages and its prices "
            "should be treated as indicative."
        )

        st.divider()

        # ════════════════════════════════════════════════════════════════════
        # SECTION 1 — DISPATCH VIEW CONTROLS
        # ════════════════════════════════════════════════════════════════════
        _dc1, _dc2, _dc3 = st.columns([1, 2, 0.8])
        sel_month = _dc1.selectbox(
            "Month", list(MONTH_NAMES.values()), index=0, key="t1_month"
        )
        view_mode = _dc2.radio(
            "View mode",
            ["Typical Week", "Full Month", "Typical Day", "Specific Date"],
            horizontal=True, key="t1_view",
        )
        show_dem = _dc3.checkbox("Show Demand", value=True, key="t1_dem")

        m     = month_num[sel_month]
        df_m  = df_merged[df_merged["Month"] == m].copy()

        # ════════════════════════════════════════════════════════════════════
        # VIEW A — SPECIFIC DATE  (5.1.2)
        # ════════════════════════════════════════════════════════════════════
        if view_mode == "Specific Date":
            _max_day_m  = int(df_m["Date"].dt.day.max())
            _default_dt = dt_date(2025, m, min(15, _max_day_m))
            sel_date = st.date_input(
                "Select a day",
                value=_default_dt,
                min_value=dt_date(2025, 1, 1),
                max_value=dt_date(2025, 12, 31),
                key="t1_date",
            )
            df_day = df_merged[df_merged["Date"].dt.date == sel_date].sort_values("Hour")

            if df_day.empty:
                st.warning(f"No data available for {sel_date}.")
            else:
                techs_d  = [t for t in DISPATCH_TECHS
                            if t in df_day.columns and df_day[t].sum() > 0]
                is_imp   = bool(df_day["_imputed"].any())

                fig_d = make_subplots(specs=[[{"secondary_y": True}]])

                if is_imp:
                    fig_d.add_vrect(
                        x0=-0.5, x1=23.5,
                        fillcolor="rgba(180,180,180,0.15)", line_width=0,
                        annotation_text="⚠ Imputed day",
                        annotation_position="top left",
                        annotation_font=dict(size=9, color="gray"),
                    )

                for _t in techs_d:
                    fig_d.add_trace(go.Scatter(
                        x=df_day["Hour"].tolist(), y=df_day[_t].tolist(),
                        mode="lines", stackgroup="one", name=_t,
                        line=dict(width=0), fillcolor=COLORS.get(_t, "#AAA"),
                        hovertemplate=f"<b>{_t}</b>: %{{y:.0f}} MW<extra></extra>",
                    ), secondary_y=False)

                if show_dem:
                    fig_d.add_trace(go.Scatter(
                        x=df_day["Hour"].tolist(), y=df_day["Demand"].tolist(),
                        mode="lines", name="Demand",
                        line=dict(color="black", width=2, dash="dot"),
                        hovertemplate="<b>Demand</b>: %{y:.0f} MW<extra></extra>",
                    ), secondary_y=False)

                fig_d.add_trace(go.Scatter(
                    x=df_day["Hour"].tolist(), y=df_day["price_eur_mwh"].tolist(),
                    mode="lines+markers", name="DA Price",
                    line=dict(color="darkorange", width=2.5),
                    marker=dict(size=5),
                    hovertemplate="<b>DA Price</b>: %{y:.1f} €/MWh<extra></extra>",
                ), secondary_y=True)

                fig_d.add_hline(y=0, line_dash="dash", line_color="red",
                                line_width=1, opacity=0.4, secondary_y=True)

                fig_d.update_layout(
                    height=490, hovermode="x unified",
                    title=(f"Hourly Dispatch + DA Price — "
                           f"{sel_date.strftime('%A, %d %B %Y')}"
                           + (" (⚠ imputed)" if is_imp else "")),
                    xaxis=dict(title="Hour (0–23)", tickmode="linear", dtick=1,
                               range=[-0.5, 23.5]),
                    yaxis=dict(title="Generation (MW)", gridcolor="#eee"),
                    yaxis2=dict(title="DA Price (€/MWh)", showgrid=False),
                    plot_bgcolor="white",
                    legend=dict(orientation="h", y=-0.30, font=dict(size=10)),
                )
                st.plotly_chart(fig_d, use_container_width=True)

                # Daily KPIs row 1
                _pd_row  = df_day.loc[df_day["Demand"].idxmax()]
                _ppv_row = df_day.loc[df_day["Solar PV"].idxmax()]
                _mnpx    = df_day.loc[df_day["price_eur_mwh"].idxmin()]
                _mxpx    = df_day.loc[df_day["price_eur_mwh"].idxmax()]
                _pv_d    = df_day["Solar PV"] > 0
                _dly_cap = (
                    (df_day.loc[_pv_d, "price_eur_mwh"] * df_day.loc[_pv_d, "Solar PV"]).sum()
                    / df_day.loc[_pv_d, "Solar PV"].sum()
                ) if _pv_d.sum() > 0 else 0.0
                _dly_vre = ((df_day["Solar PV"] + df_day["Wind"]).sum()
                            / df_day["Demand"].sum() * 100)
                _h_pv50  = int((df_day["Solar PV"]
                                / df_day["Demand"].replace(0, np.nan) > 0.5).sum())

                dk1, dk2, dk3, dk4 = st.columns(4)
                dk1.metric("Peak Demand",
                           f"{_pd_row['Demand']:,.0f} MW",
                           help=f"at hour {int(_pd_row['Hour'])}h")
                dk2.metric("Peak PV",
                           f"{_ppv_row['Solar PV']:,.0f} MW",
                           help=f"at hour {int(_ppv_row['Hour'])}h")
                dk3.metric("Daily Avg Price",   f"{df_day['price_eur_mwh'].mean():.1f} €/MWh")
                dk4.metric("PV Capture Price",  f"{_dly_cap:.1f} €/MWh")

                dk5, dk6, dk7, dk8 = st.columns(4)
                dk5.metric("Min Price",
                           f"{_mnpx['price_eur_mwh']:.1f} €/MWh",
                           help=f"at hour {int(_mnpx['Hour'])}h")
                dk6.metric("Max Price",
                           f"{_mxpx['price_eur_mwh']:.1f} €/MWh",
                           help=f"at hour {int(_mxpx['Hour'])}h")
                dk7.metric("VRE Share",             f"{_dly_vre:.1f}%")
                dk8.metric("Hours PV > 50% Demand", f"{_h_pv50} h")

                # Prev / next day context
                _prev = sel_date - dt_timedelta(days=1)
                _next = sel_date + dt_timedelta(days=1)
                _dprev = df_merged[df_merged["Date"].dt.date == _prev]
                _dnext = df_merged[df_merged["Date"].dt.date == _next]
                _pprev = f"{_dprev['price_eur_mwh'].mean():.1f} €/MWh" \
                         if len(_dprev) else "N/A"
                _pnext = f"{_dnext['price_eur_mwh'].mean():.1f} €/MWh" \
                         if len(_dnext) else "N/A"
                st.caption(
                    f"📅 Context — Previous day ({_prev}): avg price **{_pprev}** · "
                    f"Next day ({_next}): avg price **{_pnext}**"
                )

                st.info(
                    "**What this shows:** The full 24-hour generation stack for a single day, "
                    "with the OMIE Day-Ahead price on the right axis. "
                    "**How to read:** Hours where the DA price dips (orange line) while PV "
                    "generation peaks reveal the intraday cannibalisation pattern. Negative "
                    "prices (below the red dashed line) indicate surplus generation that the "
                    "market cannot absorb — key hours for BESS dispatch. "
                    "**Limitation:** Prices for imputed days are derived from the synthetic "
                    "merit-order model, not actual OMIE settlements."
                )

        # ════════════════════════════════════════════════════════════════════
        # VIEW B — TYPICAL DAY  (5.1.3)
        # ════════════════════════════════════════════════════════════════════
        elif view_mode == "Typical Day":
            hrs24     = list(range(24))
            avg_tech  = df_m.groupby("Hour")[techs_nz].mean()
            avg_dem   = df_m.groupby("Hour")["Demand"].mean()
            avg_px    = df_m.groupby("Hour")["price_eur_mwh"].mean()
            min_dem   = df_m.groupby("Hour")["Demand"].min()
            max_dem   = df_m.groupby("Hour")["Demand"].max()

            fig_td = make_subplots(specs=[[{"secondary_y": True}]])

            # Min-max demand shaded band (drawn first so it is behind stacks)
            fig_td.add_trace(go.Scatter(
                x=hrs24 + hrs24[::-1],
                y=list(max_dem) + list(min_dem[::-1]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.06)",
                line=dict(width=0),
                name="Demand range (min–max)",
                hoverinfo="skip",
            ), secondary_y=False)

            for _t in techs_nz:
                if avg_tech[_t].sum() > 0:
                    fig_td.add_trace(go.Scatter(
                        x=hrs24, y=avg_tech[_t].tolist(),
                        mode="lines", stackgroup="one", name=_t,
                        line=dict(width=0), fillcolor=COLORS.get(_t, "#AAA"),
                        hovertemplate=f"<b>{_t}</b>: %{{y:.0f}} MW<extra></extra>",
                    ), secondary_y=False)

            if show_dem:
                fig_td.add_trace(go.Scatter(
                    x=hrs24, y=avg_dem.tolist(),
                    mode="lines", name="Avg Demand",
                    line=dict(color="black", width=2, dash="dot"),
                    hovertemplate="<b>Avg Demand</b>: %{y:.0f} MW<extra></extra>",
                ), secondary_y=False)

            fig_td.add_trace(go.Scatter(
                x=hrs24, y=avg_px.tolist(),
                mode="lines+markers", name="Avg DA Price",
                line=dict(color="darkorange", width=2.5),
                marker=dict(size=5),
                hovertemplate="<b>Avg DA Price</b>: %{y:.1f} €/MWh<extra></extra>",
            ), secondary_y=True)

            fig_td.update_layout(
                height=490, hovermode="x unified",
                title=f"Typical Day Profile — {sel_month} 2025 "
                      f"(avg of {df_m['Date'].dt.date.nunique()} days)",
                xaxis=dict(title="Hour", tickmode="linear", dtick=2),
                yaxis=dict(title="Generation (MW)", gridcolor="#eee"),
                yaxis2=dict(title="Avg DA Price (€/MWh)", showgrid=False),
                plot_bgcolor="white",
                legend=dict(orientation="h", y=-0.30, font=dict(size=10)),
            )
            st.plotly_chart(fig_td, use_container_width=True)

            # Month KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("VRE Share",   f"{(df_m['Solar PV'].sum()+df_m['Wind'].sum())/df_m['Demand'].sum()*100:.1f}%")
            k2.metric("Avg Demand",  f"{df_m['Demand'].mean():,.0f} MW")
            k3.metric("Avg PV Peak", f"{avg_tech['Solar PV'].max():,.0f} MW",
                      help=f"at hour {int(avg_tech['Solar PV'].idxmax())}h")
            k4.metric("Avg DA Price", f"{avg_px.mean():.1f} €/MWh")

            st.info(
                "**What this shows:** The mean generation profile for each technology across "
                f"all days of {sel_month}, with the grey band showing the full min–max range "
                "of hourly demand (day-to-day variability). "
                "**How to read:** The orange line (right axis) shows the average price shape "
                "for this month — look for the midday trough (solar cannibalisation window) "
                "vs the evening peak (highest-value BESS discharge window). "
                "**Limitation:** Averaging removes intraday variability; individual days can "
                "deviate significantly, especially in months with mixed weather."
            )

        # ════════════════════════════════════════════════════════════════════
        # VIEW C — TYPICAL WEEK / FULL MONTH with DA price overlay  (5.1.5)
        # ════════════════════════════════════════════════════════════════════
        else:  # "Typical Week" or "Full Month"
            if view_mode == "Typical Week":
                _days = sorted(df_m["Day"].unique())[:7]
                df_v  = df_m[df_m["Day"].isin(_days)].copy()
            else:
                df_v = df_m.copy()

            techs_v = [t for t in DISPATCH_TECHS
                       if t in df_v.columns and df_v[t].sum() > 0]

            fig_w = make_subplots(specs=[[{"secondary_y": True}]])

            # Imputed day shading
            if df_v["_imputed"].any():
                for _imp_day in sorted(df_v[df_v["_imputed"]]["Date"].dt.date.unique()):
                    fig_w.add_vrect(
                        x0=pd.Timestamp(_imp_day),
                        x1=pd.Timestamp(_imp_day) + pd.Timedelta(hours=23),
                        fillcolor="rgba(180,180,180,0.18)", line_width=0,
                        annotation_text="Imputed",
                        annotation_position="top left",
                        annotation_font=dict(size=8, color="gray"),
                    )

            for _t in techs_v:
                fig_w.add_trace(go.Scatter(
                    x=df_v["Date"].tolist(), y=df_v[_t].tolist(),
                    mode="lines", stackgroup="one", name=_t,
                    line=dict(width=0), fillcolor=COLORS.get(_t, "#AAA"),
                    hovertemplate=f"<b>{_t}</b>: %{{y:.0f}} MW<extra></extra>",
                ), secondary_y=False)

            if show_dem:
                fig_w.add_trace(go.Scatter(
                    x=df_v["Date"].tolist(), y=df_v["Demand"].tolist(),
                    mode="lines", name="Demand",
                    line=dict(color="black", width=2, dash="dot"),
                    hovertemplate="<b>Demand</b>: %{y:.0f} MW<extra></extra>",
                ), secondary_y=False)

            fig_w.add_trace(go.Scatter(
                x=df_v["Date"].tolist(), y=df_v["price_eur_mwh"].tolist(),
                mode="lines", name="DA Price",
                line=dict(color="darkorange", width=1.5, dash="dash"),
                opacity=0.85,
                hovertemplate="<b>DA Price</b>: %{y:.1f} €/MWh<extra></extra>",
            ), secondary_y=True)

            fig_w.add_hline(y=0, line_dash="dot", line_color="red",
                            line_width=1, opacity=0.35, secondary_y=True)

            fig_w.update_layout(
                height=500, hovermode="x unified",
                title=f"Hourly Dispatch + DA Price — {sel_month} 2025 ({view_mode})",
                xaxis_title="Date",
                yaxis=dict(title="Generation (MW)", gridcolor="#eee"),
                yaxis2=dict(title="DA Price (€/MWh)", showgrid=False),
                plot_bgcolor="white",
                legend=dict(orientation="h", y=-0.26, font=dict(size=11)),
            )
            st.plotly_chart(fig_w, use_container_width=True)

            if df_v["_imputed"].any():
                st.caption(
                    f"ℹ️ Grey bands = imputed days "
                    f"({int(df_v['_imputed'].sum())} hours filled with monthly averages)."
                )

            # Dedicated price panel
            with st.expander("📊 Price detail panel", expanded=False):
                _px_period = df_v["price_eur_mwh"]
                _wpx  = (df_v["price_eur_mwh"] * df_v["Demand"]).sum() / df_v["Demand"].sum()
                _pv_m = df_v["Solar PV"] > 0
                _capx = (
                    (df_v.loc[_pv_m, "price_eur_mwh"] * df_v.loc[_pv_m, "Solar PV"]).sum()
                    / df_v.loc[_pv_m, "Solar PV"].sum()
                ) if _pv_m.sum() > 0 else 0.0
                _h_neg = int((_px_period < 0).sum())
                _h_10  = int((_px_period < 10).sum())

                pp1, pp2, pp3, pp4 = st.columns(4)
                pp1.metric("Period Avg DA Price",     f"{_px_period.mean():.1f} €/MWh")
                pp2.metric("Demand-Weighted Price",   f"{_wpx:.1f} €/MWh")
                pp3.metric("PV Capture Price",        f"{_capx:.1f} €/MWh")
                pp4.metric("Hours < 0 €/MWh",        f"{_h_neg} h",
                           help=f"Hours < 10 €/MWh: {_h_10}")

                fig_px = go.Figure()
                fig_px.add_trace(go.Scatter(
                    x=df_v["Date"].tolist(), y=df_v["price_eur_mwh"].tolist(),
                    mode="lines", name="DA Price",
                    line=dict(color="darkorange", width=1.5),
                    fill="tozeroy", fillcolor="rgba(255,140,0,0.12)",
                ))
                fig_px.add_hline(y=0,  line_dash="dash", line_color="red",   opacity=0.5)
                fig_px.add_hline(y=25, line_dash="dot",  line_color="green",  opacity=0.6,
                                 annotation_text="PV LCOE ≈ 25 €/MWh",
                                 annotation_position="top right",
                                 annotation_font=dict(color="green", size=10))
                fig_px.update_layout(
                    height=220, plot_bgcolor="white",
                    title="DA Price (€/MWh) — hourly",
                    xaxis_title="Date", yaxis=dict(gridcolor="#eee"),
                    showlegend=False,
                )
                st.plotly_chart(fig_px, use_container_width=True)
                st.caption(
                    "⚙️ *Structured for future intraday (IA) data: add* "
                    "`omie_intraday_2025.csv` *with columns* "
                    "`datetime, session, price_eur_mwh, volume_mwh` *to overlay IA1–IA6 sessions.*"
                )

            # Month quick KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("VRE Share",   f"{(df_m['Solar PV'].sum()+df_m['Wind'].sum())/df_m['Demand'].sum()*100:.1f}%")
            k2.metric("Avg Demand",  f"{df_m['Demand'].mean():,.0f} MW")
            k3.metric("PV Peak",     f"{df_m['Solar PV'].max():,.0f} MW")
            k4.metric("Wind Peak",   f"{df_m['Wind'].max():,.0f} MW")

        # ════════════════════════════════════════════════════════════════════
        # SECTION 2 — SEASONAL TECHNOLOGY FOCUS  (5.1.4)
        # ════════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("🔍 Seasonal Technology Focus")

        _TECH_OPTS = [
            "Solar PV", "Wind", "Nuclear", "Hydro (on-flow)",
            "Combined Cycle GT", "Cogeneration", "Solar Thermal",
            "Pumping Turbine", "Demand", "VRE", "Residual Load",
        ]

        _TECH_INSIGHTS = {
            "Solar PV": (
                "**Solar PV** shows a dramatic seasonal swing: Summer profiles peak at ~18 GW "
                "around 13h with a tight P10–P90 band (very predictable clear-sky days), while "
                "Winter peaks at ~8–10 GW with wider variability. The generation window stretches "
                "from ~8h–18h in Summer vs ~9h–16h in Winter — a 2× seasonal amplitude. "
                "**Investment implication:** This directly drives cannibalisation. Summer produces "
                "more energy but at lower prices (midday glut); Winter produces less but at "
                "structurally higher prices. PPAs with seasonal shape clauses should account for "
                "this asymmetry."
            ),
            "Wind": (
                "**Wind** shows no strong diurnal pattern (relatively flat hour-by-hour) but is "
                "highly seasonal: Winter and Autumn averages run 30–50% above Summer. The wide "
                "P10–P90 bands across all seasons reflect wind's intermittency. "
                "**Investment implication:** Wind complements PV structurally — it fills the "
                "nighttime gap and peaks in the seasons when PV is weakest. PV+Wind hybrid "
                "projects capture this natural hedge, reducing hourly variance and improving "
                "blended capture prices."
            ),
            "Combined Cycle GT": (
                "**CCGT** is Spain's swing producer. In Winter it runs a classic dual-peak "
                "(morning + evening ramp). In Spring/Summer the morning peak largely disappears "
                "(displaced by solar), and CCGT concentrates on the 18h–22h evening window. "
                "Capacity factor drops from ~35% in Winter to <15% in Summer, signalling "
                "progressive renewable displacement. "
                "**Investment implication:** This pattern creates the core BESS opportunity — "
                "batteries serving the evening ramp (18h–22h) directly displace CCGT and capture "
                "the highest-price hours of the day."
            ),
            "Nuclear": (
                "**Nuclear** runs flat 24/7 at ~7 GW year-round — the archetypal baseload. "
                "Narrow P10–P90 bands confirm very low variability (scheduled outages aside). "
                "Nuclear does not respond to price signals or VRE levels. "
                "**Investment implication:** Nuclear sets the overnight price floor. Spain's "
                "nuclear fleet faces phased retirement from 2027; replacing this baseload "
                "with storage + demand response further strengthens the BESS investment case."
            ),
            "Hydro (on-flow)": (
                "**Run-of-river hydro** is relatively flat intraday but highly seasonal "
                "(Spring snowmelt peaks, Summer/Autumn drought lows). It is partially "
                "dispatchable but largely follows natural flow. "
                "**Investment implication:** Hydro provides some seasonal flexibility but "
                "cannot scale. Its Spring peak competes with PV for the same midday slot, "
                "deepening the cannibalisation trough — a dynamic that BESS can monetise."
            ),
            "Cogeneration": (
                "**Cogeneration** (industrial CHP) runs at a near-constant 5–6 GW baseload "
                "throughout the year with minimal seasonal or diurnal variation. It is "
                "economically driven and largely insensitive to market prices. "
                "**Investment implication:** As industrial electrification grows, cogeneration "
                "capacity may shift towards more demand-responsive profiles, creating additional "
                "intraday price volatility."
            ),
            "Solar Thermal": (
                "**Solar Thermal (CSP)** follows a solar profile but with storage capabilities "
                "that shift some generation into the early evening hours. The Summer profile "
                "shows a longer, flatter peak vs the sharp midday spike of PV. "
                "**Investment implication:** CSP with thermal storage is a direct analogue to "
                "PV+BESS — its evening extension demonstrates the value of shifting solar "
                "generation to post-sunset hours."
            ),
            "Pumping Turbine": (
                "**Pumped hydro (generation mode)** concentrates its output in the high-price "
                "evening hours (18h–22h) in all seasons, charging at low-price windows "
                "(historically night; increasingly midday in Summer as solar surplus grows). "
                "**Investment implication:** Pumped hydro is the closest existing analogue to "
                "BESS arbitrage. Its charging-window shift from overnight to midday directly "
                "mirrors how BESS would operate in a high-solar grid."
            ),
            "Demand": (
                "**Total demand** shows a classic double-peak pattern (morning + evening) in "
                "Winter/Autumn, flattening in Summer (air-conditioning extends the midday "
                "plateau). Overall demand is 15–20% higher in Winter than Summer. "
                "**Investment implication:** Residual load (Demand – VRE) — not gross demand "
                "— is the key metric for BESS dispatch optimisation. It determines the hours "
                "of highest marginal price and therefore maximum BESS discharge value."
            ),
            "VRE": (
                "**VRE (Solar PV + Wind)** is the aggregate renewable supply. In Summer, "
                "midday VRE can exceed demand (11h–15h surplus window); in Winter the profile "
                "is more balanced, with wind filling the nocturnal gap. The wide P10–P90 spread "
                "in Spring reflects the highest cannibalisation risk period. "
                "**Investment implication:** The structural surplus visible in Spring/Summer is "
                "the core driver of price cannibalisation and the economic case for flexible "
                "assets (BESS, interconnectors, demand response)."
            ),
            "Residual Load": (
                "**Residual Load (Demand – VRE)** is demand net of renewables — what "
                "dispatchable assets must cover. The duck-curve shape is most pronounced in "
                "Spring/Summer: high residual in the morning, near-zero or negative at midday "
                "(solar surplus), then a sharp evening ramp (18h–22h). "
                "**Investment implication:** The steepness of this evening ramp is the key "
                "dimensioning parameter for BESS. A 4h BESS charged at midday (cheap solar) "
                "and discharged during the ramp captures maximum price spread — this is the "
                "core merchant arbitrage opportunity."
            ),
        }

        _SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

        col_ts, col_blank = st.columns([2, 3])
        tech_sel = col_ts.selectbox(
            "Select technology", _TECH_OPTS, index=0, key="t1_tech_sea"
        )

        _hrs24 = list(range(24))

        # ── 4-Season overlay chart ────────────────────────────────────────────
        fig_sea = go.Figure()

        _sea_data = {}   # cache hourly stats per season
        for _sea in _SEASON_ORDER:
            _dfs   = df_merged[df_merged["Season"] == _sea]
            _vals  = _tech_vals(_dfs, tech_sel)
            _dfs   = _dfs.copy()
            _dfs["_v"] = _vals
            _mean  = _dfs.groupby("Hour")["_v"].mean()
            _p10   = _dfs.groupby("Hour")["_v"].quantile(0.10)
            _p90   = _dfs.groupby("Hour")["_v"].quantile(0.90)
            _sea_data[_sea] = {"mean": _mean, "p10": _p10, "p90": _p90,
                               "vals": _vals, "df": _dfs}

            _clr = SEASON_COLORS[_sea]

            # P10–P90 band (drawn before mean line)
            fig_sea.add_trace(go.Scatter(
                x=_hrs24 + _hrs24[::-1],
                y=list(_p90) + list(_p10[::-1]),
                fill="toself",
                fillcolor=_hex_rgba(_clr, 0.12),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name=f"{_sea} P10–P90",
            ))
            # Mean line
            fig_sea.add_trace(go.Scatter(
                x=_hrs24, y=list(_mean),
                mode="lines", name=_sea,
                line=dict(color=_clr, width=2.5),
                hovertemplate=f"<b>{_sea}</b>: %{{y:.0f}} MW<extra></extra>",
            ))

        # Annual average dashed reference line
        _all_vals  = _tech_vals(df_merged, tech_sel)
        _df_ann    = df_merged.copy()
        _df_ann["_v"] = _all_vals
        _ann_mean  = _df_ann.groupby("Hour")["_v"].mean()
        fig_sea.add_trace(go.Scatter(
            x=_hrs24, y=list(_ann_mean),
            mode="lines", name="Annual avg",
            line=dict(color="gray", width=1.5, dash="dash"),
            hovertemplate="<b>Annual avg</b>: %{y:.0f} MW<extra></extra>",
        ))

        fig_sea.update_layout(
            height=430, hovermode="x unified",
            title=f"Average Daily Profile — {tech_sel} — by Season (shaded bands = P10–P90)",
            xaxis=dict(title="Hour", tickmode="linear", dtick=2),
            yaxis=dict(title="MW (average)", gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_sea, use_container_width=True)

        # ── Summary metrics table ─────────────────────────────────────────────
        _tbl_rows = []
        for _sea in _SEASON_ORDER:
            _dfs  = df_merged[df_merged["Season"] == _sea]
            _vals = _tech_vals(_dfs, tech_sel)
            _ndays = _dfs["Date"].dt.date.nunique()
            _daily_gwh = _vals.sum() / max(_ndays, 1) / 1000.0
            _hmean = _sea_data[_sea]["mean"]
            _pk_h  = int(_hmean.idxmax())
            _mn_h  = int(_hmean.idxmin())
            _cf    = (f"{_vals.mean() / INSTALLED_CAP_MW[tech_sel] * 100:.1f}%"
                     if tech_sel in INSTALLED_CAP_MW else "—")
            _dsh   = _vals.sum() / _dfs["Demand"].sum() * 100
            _tbl_rows.append({
                "Season":               _sea,
                "Avg Daily (GWh/day)":  round(_daily_gwh, 1),
                "Peak Hour":            f"{_pk_h}h",
                "Peak (MW)":            round(_hmean.max(), 0),
                "Min Hour":             f"{_mn_h}h",
                "Min (MW)":             round(_hmean.min(), 0),
                "Capacity Factor":      _cf,
                "Demand Share (%)":     round(_dsh, 1),
            })
        _tbl = pd.DataFrame(_tbl_rows)
        st.dataframe(
            _tbl.style
                .background_gradient(subset=["Avg Daily (GWh/day)"], cmap="YlOrRd")
                .background_gradient(subset=["Demand Share (%)"],     cmap="Greens"),
            use_container_width=True, hide_index=True,
        )

        # ── 2×2 Small multiples ───────────────────────────────────────────────
        _sm_specs  = [[{"secondary_y": True}, {"secondary_y": True}],
                      [{"secondary_y": True}, {"secondary_y": True}]]
        _sm_titles = ["Winter", "Spring", "Summer", "Autumn"]
        fig_sm = make_subplots(
            rows=2, cols=2,
            subplot_titles=_sm_titles,
            specs=_sm_specs,
            horizontal_spacing=0.14,
            vertical_spacing=0.28,
        )

        _pos = [(1,1),(1,2),(2,1),(2,2)]
        for _i, _sea in enumerate(_SEASON_ORDER):
            _r, _c    = _pos[_i]
            _dfs      = df_merged[df_merged["Season"] == _sea]
            _hmean    = _sea_data[_sea]["mean"]
            _clr_sea  = SEASON_COLORS[_sea]
            _show_leg = _i == 0   # show legend only for first subplot

            # Primary: selected tech (filled area)
            fig_sm.add_trace(go.Scatter(
                x=_hrs24, y=list(_hmean),
                mode="lines", fill="tozeroy",
                fillcolor=_hex_rgba(_clr_sea, 0.30),
                line=dict(color=_clr_sea, width=1.5),
                name=tech_sel, showlegend=_show_leg,
                hovertemplate=f"%{{y:.0f}} MW<extra>{_sea} {tech_sel}</extra>",
            ), row=_r, col=_c, secondary_y=False)

            # Context overlays — tech-specific
            _avg_dem_s = _dfs.groupby("Hour")["Demand"].mean()
            _avg_px_s  = _dfs.groupby("Hour")["price_eur_mwh"].mean()
            _avg_vre_s = _dfs.groupby("Hour").apply(
                lambda g: (g["Solar PV"] + g["Wind"]).mean()
            )
            _avg_pv_s  = _dfs.groupby("Hour")["Solar PV"].mean()
            _avg_cg_s  = _dfs.groupby("Hour")["Combined Cycle GT"].mean()

            if tech_sel in ("Solar PV", "Solar Thermal", "Residual Load"):
                # Demand line + DA price secondary
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_dem_s),
                    mode="lines", name="Demand" if _show_leg else None,
                    line=dict(color="black", width=1.2, dash="dot"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>Demand</extra>",
                ), row=_r, col=_c, secondary_y=False)
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_px_s),
                    mode="lines", name="DA Price" if _show_leg else None,
                    line=dict(color="darkorange", width=1.2, dash="dash"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.1f} €/MWh<extra>DA Price</extra>",
                ), row=_r, col=_c, secondary_y=True)

            elif tech_sel == "Combined Cycle GT":
                # VRE dashed + Demand dotted
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_vre_s),
                    mode="lines", name="VRE" if _show_leg else None,
                    line=dict(color="#87CEEB", width=1.2, dash="dash"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>VRE</extra>",
                ), row=_r, col=_c, secondary_y=False)
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_dem_s),
                    mode="lines", name="Demand" if _show_leg else None,
                    line=dict(color="black", width=1.2, dash="dot"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>Demand</extra>",
                ), row=_r, col=_c, secondary_y=False)

            elif tech_sel == "Wind":
                # PV fill + Demand dotted
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_pv_s),
                    mode="lines", fill="tozeroy",
                    fillcolor="rgba(255,215,0,0.18)",
                    line=dict(color="#FFD700", width=1.0),
                    name="Solar PV" if _show_leg else None,
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>Solar PV</extra>",
                ), row=_r, col=_c, secondary_y=False)
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_dem_s),
                    mode="lines", name="Demand" if _show_leg else None,
                    line=dict(color="black", width=1.2, dash="dot"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>Demand</extra>",
                ), row=_r, col=_c, secondary_y=False)

            elif tech_sel == "Demand":
                # VRE dashed + CCGT lighter fill
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_vre_s),
                    mode="lines", name="VRE" if _show_leg else None,
                    line=dict(color="#87CEEB", width=1.2, dash="dash"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>VRE</extra>",
                ), row=_r, col=_c, secondary_y=False)
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_cg_s),
                    mode="lines", fill="tozeroy",
                    fillcolor="rgba(255,99,71,0.18)",
                    line=dict(color="#FF6347", width=1.0),
                    name="CCGT" if _show_leg else None,
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>CCGT</extra>",
                ), row=_r, col=_c, secondary_y=False)

            elif tech_sel == "VRE":
                # Demand dotted + DA price
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_dem_s),
                    mode="lines", name="Demand" if _show_leg else None,
                    line=dict(color="black", width=1.2, dash="dot"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>Demand</extra>",
                ), row=_r, col=_c, secondary_y=False)
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_px_s),
                    mode="lines", name="DA Price" if _show_leg else None,
                    line=dict(color="darkorange", width=1.2, dash="dash"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.1f} €/MWh<extra>DA Price</extra>",
                ), row=_r, col=_c, secondary_y=True)

            else:
                # Generic: just demand line
                fig_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_dem_s),
                    mode="lines", name="Demand" if _show_leg else None,
                    line=dict(color="black", width=1.2, dash="dot"),
                    showlegend=_show_leg,
                    hovertemplate="%{y:.0f} MW<extra>Demand</extra>",
                ), row=_r, col=_c, secondary_y=False)

        # ── Compute global axis ranges for harmonisation ─────────────────────
        _sm_primary_vals = []   # all MW values on primary y
        _sm_secondary_vals = [] # all EUR/MWh values on secondary y (when used)
        _has_secondary = tech_sel in ("Solar PV", "Solar Thermal", "Residual Load", "VRE")
        for _sea in _SEASON_ORDER:
            _dfs2 = df_merged[df_merged["Season"] == _sea]
            _sm_primary_vals += list(_sea_data[_sea]["mean"])
            _sm_primary_vals += list(_dfs2.groupby("Hour")["Demand"].mean())
            if tech_sel == "Wind":
                _sm_primary_vals += list(_dfs2.groupby("Hour")["Solar PV"].mean())
            if tech_sel in ("Combined Cycle GT", "Demand"):
                _sm_primary_vals += list(_dfs2.groupby("Hour").apply(
                    lambda g: (g["Solar PV"] + g["Wind"]).mean()))
            if tech_sel == "Demand":
                _sm_primary_vals += list(_dfs2.groupby("Hour")["Combined Cycle GT"].mean())
            if _has_secondary:
                _sm_secondary_vals += list(_dfs2.groupby("Hour")["price_eur_mwh"].mean())

        _sm_mw_min = min(_sm_primary_vals) * 0.0
        _sm_mw_max = max(_sm_primary_vals) * 1.10
        _sm_mw_range = [_sm_mw_min, _sm_mw_max]
        if _has_secondary and _sm_secondary_vals:
            _sm_px_min = min(_sm_secondary_vals)
            _sm_px_max = max(_sm_secondary_vals)
            _sm_px_pad = (_sm_px_max - _sm_px_min) * 0.12
            _sm_px_range = [_sm_px_min - _sm_px_pad, _sm_px_max + _sm_px_pad]

        # ── Apply axis titles, ranges, and grid styling ───────────────────────
        for _axis in fig_sm.layout:
            if _axis.startswith("xaxis"):
                fig_sm.layout[_axis].update(tickmode="linear", dtick=4, title_text="Hour")
            if _axis.startswith("yaxis"):
                _ax_num = int(_axis.replace("yaxis", "") or 1)
                if _ax_num % 2 == 1:  # primary axes (1, 3, 5, 7) — MW
                    fig_sm.layout[_axis].update(
                        title_text="MW",
                        range=_sm_mw_range,
                        gridcolor="#eee",
                        zeroline=True, zerolinewidth=1.0, zerolinecolor="#ccc",
                    )
                else:  # secondary axes (2, 4, 6, 8) — EUR/MWh (only when used)
                    if _has_secondary:
                        fig_sm.layout[_axis].update(
                            title_text="EUR/MWh",
                            range=_sm_px_range,
                            showgrid=False,
                        )

        fig_sm.update_layout(
            height=640,
            title=f"Seasonal Panels — {tech_sel} (with context overlays)",
            legend=dict(orientation="h", y=-0.10, font=dict(size=10)),
            plot_bgcolor="white",
            margin=dict(l=60, r=60, t=80, b=80),
        )
        # Push subplot titles up slightly so they don't clash with x-axis labels
        for ann in fig_sm.layout.annotations:
            ann.update(yshift=6, font=dict(size=12, color="#333"))

        st.plotly_chart(fig_sm, use_container_width=True)

        # Dynamic pedagogical explanation
        _insight = _TECH_INSIGHTS.get(
            tech_sel,
            f"**{tech_sel}** — select a technology to see a tailored investment insight."
        )
        st.info(_insight)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — MONTHLY & SEASONAL ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Monthly and Seasonal Analysis")

        # ── Date range selector (5.3.1) ───────────────────────────────────────
        st.markdown("#### 📅 Date Range Filter")
        _t2_c1, _t2_c2 = st.columns([2, 1])
        with _t2_c1:
            _t2_date_range = st.date_input(
                "Select date range",
                value=(dt_date(2025, 1, 1), dt_date(2025, 12, 31)),
                min_value=dt_date(2025, 1, 1),
                max_value=dt_date(2025, 12, 31),
                key="t2_date_range",
            )
        with _t2_c2:
            st.markdown("**Quick select:**")
            _qc1, _qc2, _qc3 = st.columns(3)
            if _qc1.button("Winter",   key="t2_winter"):
                st.session_state.t2_date_range = (dt_date(2025, 1, 1), dt_date(2025, 2, 28))
                st.rerun()
            if _qc2.button("Spring",   key="t2_spring"):
                st.session_state.t2_date_range = (dt_date(2025, 3, 1), dt_date(2025, 5, 31))
                st.rerun()
            if _qc3.button("Summer",   key="t2_summer"):
                st.session_state.t2_date_range = (dt_date(2025, 6, 1), dt_date(2025, 8, 31))
                st.rerun()
            _qc4, _qc5, _qc6 = st.columns(3)
            if _qc4.button("Autumn",   key="t2_autumn"):
                st.session_state.t2_date_range = (dt_date(2025, 9, 1), dt_date(2025, 11, 30))
                st.rerun()
            if _qc5.button("Full Year", key="t2_full"):
                st.session_state.t2_date_range = (dt_date(2025, 1, 1), dt_date(2025, 12, 31))
                st.rerun()

        # Apply date filter
        if isinstance(_t2_date_range, (list, tuple)) and len(_t2_date_range) == 2:
            _t2_start, _t2_end = _t2_date_range
        else:
            _t2_start, _t2_end = dt_date(2025, 1, 1), dt_date(2025, 12, 31)

        _t2_mask = (df["Date"].dt.date >= _t2_start) & (df["Date"].dt.date <= _t2_end)
        df_t2 = df[_t2_mask].copy()

        _n_days = (_t2_end - _t2_start).days + 1
        st.caption(f"Showing **{_n_days} days** from {_t2_start.strftime('%d %b')} to {_t2_end.strftime('%d %b %Y')}")
        st.divider()

        # ---- Monthly generation by technology
        techs_nz = [t for t in DISPATCH_TECHS if df[t].sum() > 0]
        monthly  = df_t2.groupby("Month")[techs_nz].sum() / 1e6   # TWh
        monthly.index = [MONTH_NAMES[i] for i in monthly.index]
        dem_mo   = df_t2.groupby("Month")["Demand"].sum() / 1e6
        dem_mo.index = monthly.index

        fig_mo = go.Figure()
        for t in techs_nz:
            fig_mo.add_trace(go.Bar(
                name=t, x=monthly.index, y=monthly[t],
                marker_color=COLORS.get(t, "#AAA")
            ))
        fig_mo.add_trace(go.Scatter(
            x=dem_mo.index, y=dem_mo.values,
            mode="lines+markers", name="Demand",
            line=dict(color="black", width=2)
        ))
        fig_mo.update_layout(
            barmode="stack", height=400,
            title="Monthly Generation by Technology (TWh)",
            yaxis_title="TWh", plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
        )
        st.plotly_chart(fig_mo, use_container_width=True)

        # ---- Average daily profile by season
        st.subheader("Average Daily Profile by Season")
        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        sel_sea = st.radio("Season", seasons, horizontal=True, key="t2_sea")

        df_s    = df_t2[df_t2["Season"] == sel_sea]
        if df_s.empty:
            df_s = df[df["Season"] == sel_sea]
        avg_day = df_s.groupby("Hour")[techs_nz].mean()
        avg_dem = df_s.groupby("Hour")["Demand"].mean()

        fig_day = go.Figure()
        for t in techs_nz:
            if avg_day[t].sum() > 0:
                fig_day.add_trace(go.Scatter(
                    x=avg_day.index, y=avg_day[t],
                    mode="lines", stackgroup="one", name=t,
                    line=dict(width=0), fillcolor=COLORS.get(t, "#AAA")
                ))
        fig_day.add_trace(go.Scatter(
            x=avg_dem.index, y=avg_dem.values,
            mode="lines", name="Demand",
            line=dict(color="black", width=2, dash="dot")
        ))
        fig_day.update_layout(
            height=420, hovermode="x unified",
            title=f"Average Daily Profile — {sel_sea}",
            xaxis_title="Hour", yaxis_title="MW (average)",
            xaxis=dict(tickmode="linear", dtick=2),
            plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        )
        st.plotly_chart(fig_day, use_container_width=True)

        # ---- Renewable share by season
        c_a, c_b = st.columns(2)
        seasonal = df_t2.groupby("Season").agg(
            Solar_PV = ("Solar PV",        "sum"),
            Wind     = ("Wind",             "sum"),
            Hydro    = ("Hydro (on-flow)",  "sum"),
            SolarTh  = ("Solar Thermal",    "sum"),
            OtherRen = ("Other Renewables", "sum"),
            Demand   = ("Demand",           "sum"),
        )
        seasonal["REN_pct"] = (
            seasonal[["Solar_PV","Wind","Hydro","SolarTh","OtherRen"]].sum(axis=1)
            / seasonal["Demand"] * 100
        )
        sea_order = ["Winter", "Spring", "Summer", "Autumn"]
        seasonal  = seasonal.reindex([s for s in sea_order if s in seasonal.index])

        with c_a:
            fig_s1 = px.bar(
                x=seasonal.index, y=seasonal["REN_pct"],
                color=seasonal.index,
                color_discrete_map=SEASON_COLORS,
                title="Renewable Share by Season (%)",
                labels={"x":"Season", "y":"REN Share (%)"},
                text_auto=".1f"
            )
            fig_s1.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            fig_s1.update_layout(showlegend=False, height=300,
                                  yaxis=dict(range=[0,100], gridcolor="#eee"),
                                  plot_bgcolor="white")
            st.plotly_chart(fig_s1, use_container_width=True)

        with c_b:
            fig_s2 = go.Figure()
            for col, name, clr in [("Solar_PV","☀️ Solar PV","#FFD700"),
                                    ("Wind",    "💨 Wind",    "#87CEEB"),
                                    ("Hydro",   "💧 Hydro",   "#4169E1")]:
                fig_s2.add_trace(go.Bar(
                    name=name, x=seasonal.index,
                    y=seasonal[col]/seasonal["Demand"]*100,
                    marker_color=clr
                ))
            fig_s2.update_layout(
                barmode="group", height=300,
                title="VRE Detail by Season (%)",
                yaxis_title="%", plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
                legend=dict(orientation="h", y=-0.35)
            )
            st.plotly_chart(fig_s2, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — MIX KPIs 2025
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Spanish Energy Mix — Key Indicators 2025")

        tot_dem   = df["Demand"].sum()     / 1e6
        tot_ren   = sum(df[t].sum() for t in RENEWABLES) / 1e6
        tot_vre   = (df["Solar PV"].sum() + df["Wind"].sum()) / 1e6
        h_vre70   = int((df["VRE_share"] > 0.70).sum())
        h_pv50    = int((df["Solar PV"] / df["Demand"] > 0.50).sum())
        h_surplus = int((df["Solar PV"] + df["Wind"] > df["Demand"]).sum())

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Total Demand",         f"{tot_dem:.0f} TWh")
        k2.metric("REN Share",            f"{tot_ren/tot_dem*100:.1f}%")
        k3.metric("PV + Wind Share",      f"{tot_vre/tot_dem*100:.1f}%")
        k4.metric("Hours VRE > 70%",      f"{h_vre70:,} h",
                  help="Cannibalisation proxy: hours where PV+Wind > 70% of demand")
        k5.metric("Hours PV > 50%",       f"{h_pv50:,} h",
                  help="Hours where solar PV alone exceeds 50% of demand")
        k6.metric("Hours PV+Wind > Demand", f"{h_surplus:,} h",
                  help="Hours where VRE alone exceeds total demand")

        st.divider()
        col1, col2 = st.columns([1.2, 1])

        with col1:
            # Technology pie chart
            tech_sums = {t: df[t].sum() for t in DISPATCH_TECHS if df[t].sum() > 0}
            fig_pie = px.pie(
                values=list(tech_sums.values()),
                names=list(tech_sums.keys()),
                title="Technology Mix Share 2025",
                color=list(tech_sums.keys()),
                color_discrete_map=COLORS,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                                   textfont_size=10)
            fig_pie.update_layout(height=460, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # PV share distribution
            pv_share_pct = df["Solar PV"] / df["Demand"] * 100
            bins   = [0, 10, 20, 30, 50, 70, 100]
            labels = ["<10%", "10-20%", "20-30%", "30-50%", "50-70%", ">70%"]
            binned = pd.cut(pv_share_pct, bins=bins, labels=labels)
            dist   = binned.value_counts().sort_index()
            clrs   = ["#87CEEB","#FFC107","#FF9800","#FF6347","#E53935","#B71C1C"]

            fig_dist = px.bar(
                x=dist.index.astype(str), y=dist.values,
                color=dist.index.astype(str),
                color_discrete_sequence=clrs,
                title="Hours by PV Share of Demand",
                labels={"x":"PV Share","y":"Hours"},
                text_auto=True
            )
            fig_dist.update_layout(showlegend=False, height=230,
                                    plot_bgcolor="white", yaxis=dict(gridcolor="#eee"))
            st.plotly_chart(fig_dist, use_container_width=True)

            # CCGT vs VRE share
            vre_bins = pd.cut(df["VRE_share"]*100,
                               bins=[0,25,50,75,100], labels=["0-25%","25-50%","50-75%",">75%"])
            ccgt_vre = df.groupby(vre_bins, observed=True)["Combined Cycle GT"].mean()

            fig_ccgt = px.bar(
                x=ccgt_vre.index.astype(str), y=ccgt_vre.values,
                title="Avg CCGT Dispatch (MW) by VRE Share",
                labels={"x":"VRE Share","y":"Avg CCGT (MW)"},
                color_discrete_sequence=["#FF6347"],
                text_auto=".0f"
            )
            fig_ccgt.update_layout(showlegend=False, height=210,
                                    plot_bgcolor="white", yaxis=dict(gridcolor="#eee"))
            st.plotly_chart(fig_ccgt, use_container_width=True)

        # ════════════════════════════════════════════════════════════════════
        # SECTION — DAYS WITH ZERO CCGT  (5.4.1)
        # ════════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("🔴 Days with Zero CCGT Dispatch")

        # Compute daily CCGT totals; a day with 0 CCGT = all 24 hours at 0 MW
        _ccgt_daily = df.groupby(df["Date"].dt.date)["Combined Cycle GT"].sum()
        _zero_ccgt_days = _ccgt_daily[_ccgt_daily == 0]
        _n_zero = len(_zero_ccgt_days)
        _pct_zero = _n_zero / 365 * 100

        _zc1, _zc2, _zc3 = st.columns(3)
        _zc1.metric("Days with 0 CCGT (all 24 h)", f"{_n_zero} days",
                    help="Calendar days where CCGT generation was 0 MW for all 24 hours")
        _zc2.metric("Share of year", f"{_pct_zero:.1f}%")
        _zc3.metric("Avg CCGT (non-zero days)", f"{_ccgt_daily[_ccgt_daily>0].mean()/1000:.1f} GWh/day")

        st.info(
            "**Investment insight:** Zero-CCGT days signal how close Spain is to fully displacing "
            "gas peakers during high-renewable periods. A rising trend of zero-CCGT days strengthens "
            "the case for BESS, which can serve peak-shaving without gas. "
            "These days occur predominantly on spring weekends with high PV output and low demand."
        )

        # Monthly breakdown
        _ccgt_day_df = _ccgt_daily.reset_index()
        _ccgt_day_df.columns = ["date", "ccgt_sum"]
        _ccgt_day_df["month"] = pd.to_datetime(_ccgt_day_df["date"]).dt.month
        _zero_per_month = _ccgt_day_df[_ccgt_day_df["ccgt_sum"] == 0].groupby("month").size()
        _zero_per_month = _zero_per_month.reindex(range(1, 13), fill_value=0)

        fig_zero_mo = go.Figure(go.Bar(
            x=[MONTH_NAMES[m] for m in _zero_per_month.index],
            y=_zero_per_month.values,
            marker_color=["#E53935" if v > 0 else "#90EE90" for v in _zero_per_month.values],
            text=_zero_per_month.values, textposition="outside",
        ))
        fig_zero_mo.update_layout(
            title="Zero-CCGT Days per Month (days where CCGT = 0 MW for all 24 h)",
            xaxis_title="Month", yaxis_title="Days",
            height=280, plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            showlegend=False,
        )
        st.plotly_chart(fig_zero_mo, use_container_width=True)

        # Day-of-year strip showing CCGT intensity
        _doy_df = _ccgt_daily.reset_index()
        _doy_df.columns = ["date", "ccgt_sum"]
        _doy_df["doy"] = pd.to_datetime(_doy_df["date"]).dt.dayofyear
        _doy_df["gwh"] = _doy_df["ccgt_sum"] / 1000

        fig_strip = go.Figure(go.Bar(
            x=_doy_df["doy"],
            y=_doy_df["gwh"],
            marker_color=[
                "#E53935" if v == 0 else
                "#FFC107" if v < 20 else
                "#FF6347" if v < 60 else "#888"
                for v in _doy_df["ccgt_sum"]
            ],
            hovertemplate="Day %{x} → %{y:.1f} GWh CCGT<extra></extra>",
        ))
        # Month labels
        _mo_doy = df.groupby("Month")["DayOfYear"].min()
        for _mid, _doy in _mo_doy.items():
            fig_strip.add_vline(x=_doy, line_width=0.5, line_color="gray", opacity=0.4)
            fig_strip.add_annotation(x=_doy+5, y=_doy_df["gwh"].max()*1.05,
                                     text=MONTH_NAMES[_mid], showarrow=False,
                                     font=dict(size=9, color="gray"))
        fig_strip.update_layout(
            title="Daily CCGT Generation — Full Year Strip (red = zero days)",
            xaxis_title="Day of Year", yaxis_title="GWh/day",
            height=250, plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            showlegend=False,
        )
        st.plotly_chart(fig_strip, use_container_width=True)

        if _n_zero > 0:
            _top_zero = _ccgt_day_df[_ccgt_day_df["ccgt_sum"] == 0][["date"]].head(15)
            _top_zero["date"] = pd.to_datetime(_top_zero["date"])
            _top_zero["Day"] = _top_zero["date"].dt.strftime("%A, %d %b")
            _top_zero["Month"] = _top_zero["date"].dt.month.map(MONTH_NAMES)
            st.caption(f"**Zero-CCGT days ({_n_zero} total):** " +
                       " · ".join(_top_zero["Day"].tolist()))

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — OMIE PRICES & PV CANNIBALISATION
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("💶 OMIE Day-Ahead Prices & Solar Cannibalisation")

        with st.expander("ℹ️ Methodology & Sources", expanded=False):
            st.markdown("""
            **Objective:** Quantify the correlation between instantaneous PV penetration
            and Day-Ahead price collapse on the Iberian market (OMIE).

            **Data:** OMIE 2025 hourly Day-Ahead prices cross-referenced with REE dispatch data.
            If `omie_prices_2025.csv` is present it is loaded directly.
            Otherwise, a calibrated synthetic price model is used (simplified merit-order
            with coefficients adjusted to Q1-Q3 2025 observations).

            **Key metrics:**
            - **PV Capture Price** = Σ(price_h × PV_output_h) / Σ(PV_output_h) —
              the price effectively captured by a PV producer with no storage
            - **Cannibalisation Factor** = PV Capture Price / average base price —
              measures the structural discount suffered by PV relative to the market average
            - **R² PV penetration vs Price** — statistical strength of the causal relationship
            """)

        # ── Load OMIE prices ──────────────────────────────────────────────────
        df_omie = load_or_generate_omie_prices(df)

        # Merge with dispatch data by date + hour
        df_omie["_date"] = df_omie["datetime"].dt.date.astype(str)
        df_omie["_hour"] = df_omie["hour"].astype(int)
        df_work = df.copy()
        df_work["_date"] = df_work["Date"].dt.date.astype(str)
        df_work["_hour"] = df_work["Hour"].astype(int)
        df_merged = df_work.merge(
            df_omie[["_date", "_hour", "price_eur_mwh"]],
            on=["_date", "_hour"], how="left"
        )
        # Fill any remaining NaNs with month/hour median
        if df_merged["price_eur_mwh"].isna().any():
            median_fill = df_merged.groupby(["Month", "Hour"])["price_eur_mwh"].transform("median")
            df_merged["price_eur_mwh"] = df_merged["price_eur_mwh"].fillna(median_fill)

        df_merged["PV_penetration"]  = df_merged["Solar PV"] / df_merged["Demand"]
        df_merged["VRE_penetration"] = (df_merged["Solar PV"] + df_merged["Wind"]) / df_merged["Demand"]

        # ────────────────────────────────────────────────────────────────────
        # SECTION 1 — KEY MARKET KPIs
        # ────────────────────────────────────────────────────────────────────
        st.markdown("### 📌 2025 Market Indicators")

        avg_price = df_merged["price_eur_mwh"].mean()

        pv_mask = df_merged["Solar PV"] > 0
        if pv_mask.sum() > 0:
            capture_price_pv = (
                (df_merged.loc[pv_mask, "price_eur_mwh"] * df_merged.loc[pv_mask, "Solar PV"]).sum()
                / df_merged.loc[pv_mask, "Solar PV"].sum()
            )
        else:
            capture_price_pv = avg_price

        cannib_factor = capture_price_pv / avg_price * 100
        h_below_20    = int((df_merged["price_eur_mwh"] < 20).sum())
        h_below_10    = int((df_merged["price_eur_mwh"] < 10).sum())
        h_below_0     = int((df_merged["price_eur_mwh"] < 0).sum())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Avg DA Price",       f"{avg_price:.1f} €/MWh")
        k2.metric("PV Capture Price",   f"{capture_price_pv:.1f} €/MWh",
                  delta=f"{capture_price_pv - avg_price:.1f} €", delta_color="inverse")
        k3.metric("Cannibalisation Factor", f"{cannib_factor:.1f}%",
                  help="PV Capture Price / avg market price. <100% means PV earns below average.")
        k4.metric("Hours < 10 €/MWh",  f"{h_below_10:,} h",
                  help=f"of which {h_below_0:,} h at negative prices")
        k5.metric("Hours < 0 €/MWh",   f"{h_below_0:,} h")

        st.divider()

        # ────────────────────────────────────────────────────────────────────
        # SECTION 2 — PV PENETRATION vs DA PRICE SCATTER
        # ────────────────────────────────────────────────────────────────────
        st.markdown("### 📉 PV Penetration → Day-Ahead Price Correlation")

        fc1, fc2, fc3 = st.columns(3)
        sel_months_s2 = fc1.multiselect(
            "Filter by month", list(MONTH_NAMES.values()),
            default=list(MONTH_NAMES.values()), key="t6_months"
        )
        hour_range = fc2.slider("Hour range", 0, 23, (8, 18), key="t6_hours")
        color_by   = fc3.radio("Colour by", ["Season", "Month", "Hour"],
                                horizontal=True, key="t6_color")

        # Filtering
        sel_m_nums = [month_num[m] for m in sel_months_s2]
        mask_s2 = (
            df_merged["Month"].isin(sel_m_nums) &
            (df_merged["Hour"] >= hour_range[0]) &
            (df_merged["Hour"] <= hour_range[1]) &
            (df_merged["Solar PV"] > 10)   # exclude nighttime
        )
        df_s2 = df_merged[mask_s2].copy()

        if len(df_s2) > 0:
            color_col = ("Season"    if color_by == "Season"
                         else "MonthName" if color_by == "Month"
                         else "Hour")

            fig_scatter = px.scatter(
                df_s2,
                x=df_s2["PV_penetration"] * 100,
                y="price_eur_mwh",
                color=color_col,
                color_discrete_map=SEASON_COLORS if color_by == "Season" else None,
                size="Solar PV",
                size_max=8,
                opacity=0.35,
                labels={
                    "x":             "Instantaneous PV Penetration (%)",
                    "price_eur_mwh": "Day-Ahead Price (€/MWh)",
                    "Solar PV":      "PV Output (MW)",
                },
                title="PV Penetration (%) vs Day-Ahead Price (€/MWh)",
            )

            # Polynomial regression (degree 2)
            x_vals = df_s2["PV_penetration"].values * 100
            y_vals = df_s2["price_eur_mwh"].values
            coeffs = np.polyfit(x_vals, y_vals, 2)
            poly_fn = np.poly1d(coeffs)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 200)

            y_pred  = poly_fn(x_vals)
            ss_res  = np.sum((y_vals - y_pred) ** 2)
            ss_tot  = np.sum((y_vals - np.mean(y_vals)) ** 2)
            r2_global = 1 - ss_res / ss_tot

            r_pearson, p_value = scipy_stats.pearsonr(x_vals, y_vals)

            fig_scatter.add_trace(go.Scatter(
                x=x_range, y=poly_fn(x_range),
                mode="lines", name=f"Polynomial regression (R²={r2_global:.3f})",
                line=dict(color="red", width=3, dash="dash"),
            ))

            # Unsubsidised PV LCOE threshold
            fig_scatter.add_hline(
                y=25, line_dash="dot", line_color="orange", line_width=2,
                annotation_text="Unsubsidised PV LCOE ≈ 25 €/MWh",
                annotation_position="top right",
                annotation_font=dict(color="orange", size=11),
            )

            fig_scatter.update_layout(
                height=520, plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee", title="DA Price (€/MWh)"),
                xaxis=dict(gridcolor="#eee", title="PV Penetration (%)"),
                legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Statistical summary
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("R² (degree-2 poly.)", f"{r2_global:.3f}")
            sc2.metric("Pearson r",           f"{r_pearson:.3f}")
            sc3.metric("p-value",             f"{p_value:.2e}")
            sc4.metric("N observations",      f"{len(df_s2):,}")

            st.info(
                f"**How to read:** Each additional percentage point of PV penetration is "
                f"associated with a price decline of ~{abs(coeffs[1]):.1f} €/MWh (linear term) "
                f"with a quadratic acceleration of {abs(coeffs[0]):.2f} €/MWh/pp². "
                f"The correlation is "
                f"{'very strong' if abs(r_pearson) > 0.7 else 'strong' if abs(r_pearson) > 0.5 else 'moderate'} "
                f"(r = {r_pearson:.3f}, p < {'0.001' if p_value < 0.001 else f'{p_value:.3f}'})."
            )
        else:
            st.warning("No data for the selected filters.")

        st.divider()

        # ────────────────────────────────────────────────────────────────────
        # SECTION 3 — PRICE × PV PENETRATION HEATMAPS (HOUR × MONTH)
        # ────────────────────────────────────────────────────────────────────
        st.markdown("### 🌡️ Structural Cannibalisation Zones (Hour × Month)")

        col_hm1, col_hm2 = st.columns(2)

        with col_hm1:
            pivot_price = df_merged.pivot_table(
                values="price_eur_mwh", index="Hour", columns="Month", aggfunc="mean"
            )
            pivot_price.columns = [MONTH_NAMES[c] for c in pivot_price.columns]

            fig_hm_price = go.Figure(go.Heatmap(
                z=pivot_price.values,
                x=pivot_price.columns, y=pivot_price.index,
                colorscale="RdYlGn", hoverongaps=False,
                colorbar=dict(title="€/MWh"),
                hovertemplate="%{x} · %{y}h → %{z:.1f} €/MWh<extra></extra>",
            ))
            fig_hm_price.update_layout(
                title="Avg DA Price (€/MWh)",
                xaxis_title="Month", yaxis_title="Hour",
                yaxis=dict(tickmode="linear", dtick=2),
                height=400,
            )
            st.plotly_chart(fig_hm_price, use_container_width=True)

        with col_hm2:
            pivot_pvpen = df_merged.pivot_table(
                values="PV_penetration", index="Hour", columns="Month", aggfunc="mean"
            )
            pivot_pvpen.columns = [MONTH_NAMES[c] for c in pivot_pvpen.columns]

            fig_hm_pv = go.Figure(go.Heatmap(
                z=pivot_pvpen.values * 100,
                x=pivot_pvpen.columns, y=pivot_pvpen.index,
                colorscale="YlOrRd", hoverongaps=False,
                colorbar=dict(title="% of demand"),
                hovertemplate="%{x} · %{y}h → %{z:.1f}% PV<extra></extra>",
            ))
            fig_hm_pv.update_layout(
                title="Avg PV Penetration (%)",
                xaxis_title="Month", yaxis_title="Hour",
                yaxis=dict(tickmode="linear", dtick=2),
                height=400,
            )
            st.plotly_chart(fig_hm_pv, use_container_width=True)

        st.info(
            "**How to read:** Red zones (right map) of high PV penetration correspond to "
            "green/low zones (left map) of low prices. The structural trough sits between "
            "10h and 16h from March to August — the core of solar cannibalisation. "
            "This is the window where merchant PV revenue is most at risk."
        )

        st.divider()

        # ────────────────────────────────────────────────────────────────────
        # SECTION 4 — PRICE DISTRIBUTION BY PV PENETRATION QUINTILE
        # ────────────────────────────────────────────────────────────────────
        st.markdown("### 📊 Price Distribution by PV Penetration Quintile")

        df_pv = df_merged[df_merged["Solar PV"] > 10].copy()

        if len(df_pv) > 100:
            df_pv["PV_quintile"] = pd.qcut(
                df_pv["PV_penetration"], q=5,
                labels=["Q1 (Low PV)", "Q2", "Q3", "Q4", "Q5 (Very High PV)"]
            )

            quintile_colors = {
                "Q1 (Low PV)":      "#4169E1",
                "Q2":               "#87CEEB",
                "Q3":               "#FFC107",
                "Q4":               "#FF9800",
                "Q5 (Very High PV)":"#E53935",
            }

            fig_box = px.box(
                df_pv, x="PV_quintile", y="price_eur_mwh",
                color="PV_quintile",
                color_discrete_map=quintile_colors,
                title="Day-Ahead Price Distribution by PV Penetration Quintile",
                labels={
                    "PV_quintile":   "PV Penetration Quintile",
                    "price_eur_mwh": "Day-Ahead Price (€/MWh)",
                },
            )
            fig_box.add_hline(
                y=25, line_dash="dot", line_color="orange", line_width=2,
                annotation_text="Unsubsidised PV LCOE ≈ 25 €/MWh",
                annotation_position="top right",
                annotation_font=dict(color="orange", size=10),
            )
            fig_box.add_hline(y=0, line_dash="solid", line_color="red", line_width=1, opacity=0.5)
            fig_box.update_layout(
                height=450, plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Stats table by quintile
            q_stats = df_pv.groupby("PV_quintile", observed=True).agg(
                PV_pen_min  =("PV_penetration", lambda x: f"{x.min()*100:.1f}%"),
                PV_pen_max  =("PV_penetration", lambda x: f"{x.max()*100:.1f}%"),
                Price_median=("price_eur_mwh", "median"),
                Price_mean  =("price_eur_mwh", "mean"),
                Price_P10   =("price_eur_mwh", lambda x: x.quantile(0.10)),
                Hours_neg   =("price_eur_mwh", lambda x: int((x < 0).sum())),
                Hours       =("price_eur_mwh", "count"),
            ).reset_index()
            q_stats.columns = [
                "Quintile", "PV pen. min", "PV pen. max",
                "Median Price (€)", "Avg Price (€)", "P10 Price (€)",
                "Hours < 0€", "# Hours"
            ]
            st.dataframe(
                q_stats.style.format({
                    "Median Price (€)": "{:.1f}",
                    "Avg Price (€)":    "{:.1f}",
                    "P10 Price (€)":    "{:.1f}",
                }),
                use_container_width=True, hide_index=True,
            )

            st.info(
                "**How to read:** The relationship is monotonic — each higher PV quintile "
                "corresponds to a lower median price. At Q5, the median price is "
                f"**{q_stats.iloc[-1]['Median Price (€)']:.0f} €/MWh** vs "
                f"**{q_stats.iloc[0]['Median Price (€)']:.0f} €/MWh** at Q1 — "
                f"a cannibalisation factor of "
                f"**{q_stats.iloc[-1]['Median Price (€)']/max(q_stats.iloc[0]['Median Price (€)'], 0.1)*100:.0f}%**."
            )
        else:
            st.warning("Not enough PV data for quintile analysis.")

        st.divider()

        # ────────────────────────────────────────────────────────────────────
        # SECTION 5 — FORWARD SIMULATION: CAPTURE PRICE vs PV CAPACITY
        # ────────────────────────────────────────────────────────────────────
        st.markdown("### 🔮 Forward Simulation — Capture Price vs Additional PV Capacity")

        with st.expander("ℹ️ Simulation methodology", expanded=False):
            st.markdown("""
            **Principle:** PV hourly output from 2025 is scaled proportionally to the
            additional capacity, then the price = f(PV_penetration) regression estimated
            in Section 2 is applied to recalculate simulated hourly prices and the
            resulting Capture Price.

            **Assumptions:**
            - Constant PV profile shape (same hours, same shape) — only amplitude changes.
            - The price / PV-penetration relationship is extrapolable (conservative: likely
              underestimates the price drop at very high penetrations).
            - Constant demand (no demand growth integrated here).
            """)

        sim_max_gw = st.slider(
            "Maximum additional PV capacity (GW)", 5, 30, 20, 1, key="t6_sim_max"
        )

        df_solar = df_merged[df_merged["Solar PV"] > 10].copy()
        if len(df_solar) > 100:
            x_base     = df_solar["PV_penetration"].values * 100
            y_base     = df_solar["price_eur_mwh"].values
            sim_coeffs = np.polyfit(x_base, y_base, 2)
            sim_poly   = np.poly1d(sim_coeffs)

            gw_steps     = np.arange(0, sim_max_gw + 0.5, 0.5)
            capture_prices = []
            avg_prices     = []
            h_negative     = []
            h_below_lcoe   = []

            for delta_gw in gw_steps:
                scale       = 1 + delta_gw / BASE_SOLAR_GW
                sim_pv      = df_merged["Solar PV"] * scale
                sim_pv_pen  = sim_pv / df_merged["Demand"] * 100

                sim_price   = df_merged["price_eur_mwh"].copy()
                solar_mask  = df_merged["Solar PV"] > 10
                sim_price.loc[solar_mask] = sim_poly(sim_pv_pen[solar_mask].values)

                pv_gen = sim_pv[solar_mask]
                cp     = (sim_price[solar_mask] * pv_gen).sum() / pv_gen.sum()
                capture_prices.append(cp)
                avg_prices.append(sim_price.mean())
                h_negative.append(int((sim_price < 0).sum()))
                h_below_lcoe.append(int((sim_price[solar_mask] < 25).sum()))

            total_gw = BASE_SOLAR_GW + gw_steps

            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(
                x=total_gw, y=capture_prices,
                mode="lines+markers", name="PV Capture Price",
                line=dict(color="#FFD700", width=3),
                marker=dict(size=4),
                hovertemplate="PV: %{x:.0f} GW → Capture: %{y:.1f} €/MWh<extra></extra>",
            ))
            fig_sim.add_trace(go.Scatter(
                x=total_gw, y=avg_prices,
                mode="lines", name="Avg DA Price",
                line=dict(color="#4169E1", width=2, dash="dash"),
                hovertemplate="PV: %{x:.0f} GW → Avg price: %{y:.1f} €/MWh<extra></extra>",
            ))

            # LCOE threshold
            fig_sim.add_hline(
                y=25, line_dash="dot", line_color="red", line_width=2,
                annotation_text="Unsubsidised PV LCOE ≈ 25 €/MWh",
                annotation_position="bottom right",
                annotation_font=dict(color="red", size=11),
            )

            # Tipping-point identification
            cp_arr     = np.array(capture_prices)
            below_lcoe = np.where(cp_arr < 25)[0]
            if len(below_lcoe) > 0:
                bascule_idx = below_lcoe[0]
                bascule_gw  = total_gw[bascule_idx]
                fig_sim.add_vline(
                    x=bascule_gw, line_dash="dot", line_color="red", line_width=1.5,
                    annotation_text=f"Tipping point: {bascule_gw:.0f} GW",
                    annotation_position="top left",
                    annotation_font=dict(color="red", size=11),
                )

            fig_sim.update_layout(
                title="PV Capture Price Trajectory vs Installed Capacity",
                xaxis_title="Total Installed PV Capacity (GW)",
                yaxis_title="€/MWh",
                height=450, plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Secondary charts: negative hours + cannibalisation ratio
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                fig_neg = go.Figure(go.Bar(
                    x=total_gw, y=h_negative,
                    marker_color=["#E53935" if h > 500 else "#FFC107" if h > 200 else "#4CAF50"
                                  for h in h_negative],
                    hovertemplate="PV: %{x:.0f} GW → %{y} negative hours<extra></extra>",
                ))
                fig_neg.update_layout(
                    title="Hours at Negative Price (h/yr)",
                    xaxis_title="PV Capacity (GW)", yaxis_title="Hours",
                    height=300, plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
                )
                st.plotly_chart(fig_neg, use_container_width=True)

            with col_s2:
                cannib_ratio = [cp / ap * 100 if ap > 0 else 0
                                for cp, ap in zip(capture_prices, avg_prices)]
                fig_cannib = go.Figure(go.Scatter(
                    x=total_gw, y=cannib_ratio,
                    mode="lines+markers",
                    line=dict(color="#9370DB", width=2.5),
                    marker=dict(size=4),
                    hovertemplate="PV: %{x:.0f} GW → Cannib: %{y:.1f}%<extra></extra>",
                ))
                fig_cannib.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
                fig_cannib.update_layout(
                    title="Cannibalisation Factor (%)",
                    xaxis_title="PV Capacity (GW)",
                    yaxis_title="Capture Price / Base Price (%)",
                    height=300, plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
                )
                st.plotly_chart(fig_cannib, use_container_width=True)

            # Investment insight box
            if len(below_lcoe) > 0:
                st.error(
                    f"**Tipping Point:** The PV Capture Price falls below the unsubsidised "
                    f"LCOE (25 €/MWh) from **{bascule_gw:.0f} GW installed** "
                    f"(+{bascule_gw - BASE_SOLAR_GW:.0f} GW vs 2025 baseline). "
                    f"Beyond this threshold, a standalone PV project (no storage) is no longer "
                    f"economically viable on a pure merchant basis."
                )
            else:
                st.success(
                    "The PV Capture Price remains above the unsubsidised LCOE (25 €/MWh) "
                    "across the entire simulated range."
                )

            st.info(
                "**Investment implication:** This simulation supports the PV+BESS hybridisation "
                "thesis as a necessary condition for merchant viability. Storage shifts sales "
                "to evening peak hours (18h-21h) where prices remain structurally elevated, "
                "restoring an effective Capture Price above the profitability threshold."
            )
        else:
            st.warning("Not enough data for the forward simulation.")

        # ────────────────────────────────────────────────────────────────────
        # SECTION 6 — CAPTURE PRICE PV BY SEASON  (5.5.1)
        # ────────────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🌤️ PV Capture Price by Season")

        _sea_order_pv = ["Winter", "Spring", "Summer", "Autumn"]
        _cpv_rows = []
        for _sea in _sea_order_pv:
            _ds = df_merged[df_merged["Season"] == _sea]
            _pvm = _ds["Solar PV"] > 0
            _avg_px = _ds["price_eur_mwh"].mean()
            _cap_px = (
                (_ds.loc[_pvm, "price_eur_mwh"] * _ds.loc[_pvm, "Solar PV"]).sum()
                / _ds.loc[_pvm, "Solar PV"].sum()
            ) if _pvm.sum() > 0 else _avg_px
            _cannib = _cap_px / _avg_px * 100 if _avg_px > 0 else 100.0
            _h_neg  = int((_ds["price_eur_mwh"] < 0).sum())
            _h_10   = int((_ds["price_eur_mwh"] < 10).sum())
            _pv_twh = _ds["Solar PV"].sum() / 1e6
            _cpv_rows.append({
                "Season":             _sea,
                "_avg_px":            _avg_px,
                "_cap_px":            _cap_px,
                "_cannib":            _cannib,
                "Hours < 10 €":       _h_10,
                "Hours < 0 €":        _h_neg,
                "PV Generation (TWh)": round(_pv_twh, 1),
            })

        _cpv_df = pd.DataFrame(_cpv_rows)

        # Grouped bar: Avg DA Price vs Capture Price by season
        fig_cpv = go.Figure()
        fig_cpv.add_trace(go.Bar(
            x=_cpv_df["Season"], y=_cpv_df["_avg_px"].round(1),
            name="Avg DA Price", marker_color="#4169E1",
            text=_cpv_df["_avg_px"].round(1), textposition="outside",
        ))
        fig_cpv.add_trace(go.Bar(
            x=_cpv_df["Season"], y=_cpv_df["_cap_px"].round(1),
            name="PV Capture Price", marker_color="#FFD700",
            text=_cpv_df["_cap_px"].round(1), textposition="outside",
        ))
        fig_cpv.add_hline(y=25, line_dash="dot", line_color="red", line_width=2,
                          annotation_text="PV LCOE ≈ 25 €/MWh",
                          annotation_position="top right",
                          annotation_font=dict(color="red", size=10))
        # Cannibalisation factor annotations
        for _, _row in _cpv_df.iterrows():
            fig_cpv.add_annotation(
                x=_row["Season"], y=min(_row["_cap_px"], _row["_avg_px"]) - 4,
                text=f"Cannib: {_row['_cannib']:.0f}%",
                showarrow=False, font=dict(size=9, color="gray"),
            )
        fig_cpv.update_layout(
            barmode="group", height=420,
            title="PV Capture Price vs Average DA Price by Season",
            yaxis=dict(title="€/MWh", gridcolor="#eee"),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_cpv, use_container_width=True)

        # Summary table
        _tbl_cpv = _cpv_df[["Season", "_avg_px", "_cap_px", "_cannib",
                              "Hours < 10 €", "Hours < 0 €", "PV Generation (TWh)"]].copy()
        _tbl_cpv.columns = ["Season", "Avg DA Price (€/MWh)", "Capture Price PV (€/MWh)",
                             "Cannibalisation (%)", "Hours < 10 €/MWh",
                             "Hours < 0 €/MWh", "PV Generation (TWh)"]
        _tbl_cpv["Avg DA Price (€/MWh)"]      = _tbl_cpv["Avg DA Price (€/MWh)"].round(1)
        _tbl_cpv["Capture Price PV (€/MWh)"]  = _tbl_cpv["Capture Price PV (€/MWh)"].round(1)
        _tbl_cpv["Cannibalisation (%)"]        = _tbl_cpv["Cannibalisation (%)"].round(1)
        st.dataframe(
            _tbl_cpv.style
                .background_gradient(subset=["Cannibalisation (%)"], cmap="RdYlGn", vmin=60, vmax=100)
                .background_gradient(subset=["Hours < 0 €/MWh"],     cmap="OrRd"),
            use_container_width=True, hide_index=True,
        )
        st.info(
            "**Key insight:** Spring typically has the worst cannibalisation — high PV "
            "output collides with low demand, pushing prices and the capture ratio to annual "
            "lows. Winter typically has the best capture ratio because PV generates less but "
            "at higher prices (no midday glut). **PPA implication:** Seasonal shape clauses "
            "should reflect this asymmetry — paying a premium for summer production is "
            "structurally misaligned with capture price dynamics."
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — HEATMAPS  (unchanged positioning)
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("🌡️ Heatmaps — Hourly Intensity by Technology")

        tech_opts = ["Solar PV", "Wind", "Demand", "Combined Cycle GT",
                     "Nuclear", "Hydro (on-flow)", "VRE_share"]
        tech_hm   = st.selectbox("Technology / Variable", tech_opts, index=0, key="hm_tech")
        cscale_map = {
            "Solar PV":"YlOrRd", "Wind":"Blues", "Demand":"Viridis",
            "Combined Cycle GT":"Oranges", "Nuclear":"Purples",
            "Hydro (on-flow)":"BuPu", "VRE_share":"RdYlGn",
        }
        cscale = cscale_map.get(tech_hm, "Plasma")
        label  = "VRE Share (%)" if tech_hm == "VRE_share" else "MW"
        col_hm = tech_hm

        # Heatmap Hour × Day-of-year
        pivot_yr = df.pivot_table(values=col_hm, index="Hour", columns="DayOfYear", aggfunc="mean")
        fig_hm1 = go.Figure(go.Heatmap(
            z=pivot_yr.values * (100 if tech_hm == "VRE_share" else 1),
            x=pivot_yr.columns, y=pivot_yr.index,
            colorscale=cscale, hoverongaps=False,
            colorbar=dict(title=label),
            hovertemplate="Day %{x} · %{y}h → %{z:.1f}<extra></extra>",
        ))
        mo_starts_hm = df.groupby("Month")["DayOfYear"].min().sort_index()
        for m_id, doy in mo_starts_hm.items():
            fig_hm1.add_vline(x=doy, line_width=0.8, line_color="white", opacity=0.6)
            fig_hm1.add_annotation(x=doy+7, y=23.5, text=MONTH_NAMES[m_id],
                                   showarrow=False, font=dict(color="white", size=8))
        fig_hm1.update_layout(
            title=f"Heatmap {tech_hm} — Hour × Day of Year 2025",
            xaxis_title="Day of Year", yaxis_title="Hour",
            yaxis=dict(tickmode="linear", dtick=2), height=430,
        )
        st.plotly_chart(fig_hm1, use_container_width=True)

        # Heatmap Hour × Month
        pivot_mo_hm = df.pivot_table(values=col_hm, index="Hour", columns="Month", aggfunc="mean")
        pivot_mo_hm.columns = [MONTH_NAMES[c] for c in pivot_mo_hm.columns]
        fig_hm2 = go.Figure(go.Heatmap(
            z=pivot_mo_hm.values * (100 if tech_hm == "VRE_share" else 1),
            x=pivot_mo_hm.columns, y=pivot_mo_hm.index,
            colorscale=cscale, hoverongaps=False,
            colorbar=dict(title=label),
            hovertemplate="%{x} · %{y}h → %{z:.1f}<extra></extra>",
        ))
        fig_hm2.update_layout(
            title=f"Average Profile {tech_hm} — Hour × Month",
            xaxis_title="Month", yaxis_title="Hour",
            yaxis=dict(tickmode="linear", dtick=2), height=370,
        )
        st.plotly_chart(fig_hm2, use_container_width=True)

        st.info(
            "**How to read:** The top heatmap shows the full year (each column = 1 day, each row = 1 hour). "
            "The bottom heatmap compresses to monthly averages. Look for the bright yellow/orange diagonal "
            "band in Solar PV (morning-to-evening arc shifting with seasons) and the deep blue trough in "
            "CCGT during spring/summer midday — the structural consequence of solar cannibalisation."
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 — FLEXIBILITY ASSETS (CCGT, PUMPING, INTERCONNECTORS)
    # ════════════════════════════════════════════════════════════════════════
    with tab6:
        st.subheader("🏭 Flexibility Assets — CCGT, Pumped Hydro & Interconnectors")
        st.markdown(
            "This tab analyses Spain's **existing flexibility stack** — the assets that absorb VRE "
            "surplus and fill the evening ramp. Understanding their behaviour patterns reveals where "
            "BESS adds value and where it competes with existing providers."
        )

        # ── Sub-section selector ──────────────────────────────────────────────
        _flex_section = st.radio(
            "Section",
            ["A · Overview", "B · CCGT Deep Dive", "C · Pumped Hydro", "D · Interconnectors"],
            horizontal=True, key="flex_sec",
        )

        # ── Shared pre-computations ───────────────────────────────────────────
        _hrs24 = list(range(24))
        _seas4 = ["Winter", "Spring", "Summer", "Autumn"]

        # Interconnector columns available in the dataset
        _interconn_cols = ["France Balance", "Portugal Balance", "Morocco Balance",
                           "Balearic Interconnection"]
        _interconn_cols = [c for c in _interconn_cols if c in df_merged.columns]

        # ════════════════════════════════════════════════════════════════════
        # A — OVERVIEW
        # ════════════════════════════════════════════════════════════════════
        if _flex_section == "A · Overview":
            st.markdown("### Overview — Flexibility Stack (Daily Averages)")

            # Headline KPIs
            _fl1, _fl2, _fl3, _fl4 = st.columns(4)
            _ccgt_twh_tot = df_merged["Combined Cycle GT"].sum() / 1e6
            _pump_gen_twh = df_merged["Pumping Turbine"].sum() / 1e6
            _pump_con_twh = df_merged["Pumping Consumption"].sum() / 1e6
            _net_intercon = df_merged["Total Interconnection Balance"].sum() / 1e6 if "Total Interconnection Balance" in df_merged.columns else 0.0
            _fl1.metric("CCGT Generation", f"{_ccgt_twh_tot:.1f} TWh",
                        help=f"Capacity factor: {_ccgt_twh_tot*1e6/df_merged['Combined Cycle GT'].max()/8760*100:.1f}%")
            _fl2.metric("Pumping Discharge", f"{_pump_gen_twh:.1f} TWh")
            _fl3.metric("Pumping Charge", f"{_pump_con_twh:.1f} TWh",
                        help=f"Round-trip efficiency: {_pump_gen_twh/_pump_con_twh*100:.0f}%" if _pump_con_twh > 0 else "N/A")
            _fl4.metric("Net Interconnection Balance", f"{_net_intercon:+.1f} TWh",
                        help="Positive = net import; negative = net export")

            # Daily-average stacked area of flexibility sources
            _flex_daily = df_merged.groupby(df_merged["Date"].dt.date).agg(
                CCGT=("Combined Cycle GT", "mean"),
                Pumping_Discharge=("Pumping Turbine", "mean"),
                Pumping_Charge=("Pumping Consumption", lambda x: x.mean()),
                Net_Intercon=("Total Interconnection Balance", "mean") if "Total Interconnection Balance" in df_merged.columns else ("Combined Cycle GT", lambda x: 0*x.mean()),
            ).reset_index()
            _flex_daily["Date"] = pd.to_datetime(_flex_daily["Date"])

            fig_flex_ov = go.Figure()
            fig_flex_ov.add_trace(go.Scatter(
                x=_flex_daily["Date"], y=_flex_daily["CCGT"],
                mode="lines", stackgroup="pos", name="CCGT",
                line=dict(width=0), fillcolor=COLORS["Combined Cycle GT"],
            ))
            fig_flex_ov.add_trace(go.Scatter(
                x=_flex_daily["Date"], y=_flex_daily["Pumping_Discharge"],
                mode="lines", stackgroup="pos", name="Pumping Discharge",
                line=dict(width=0), fillcolor=COLORS["Pumping Turbine"],
            ))
            if "Total Interconnection Balance" in df_merged.columns:
                _pos_intercon = _flex_daily["Net_Intercon"].clip(lower=0)
                _neg_intercon = _flex_daily["Net_Intercon"].clip(upper=0)
                fig_flex_ov.add_trace(go.Scatter(
                    x=_flex_daily["Date"], y=_pos_intercon,
                    mode="lines", stackgroup="pos", name="Net Import",
                    line=dict(width=0), fillcolor="rgba(100,180,255,0.5)",
                ))
                fig_flex_ov.add_trace(go.Scatter(
                    x=_flex_daily["Date"], y=_neg_intercon,
                    mode="lines", stackgroup="neg", name="Net Export",
                    line=dict(width=0), fillcolor="rgba(255,160,80,0.5)",
                ))
            fig_flex_ov.add_trace(go.Scatter(
                x=_flex_daily["Date"], y=_flex_daily["Pumping_Charge"],
                mode="lines", stackgroup="neg", name="Pumping Charge (−)",
                line=dict(width=0), fillcolor="rgba(255,105,180,0.5)",
            ))
            fig_flex_ov.update_layout(
                height=420, hovermode="x unified",
                title="Daily Average Flexibility Assets — CCGT, Pumping Storage & Interconnectors",
                xaxis_title="Date", yaxis_title="MW (daily avg)",
                plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
                legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
            )
            st.plotly_chart(fig_flex_ov, use_container_width=True)

            st.info(
                "**What this shows:** The daily average output of Spain's three main flexibility "
                "providers. CCGT (red) dominates in winter; storage and cross-border flows smooth "
                "the system year-round. The negative bars (pumping charge + net export) represent "
                "VRE surplus absorption. **Investment angle:** BESS would appear in this chart "
                "as an additional positive bar (discharge) competing with CCGT for the evening "
                "ramp, and an additional negative bar (charge) competing with pumped hydro for "
                "midday solar surplus absorption."
            )

            # Correlation matrix
            st.markdown("#### Flexibility Correlation Matrix")
            _corr_cols = {
                "CCGT": "Combined Cycle GT",
                "Pump Discharge": "Pumping Turbine",
                "Pump Charge": "Pumping Consumption",
                "Solar PV": "Solar PV",
                "Wind": "Wind",
                "DA Price": "price_eur_mwh",
                "VRE Share": "VRE_share",
            }
            _corr_cols = {k: v for k, v in _corr_cols.items() if v in df_merged.columns}
            _corr_data = df_merged[[v for v in _corr_cols.values()]].copy()
            _corr_data.columns = list(_corr_cols.keys())
            if "Total Interconnection Balance" in df_merged.columns:
                _corr_data["Net Interconn."] = df_merged["Total Interconnection Balance"]
            _corr_matrix = _corr_data.corr()

            fig_corr = go.Figure(go.Heatmap(
                z=_corr_matrix.values,
                x=_corr_matrix.columns.tolist(),
                y=_corr_matrix.index.tolist(),
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=_corr_matrix.values.round(2),
                texttemplate="%{text}",
                colorbar=dict(title="Pearson r"),
                hovertemplate="%{x} × %{y} → r=%{z:.2f}<extra></extra>",
            ))
            fig_corr.update_layout(
                height=400,
                title="Pearson Correlation — Flexibility Assets vs VRE & Price",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.info(
                "**How to read:** Strong negative correlation between CCGT and VRE Share (−) "
                "confirms VRE displacement of gas. Pumping Discharge correlates positively with "
                "DA Price — it discharges when prices are high (arbitrage behaviour). "
                "Pumping Charge has a negative price correlation — it charges when prices are low."
            )

        # ════════════════════════════════════════════════════════════════════
        # B — CCGT DEEP DIVE
        # ════════════════════════════════════════════════════════════════════
        elif _flex_section == "B · CCGT Deep Dive":
            st.markdown("### CCGT Deep Dive — When does Spain need gas?")

            # Seasonal profiles 2×2
            _ccgt_sm = make_subplots(rows=2, cols=2, subplot_titles=_seas4,
                                     specs=[[{"secondary_y": True}]*2]*2,
                                     horizontal_spacing=0.10, vertical_spacing=0.18)
            _pos_ccgt = [(1,1),(1,2),(2,1),(2,2)]
            for _i, _sea in enumerate(_seas4):
                _r, _c = _pos_ccgt[_i]
                _ds = df_merged[df_merged["Season"] == _sea]
                _avg_ccgt = _ds.groupby("Hour")["Combined Cycle GT"].mean()
                _avg_vre  = _ds.groupby("Hour").apply(lambda g: (g["Solar PV"]+g["Wind"]).mean())
                _avg_dem  = _ds.groupby("Hour")["Demand"].mean()
                _avg_px   = _ds.groupby("Hour")["price_eur_mwh"].mean()
                _clr = SEASON_COLORS[_sea]
                _sl = _i == 0

                _ccgt_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_ccgt), mode="lines", fill="tozeroy",
                    fillcolor=f"rgba(255,99,71,0.35)", line=dict(color="#FF6347", width=2),
                    name="CCGT", showlegend=_sl,
                    hovertemplate="%{y:.0f} MW CCGT<extra></extra>",
                ), row=_r, col=_c, secondary_y=False)
                _ccgt_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_vre), mode="lines",
                    line=dict(color="#87CEEB", width=1.5, dash="dash"),
                    name="VRE", showlegend=_sl,
                    hovertemplate="%{y:.0f} MW VRE<extra></extra>",
                ), row=_r, col=_c, secondary_y=False)
                _ccgt_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_avg_dem), mode="lines",
                    line=dict(color="black", width=1.2, dash="dot"),
                    name="Demand", showlegend=_sl,
                    hovertemplate="%{y:.0f} MW Demand<extra></extra>",
                ), row=_r, col=_c, secondary_y=False)

            for _ax in _ccgt_sm.layout:
                if _ax.startswith("xaxis"):
                    _ccgt_sm.layout[_ax].update(tickmode="linear", dtick=4)
            _ccgt_sm.update_layout(
                height=520,
                title="Average Hourly CCGT Dispatch by Season (with VRE & Demand context)",
                legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
                plot_bgcolor="white",
            )
            st.plotly_chart(_ccgt_sm, use_container_width=True)

            st.info(
                "**Pattern:** CCGT peaks at 19–21h across all seasons — the evening ramp after "
                "solar generation collapses. In Spring/Summer, the morning CCGT peak largely "
                "disappears (solar takes over from 9h). In Winter, CCGT runs a true dual-peak "
                "(morning + evening). **BESS opportunity:** A 4h battery charged at midday "
                "(cheap solar) and discharged at 18–22h directly displaces this CCGT peak."
            )

            st.divider()

            # CCGT vs VRE scatter
            st.markdown("#### CCGT vs VRE Penetration")
            _ccgt_vre_df = df_merged[["Combined Cycle GT", "VRE_share", "Season"]].copy()
            _ccgt_vre_df = _ccgt_vre_df.sample(min(3000, len(_ccgt_vre_df)), random_state=42)

            fig_ccgt_vre = px.scatter(
                _ccgt_vre_df, x=_ccgt_vre_df["VRE_share"]*100,
                y="Combined Cycle GT", color="Season",
                color_discrete_map=SEASON_COLORS,
                opacity=0.3, size_max=4,
                labels={"x":"VRE Share (%)", "Combined Cycle GT":"CCGT (MW)"},
                title="CCGT Dispatch vs VRE Share — hourly scatter",
            )
            _xv = _ccgt_vre_df["VRE_share"].values * 100
            _yv = _ccgt_vre_df["Combined Cycle GT"].values
            _cv = np.polyfit(_xv, _yv, 2)
            _xr = np.linspace(_xv.min(), _xv.max(), 200)
            fig_ccgt_vre.add_trace(go.Scatter(
                x=_xr, y=np.poly1d(_cv)(_xr), mode="lines",
                name="Poly. regression", line=dict(color="black", width=2.5, dash="dash"),
            ))
            _r2_cv = 1 - np.sum((_yv - np.poly1d(_cv)(_xv))**2) / np.sum((_yv - _yv.mean())**2)
            fig_ccgt_vre.update_layout(height=420, plot_bgcolor="white",
                                        yaxis=dict(gridcolor="#eee"),
                                        annotations=[dict(x=0.02, y=0.95, xref="paper", yref="paper",
                                                          text=f"R² = {_r2_cv:.3f}", showarrow=False,
                                                          font=dict(size=12))])
            st.plotly_chart(fig_ccgt_vre, use_container_width=True)

            st.divider()

            # CCGT operating regime histogram
            st.markdown("#### CCGT Operating Regime")
            _ccgt_c1, _ccgt_c2 = st.columns(2)
            with _ccgt_c1:
                _ccgt_vals = df_merged["Combined Cycle GT"].values
                fig_ccgt_hist = go.Figure(go.Histogram(
                    x=_ccgt_vals, nbinsx=40,
                    marker_color="#FF6347", opacity=0.8,
                ))
                fig_ccgt_hist.update_layout(
                    title="CCGT Output Distribution (all hours)",
                    xaxis_title="MW", yaxis_title="Hours",
                    height=320, plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
                )
                st.plotly_chart(fig_ccgt_hist, use_container_width=True)

            with _ccgt_c2:
                _ccgt_mo_avg = df_merged.groupby("Month")["Combined Cycle GT"].mean()
                _ccgt_mo_z   = df_merged[df_merged["Combined Cycle GT"] == 0].groupby("Month").size()
                _ccgt_mo_z   = _ccgt_mo_z.reindex(range(1, 13), fill_value=0)
                fig_ccgt_mo = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ccgt_mo.add_trace(go.Bar(
                    x=[MONTH_NAMES[m] for m in _ccgt_mo_avg.index],
                    y=_ccgt_mo_avg.values, name="Avg CCGT (MW)",
                    marker_color="#FF6347",
                ), secondary_y=False)
                fig_ccgt_mo.add_trace(go.Scatter(
                    x=[MONTH_NAMES[m] for m in _ccgt_mo_z.index],
                    y=_ccgt_mo_z.values, mode="lines+markers",
                    name="Zero-CCGT hours",
                    line=dict(color="darkred", width=2),
                ), secondary_y=True)
                fig_ccgt_mo.update_layout(
                    title="Monthly Avg CCGT & Zero-CCGT Hours",
                    height=320, plot_bgcolor="white",
                    yaxis=dict(title="Avg MW", gridcolor="#eee"),
                    yaxis2=dict(title="Zero-CCGT hours", showgrid=False),
                    legend=dict(orientation="h", y=-0.3),
                )
                st.plotly_chart(fig_ccgt_mo, use_container_width=True)

        # ════════════════════════════════════════════════════════════════════
        # C — PUMPED HYDRO
        # ════════════════════════════════════════════════════════════════════
        elif _flex_section == "C · Pumped Hydro":
            st.markdown("### Pumped Hydro Storage — Arbitrage Patterns")

            _pump_c1, _pump_c2, _pump_c3 = st.columns(3)
            _pump_gen_t = df_merged["Pumping Turbine"].sum() / 1e6          # positive (production)
            _pump_con_t = abs(df_merged["Pumping Consumption"].sum() / 1e6) # absolute value (consumption stored negative)
            _pump_rte   = _pump_gen_t / _pump_con_t * 100 if _pump_con_t > 0 else 0
            _pump_c1.metric("Annual Discharge", f"{_pump_gen_t:.1f} TWh")
            _pump_c2.metric("Annual Charge", f"{_pump_con_t:.1f} TWh")
            _pump_c3.metric("Round-Trip Efficiency", f"{_pump_rte:.0f}%")

            # Seasonal daily cycle: 2×2 small multiples
            # Pre-compute all series to derive global axis ranges for harmonisation
            _pump_sea_data = {}
            for _sea in _seas4:
                _dsp = df_merged[df_merged["Season"] == _sea]
                _pump_sea_data[_sea] = {
                    "dis": _dsp.groupby("Hour")["Pumping Turbine"].mean(),
                    "cha": _dsp.groupby("Hour")["Pumping Consumption"].mean(),
                    "px":  _dsp.groupby("Hour")["price_eur_mwh"].mean(),
                }
            # Global MW range (symmetric around 0, with padding)
            _mw_max = max(v["dis"].max() for v in _pump_sea_data.values())
            _mw_min = min(v["cha"].min() for v in _pump_sea_data.values())  # already negative
            _mw_pad = (_mw_max - _mw_min) * 0.12
            _mw_range = [_mw_min - _mw_pad, _mw_max + _mw_pad]
            # Global EUR/MWh range (add room below for 0-line annotation)
            _px_max = max(v["px"].max() for v in _pump_sea_data.values())
            _px_min = min(v["px"].min() for v in _pump_sea_data.values())
            _px_pad = (_px_max - _px_min) * 0.15
            _px_range = [min(_px_min - _px_pad, -5), _px_max + _px_pad]

            _pump_sm = make_subplots(rows=2, cols=2, subplot_titles=_seas4,
                                     specs=[[{"secondary_y": True}]*2]*2,
                                     horizontal_spacing=0.10, vertical_spacing=0.18)
            _pos_p = [(1,1),(1,2),(2,1),(2,2)]
            for _i, _sea in enumerate(_seas4):
                _r, _c = _pos_p[_i]
                _sl = _i == 0
                _d = _pump_sea_data[_sea]

                _pump_sm.add_trace(go.Bar(
                    x=_hrs24, y=list(_d["dis"]),
                    name="Discharge (+MW)", marker_color="#00CED1",
                    showlegend=_sl, opacity=0.85,
                ), row=_r, col=_c, secondary_y=False)
                _pump_sm.add_trace(go.Bar(
                    x=_hrs24, y=list(_d["cha"]),
                    name="Charge (−MW)", marker_color="#FF69B4",
                    showlegend=_sl, opacity=0.85,
                ), row=_r, col=_c, secondary_y=False)
                _pump_sm.add_trace(go.Scatter(
                    x=_hrs24, y=list(_d["px"]), mode="lines",
                    line=dict(color="darkorange", width=1.8),
                    name="DA Price (EUR/MWh)", showlegend=_sl,
                ), row=_r, col=_c, secondary_y=True)
                # 0 EUR/MWh reference line on secondary axis
                _pump_sm.add_trace(go.Scatter(
                    x=_hrs24, y=[0]*24, mode="lines",
                    line=dict(color="green", width=1.2, dash="dash"),
                    name="0 EUR/MWh", showlegend=_sl,
                ), row=_r, col=_c, secondary_y=True)

            _pump_sm.update_layout(
                height=560, barmode="relative",
                title="Pumped Hydro Daily Cycle by Season — Discharge (+MW, above 0) / Charge (−MW, below 0) / DA Price (orange) / 0 €/MWh (green dashed)",
                legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
                plot_bgcolor="white",
            )
            # Harmonise all axes: same MW range on primary, same EUR/MWh range on secondary
            for _ax in _pump_sm.layout:
                if _ax.startswith("xaxis"):
                    _pump_sm.layout[_ax].update(tickmode="linear", dtick=4, title_text="Hour")
                if _ax.startswith("yaxis"):
                    _ax_num = int(_ax.replace("yaxis", "") or 1)
                    if _ax_num % 2 == 1:  # primary axes (1,3,5,7) — MW
                        _pump_sm.layout[_ax].update(
                            title_text="MW", range=_mw_range,
                            zeroline=True, zerolinewidth=1.5, zerolinecolor="black",
                            gridcolor="#eee",
                        )
                    else:                 # secondary axes (2,4,6,8) — EUR/MWh
                        _pump_sm.layout[_ax].update(
                            title_text="EUR/MWh", range=_px_range,
                            showgrid=False,
                            zeroline=True, zerolinewidth=1.0, zerolinecolor="green",
                        )
            st.plotly_chart(_pump_sm, use_container_width=True)

            st.info(
                "**Pattern:** Pumped hydro charges (negative bars, pink) primarily at night and "
                "increasingly at midday in summer (cheap solar), and discharges (positive bars, "
                "teal) during the evening price peak. The shift from overnight charging to midday "
                "solar charging is a direct consequence of the duck curve — the same pattern BESS "
                "would replicate. **Limitation:** Spain's pumped hydro (~3.3 GW) is "
                "geographically constrained and cannot scale. BESS fills this gap."
            )

            st.divider()

            # Monthly efficiency chart
            st.markdown("#### Monthly Pumping Volume & Efficiency")
            _pump_mo_gen = df_merged.groupby("Month")["Pumping Turbine"].sum() / 1e6
            _pump_mo_con = (df_merged.groupby("Month")["Pumping Consumption"].sum() / 1e6).abs()
            _pump_mo_eff = (_pump_mo_gen / _pump_mo_con * 100).replace([np.inf, -np.inf], 0).fillna(0)

            fig_pump_mo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pump_mo.add_trace(go.Bar(
                x=[MONTH_NAMES[m] for m in _pump_mo_gen.index],
                y=_pump_mo_gen.values, name="Discharge (TWh)", marker_color="#00CED1",
            ), secondary_y=False)
            fig_pump_mo.add_trace(go.Bar(
                x=[MONTH_NAMES[m] for m in _pump_mo_con.index],
                y=_pump_mo_con.values, name="Charge (TWh)", marker_color="#FF69B4",
            ), secondary_y=False)
            fig_pump_mo.add_trace(go.Scatter(
                x=[MONTH_NAMES[m] for m in _pump_mo_eff.index],
                y=_pump_mo_eff.values, mode="lines+markers",
                name="Round-trip eff. (%)", line=dict(color="purple", width=2),
            ), secondary_y=True)
            fig_pump_mo.update_layout(
                barmode="relative", height=360,
                title="Monthly Pumping Volume (TWh) & Round-Trip Efficiency",
                yaxis=dict(title="TWh", gridcolor="#eee"),
                yaxis2=dict(title="Efficiency (%)", showgrid=False, range=[0, 110]),
                plot_bgcolor="white",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_pump_mo, use_container_width=True)

            # Pumping vs Solar scatter
            st.markdown("#### Midday Charging Pattern vs Solar Generation")
            _midday_mask = (df_merged["Hour"] >= 10) & (df_merged["Hour"] <= 15)
            _pump_solar  = df_merged[_midday_mask][["Solar PV", "Pumping Consumption", "Season"]].copy()
            _pump_solar_s = _pump_solar.sample(min(2000, len(_pump_solar)), random_state=42)
            fig_pump_solar = px.scatter(
                _pump_solar_s, x="Solar PV", y="Pumping Consumption",
                color="Season", color_discrete_map=SEASON_COLORS,
                opacity=0.4, size_max=5,
                title="Midday (10h–15h): Pumping Consumption vs Solar PV Output",
                labels={"Solar PV":"Solar PV (MW)", "Pumping Consumption":"Pumping Charge (MW)"},
            )
            fig_pump_solar.update_layout(height=380, plot_bgcolor="white",
                                          yaxis=dict(gridcolor="#eee"))
            st.plotly_chart(fig_pump_solar, use_container_width=True)
            st.info(
                "**Key finding:** As solar output increases during midday, pumping consumption "
                "also increases — evidence that pumped hydro is increasingly using solar surplus "
                "for charging (summer pattern). This is the closest real-world analogue to BESS "
                "intraday arbitrage. The correlation is strongest in Summer (gold dots)."
            )

        # ════════════════════════════════════════════════════════════════════
        # D — INTERCONNECTORS
        # ════════════════════════════════════════════════════════════════════
        elif _flex_section == "D · Interconnectors":
            st.markdown("### Interconnectors — Cross-Border Flow Patterns")

            # NTC estimates (GW)
            _ntc = {"France Balance": 2800, "Portugal Balance": 3500,
                    "Morocco Balance": 900,  "Balearic Interconnection": 400}

            if not _interconn_cols:
                st.warning("No interconnector columns found in the dataset.")
            else:
                # Headline KPIs
                _ic_cols_kpi = st.columns(len(_interconn_cols))
                for _i, _col in enumerate(_interconn_cols):
                    _net_twh = df_merged[_col].sum() / 1e6
                    _ic_cols_kpi[_i].metric(
                        _col.replace(" Balance", "").replace(" Interconnection", ""),
                        f"{_net_twh:+.1f} TWh",
                        help="+ve = import to Spain, −ve = export from Spain",
                    )

                # Seasonal hourly profiles for each interconnector
                st.markdown("#### Hourly Flow Profiles by Season")
                _ic_sel = st.selectbox("Select Interconnector", _interconn_cols, key="ic_sel")

                fig_ic_sea = go.Figure()
                for _sea in _seas4:
                    _dsi = df_merged[df_merged["Season"] == _sea]
                    _avg_flow = _dsi.groupby("Hour")[_ic_sel].mean()
                    fig_ic_sea.add_trace(go.Scatter(
                        x=_hrs24, y=list(_avg_flow), mode="lines",
                        name=_sea, line=dict(color=SEASON_COLORS[_sea], width=2.5),
                        hovertemplate=f"<b>{_sea}</b>: %{{y:.0f}} MW<extra></extra>",
                    ))
                fig_ic_sea.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_ic_sea.update_layout(
                    height=380, hovermode="x unified",
                    title=f"Avg Hourly Flow — {_ic_sel} by Season (+ve = import to Spain)",
                    xaxis=dict(title="Hour", tickmode="linear", dtick=2),
                    yaxis=dict(title="MW (avg)", gridcolor="#eee"),
                    plot_bgcolor="white",
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig_ic_sea, use_container_width=True)

                st.divider()

                # Flow vs VRE penetration scatter
                st.markdown("#### Flow vs VRE Penetration")
                if "Total Interconnection Balance" in df_merged.columns:
                    _ic_vre = df_merged[["Total Interconnection Balance", "VRE_share", "Season"]].sample(
                        min(3000, len(df_merged)), random_state=42
                    )
                    fig_ic_vre = px.scatter(
                        _ic_vre, x=_ic_vre["VRE_share"]*100,
                        y="Total Interconnection Balance", color="Season",
                        color_discrete_map=SEASON_COLORS, opacity=0.35,
                        title="Net Interconnection Balance vs VRE Share (+ve = import, −ve = export)",
                        labels={"x":"VRE Share (%)", "Total Interconnection Balance":"Net Balance (MW)"},
                    )
                    _xic = _ic_vre["VRE_share"].values * 100
                    _yic = _ic_vre["Total Interconnection Balance"].values
                    _cic = np.polyfit(_xic, _yic, 1)
                    _xri = np.linspace(_xic.min(), _xic.max(), 200)
                    fig_ic_vre.add_trace(go.Scatter(
                        x=_xri, y=np.poly1d(_cic)(_xri), mode="lines",
                        name="Linear trend", line=dict(color="black", width=2, dash="dash"),
                    ))
                    fig_ic_vre.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_ic_vre.update_layout(height=400, plot_bgcolor="white",
                                              yaxis=dict(gridcolor="#eee"))
                    st.plotly_chart(fig_ic_vre, use_container_width=True)

                    _r_ic, _ = scipy_stats.pearsonr(_xic, _yic)
                    st.caption(
                        f"Pearson r = {_r_ic:.3f} — "
                        f"{'negative' if _r_ic < 0 else 'positive'} correlation: "
                        f"{'Spain exports more when VRE is high (surplus pressure)' if _r_ic < 0 else 'Spain imports more when VRE is high (unexpected — check data)'}"
                    )

                # Monthly balance per interconnector
                st.markdown("#### Monthly Balance by Interconnector")
                fig_ic_mo = go.Figure()
                for _col in _interconn_cols:
                    _mo_ic = df_merged.groupby("Month")[_col].sum() / 1e6
                    fig_ic_mo.add_trace(go.Bar(
                        x=[MONTH_NAMES[m] for m in _mo_ic.index],
                        y=_mo_ic.values, name=_col.replace(" Balance", "").replace(" Interconnection", ""),
                    ))
                fig_ic_mo.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
                fig_ic_mo.update_layout(
                    barmode="group", height=360,
                    title="Monthly Net Balance per Interconnector (TWh — +ve import, −ve export)",
                    yaxis=dict(title="TWh", gridcolor="#eee"),
                    plot_bgcolor="white",
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_ic_mo, use_container_width=True)

                # Capacity utilisation
                st.markdown("#### Capacity Utilisation vs NTC")
                if _interconn_cols:
                    _ntc_rows = []
                    for _col in _interconn_cols:
                        _ntc_mw = _ntc.get(_col, 1000)
                        _avg_abs = df_merged[_col].abs().mean()
                        _max_abs = df_merged[_col].abs().max()
                        _util    = _avg_abs / _ntc_mw * 100
                        _congested = int((df_merged[_col].abs() > _ntc_mw * 0.95).sum())
                        _ntc_rows.append({
                            "Interconnector":      _col.replace(" Balance","").replace(" Interconnection",""),
                            "Est. NTC (MW)":       _ntc_mw,
                            "Avg Flow (MW)":       round(_avg_abs, 0),
                            "Peak Flow (MW)":      round(_max_abs, 0),
                            "Avg Utilisation (%)": round(_util, 1),
                            "Hours ≥ 95% NTC":     _congested,
                        })
                    st.dataframe(pd.DataFrame(_ntc_rows), use_container_width=True, hide_index=True)

                st.info(
                    "**Key insight:** Interconnectors act as a pressure valve for Spain's "
                    "renewable surplus. When midday PV drives Spanish prices below French or "
                    "Portuguese prices, Spain exports — sharing its solar surplus with neighbours. "
                    "However, total NTC (~7 GW) is small relative to 32 GW of PV, so congestion "
                    "is growing. As PV capacity increases further, more surplus is trapped "
                    "within Spain, accelerating price cannibalisation — strengthening the BESS "
                    "investment case."
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 7 — PROJECTIONS 2026-2030  (moved, with sliders + presets)
    # ════════════════════════════════════════════════════════════════════════
    with tab7:
        st.subheader("🔭 Projections 2026-2030 — VRE + BESS Scenario")

        # ── SCENARIO PRESETS  (5.2.1) ────────────────────────────────────────
        st.markdown("#### Pre-configured Scenarios")
        _PRESETS = {
            "Status Quo": dict(solar=0.0, wind=0.0, bess=0.0, demand=1.5),
            "Low":        dict(solar=2.5, wind=1.0, bess=0.25, demand=1.0),
            "Central":    dict(solar=4.0, wind=2.0, bess=0.5,  demand=1.5),
            "High":       dict(solar=6.0, wind=3.0, bess=1.0,  demand=2.0),
        }
        _pr1, _pr2, _pr3, _pr4 = st.columns(4)
        for _pcol, (_pname, _pvals) in zip([_pr1, _pr2, _pr3, _pr4], _PRESETS.items()):
            if _pcol.button(f"📋 {_pname}", use_container_width=True,
                            help=f"PV {_pvals['solar']} · Wind {_pvals['wind']} · BESS {_pvals['bess']} GW/yr"):
                st.session_state.proj_solar    = _pvals["solar"]
                st.session_state.proj_wind     = _pvals["wind"]
                st.session_state.proj_bess     = _pvals["bess"]
                st.session_state.proj_demand   = _pvals["demand"]
                st.rerun()

        # ── SLIDERS  (5.2.2 — moved from sidebar) ───────────────────────────
        st.markdown("#### Capacity Addition Parameters")
        _sc1, _sc2, _sc3 = st.columns(3)
        solar_annual  = _sc1.slider("☀️ Solar PV (GW/yr)",  0.0, 8.0,
                                     float(st.session_state.proj_solar), 0.5, key="proj_solar")
        wind_annual   = _sc2.slider("💨 Wind (GW/yr)",       0.0, 5.0,
                                     float(st.session_state.proj_wind),  0.5, key="proj_wind")
        bess_annual   = _sc3.slider("🔋 BESS (GW/yr)",       0.0, 3.0,
                                     float(st.session_state.proj_bess),  0.25, key="proj_bess")
        _sd1, _sd2 = st.columns(2)
        bess_duration = _sd1.radio("BESS Storage Duration", [2, 4], index=1 if st.session_state.proj_bess_dur == 4 else 0,
                                    horizontal=True, format_func=lambda x: f"{x}h",
                                    help="2h = intraday arbitrage | 4h = evening peak shift",
                                    key="proj_bess_dur")
        demand_growth = _sd2.slider("📈 Demand Growth (%/yr)", -1.0, 3.0,
                                     float(st.session_state.proj_demand), 0.5, key="proj_demand")

        # Highlight active preset
        _active = next((n for n, v in _PRESETS.items()
                        if abs(solar_annual-v["solar"])<0.01 and abs(wind_annual-v["wind"])<0.01
                        and abs(bess_annual-v["bess"])<0.01 and abs(demand_growth-v["demand"])<0.01),
                       "Custom")
        st.caption(f"Active scenario: **{_active}** — "
                   f"☀️ {solar_annual} GW/yr · 💨 {wind_annual} GW/yr · "
                   f"🔋 {bess_annual} GW/yr · 📈 {demand_growth}%/yr")
        st.divider()

        with st.expander("ℹ️ Model Methodology", expanded=False):
            st.markdown("""
            **Simplified model assumptions:**
            - **PV and wind** production are scaled proportionally to new capacity (2025 profile × capacity ratio).
            - **Nuclear**, **hydro** and **cogeneration** remain constant.
            - **BESS** follows a simple dispatch: charge on instantaneous VRE surplus, discharge on residual deficit.
            - **CCGT** fills residual demand (capped at the maximum observed 2025 output).
            - **Unabsorbed surplus** is counted as curtailment.

            **Limitations:** The model does not simulate market prices, merit-order dynamics or
            financial BESS optimisation. It provides a **structural view** of how the mix evolves,
            suitable for VRE + storage investment analysis.
            """)

        years = [2025, 2026, 2027, 2028, 2029, 2030]
        res   = {}
        for yr in years:
            n = yr - 2025
            res[yr] = project_year(
                delta_solar_gw        = n * solar_annual,
                delta_wind_gw         = n * wind_annual,
                bess_total_gw         = BASE_BESS_GW + n * bess_annual,
                bess_dur_h            = bess_duration,
                cum_demand_growth_pct = n * demand_growth,
            )

        # ── Summary table ────────────────────────────────────────────────────
        summary = pd.DataFrame({
            "Year":                 years,
            "☀️ Solar PV (GW)":    [round(BASE_SOLAR_GW + (y-2025)*solar_annual, 1) for y in years],
            "💨 Wind (GW)":        [round(BASE_WIND_GW  + (y-2025)*wind_annual,  1) for y in years],
            "🔋 BESS (GW)":        [round(BASE_BESS_GW  + (y-2025)*bess_annual,  2) for y in years],
            "Demand (TWh)":         [round(res[y]["demand_twh"],    1) for y in years],
            "REN Share (%)":        [round(res[y]["renewable_pct"],  1) for y in years],
            "Curtailment (TWh)":    [round(res[y]["curtailment_twh"],2) for y in years],
            "Residual CCGT (TWh)":  [round(res[y]["ccgt_twh"],       1) for y in years],
            "Hours VRE > 70%":      [res[y]["cannib_70"]               for y in years],
            "Hours VRE > 90%":      [res[y]["cannib_90"]               for y in years],
            "BESS Cycles/yr":       [round(res[y]["batt_cycles"],    0) for y in years],
        })

        st.markdown("### 📋 Scenario Summary Table")
        st.dataframe(
            summary.style
                   .format({"REN Share (%)":       "{:.1f}%",
                             "Curtailment (TWh)":   "{:.2f}",
                             "BESS Cycles/yr":      "{:.0f}"})
                   .background_gradient(subset=["REN Share (%)"],     cmap="Greens", vmin=40, vmax=100)
                   .background_gradient(subset=["Curtailment (TWh)"], cmap="OrRd",   vmin=0,  vmax=10)
                   .background_gradient(subset=["Hours VRE > 70%"],   cmap="YlOrRd", vmin=0,  vmax=6000),
            use_container_width=True,
            hide_index=True,
        )

        # ── Trajectory charts ────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            fig_ren = go.Figure(go.Bar(
                x=years,
                y=[res[y]["renewable_pct"] for y in years],
                marker_color=["#90EE90" if y > 2025 else "#4169E1" for y in years],
                text=[f"{res[y]['renewable_pct']:.1f}%" for y in years],
                textposition="outside",
            ))
            fig_ren.update_layout(
                title="Renewable Share Evolution (%)",
                yaxis=dict(range=[0, 100], gridcolor="#eee"),
                height=300, plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(fig_ren, use_container_width=True)

            fig_disp = go.Figure()
            fig_disp.add_trace(go.Bar(
                x=years, y=[res[y]["curtailment_twh"] for y in years],
                name="Curtailment (TWh)", marker_color="#FF6347",
            ))
            fig_disp.add_trace(go.Bar(
                x=years, y=[res[y]["ccgt_twh"] for y in years],
                name="Residual CCGT (TWh)", marker_color="#FFA500",
            ))
            fig_disp.update_layout(
                barmode="group", height=300,
                title="VRE Curtailment & Residual CCGT (TWh)",
                plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
                legend=dict(orientation="h", y=-0.35),
            )
            st.plotly_chart(fig_disp, use_container_width=True)

        with col2:
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(
                x=years, y=[BASE_SOLAR_GW + (y-2025)*solar_annual for y in years],
                mode="lines+markers+text", name="☀️ Solar PV (GW)",
                line=dict(color="#FFD700", width=2.5),
                text=[f"{BASE_SOLAR_GW+(y-2025)*solar_annual:.0f}" for y in years],
                textposition="top center", textfont=dict(color="#997700"),
            ))
            fig_cap.add_trace(go.Scatter(
                x=years, y=[BASE_WIND_GW + (y-2025)*wind_annual for y in years],
                mode="lines+markers+text", name="💨 Wind (GW)",
                line=dict(color="#87CEEB", width=2.5),
                text=[f"{BASE_WIND_GW+(y-2025)*wind_annual:.0f}" for y in years],
                textposition="bottom center", textfont=dict(color="#336699"),
            ))
            fig_cap.add_trace(go.Scatter(
                x=years, y=[BASE_BESS_GW + (y-2025)*bess_annual for y in years],
                mode="lines+markers+text", name="🔋 BESS (GW)",
                line=dict(color="#00FA9A", width=2.5),
                text=[f"{BASE_BESS_GW+(y-2025)*bess_annual:.1f}" for y in years],
                textposition="top center", textfont=dict(color="#009955"),
                yaxis="y2",
            ))
            fig_cap.update_layout(
                title="Installed Capacity Trajectory (GW)",
                yaxis=dict(title="GW (Solar PV & Wind)", gridcolor="#eee"),
                yaxis2=dict(title="GW (BESS)", overlaying="y", side="right", showgrid=False),
                height=300, plot_bgcolor="white",
                legend=dict(orientation="h", y=-0.35),
            )
            st.plotly_chart(fig_cap, use_container_width=True)

            fig_cann = go.Figure()
            fig_cann.add_trace(go.Bar(
                x=years, y=[res[y]["cannib_70"] for y in years],
                name="VRE > 70%", marker_color="#E53935",
                text=[f"{res[y]['cannib_70']}" for y in years],
                textposition="outside",
            ))
            fig_cann.add_trace(go.Bar(
                x=years, y=[res[y]["cannib_90"] for y in years],
                name="VRE > 90%", marker_color="#B71C1C",
                text=[f"{res[y]['cannib_90']}" for y in years],
                textposition="outside",
            ))
            fig_cann.update_layout(
                barmode="group", height=300,
                title="VRE Cannibalisation Hours (h/yr)",
                yaxis_title="Hours", plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
                legend=dict(orientation="h", y=-0.35),
            )
            st.plotly_chart(fig_cann, use_container_width=True)

        # ── Projected dispatch for a selected year ───────────────────────────
        st.markdown("---")
        st.subheader("Projected Hourly Dispatch — Monthly Zoom")

        ca, cb = st.columns(2)
        yr_sel  = ca.select_slider("Year",  options=years, value=2028, key="yr_sel")
        mo_sel2 = cb.selectbox("Month", list(MONTH_NAMES.values()), index=3, key="mo_sel2")

        m2    = month_num[mo_sel2]
        df_p  = res[yr_sel]["df"]
        df_pm = df_p[df_p["Month"] == m2].copy()
        idx   = df_pm.index

        techs_nz_proj = [t for t in DISPATCH_TECHS if t in df_pm.columns and df_pm[t].sum() > 0]
        df_pm["Battery Discharge"] = res[yr_sel]["batt_discharge"].reindex(idx).fillna(0)
        df_pm["Battery Charging"]  = -res[yr_sel]["batt_charge"].reindex(idx).fillna(0)
        curtm = res[yr_sel]["curtailment"].reindex(idx).fillna(0)

        techs_proj = [t for t in techs_nz_proj + ["Battery Discharge"]
                      if t in df_pm.columns and df_pm[t].sum() > 0]

        fig_proj = go.Figure()
        for t in techs_proj:
            fig_proj.add_trace(go.Scatter(
                x=df_pm["Date"], y=df_pm[t],
                mode="lines", stackgroup="one", name=t,
                line=dict(width=0), fillcolor=COLORS.get(t, "#AAA"),
                hovertemplate=f"<b>{t}</b>: %{{y:.0f}} MW<extra></extra>"
            ))
        if curtm.sum() > 0:
            fig_proj.add_trace(go.Scatter(
                x=df_pm["Date"], y=curtm,
                mode="lines", stackgroup="one", name="⚡ Curtailment",
                line=dict(width=0), fillcolor="rgba(220,0,0,0.25)",
                hovertemplate="<b>Curtailment</b>: %{y:.0f} MW<extra></extra>"
            ))
        fig_proj.add_trace(go.Scatter(
            x=df_pm["Date"], y=df_pm["Demand"],
            mode="lines", name="Demand",
            line=dict(color="black", width=2, dash="dot"),
            hovertemplate="<b>Demand</b>: %{y:.0f} MW<extra></extra>"
        ))

        bess_total = BASE_BESS_GW + (yr_sel-2025)*bess_annual
        fig_proj.update_layout(
            height=500, hovermode="x unified",
            title=(f"Dispatch — {mo_sel2} {yr_sel}  |  "
                   f"+{(yr_sel-2025)*solar_annual:.0f} GW Solar PV  •  "
                   f"+{(yr_sel-2025)*wind_annual:.0f} GW Wind  •  "
                   f"{bess_total:.1f} GW BESS"),
            xaxis_title="Date", yaxis_title="MW",
            plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
        )
        st.plotly_chart(fig_proj, use_container_width=True)

        if bess_annual > 0 or BASE_BESS_GW > 0:
            bess_mwh = bess_total * 1000 * bess_duration
            soc_m    = res[yr_sel]["batt_soc"].reindex(idx).fillna(0)
            fig_soc  = go.Figure()
            fig_soc.add_trace(go.Scatter(
                x=df_pm["Date"], y=soc_m / max(bess_mwh, 1) * 100,
                mode="lines", fill="tozeroy",
                fillcolor="rgba(0,250,154,0.25)",
                line=dict(color="#00FA9A", width=1.5),
                name="SOC (%)",
                hovertemplate="SOC: %{y:.1f}%<extra></extra>",
            ))
            fig_soc.update_layout(
                title=f"BESS State of Charge — {mo_sel2} {yr_sel}  |  {bess_total:.1f} GW / {bess_mwh:.0f} MWh",
                yaxis=dict(range=[0, 110], title="SOC (%)", gridcolor="#eee"),
                height=240, plot_bgcolor="white",
            )
            st.plotly_chart(fig_soc, use_container_width=True)

        st.info(
            "**What this shows:** Projected hourly dispatch for a selected year and month based on "
            "the chosen scenario. Solar PV and Wind are scaled proportionally to capacity additions. "
            "BESS charges on surplus and discharges on deficit. CCGT fills remaining demand. "
            "**Limitation:** This is a structural energy balance model, not a market simulation — "
            "prices and financial metrics are not projected here."
        )


# ============================================================================
if __name__ == "__main__":
    main()
