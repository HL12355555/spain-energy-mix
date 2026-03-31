"""
Mix Énergétique Espagnol — Analyse & Projections 2025-2030
Développé pour EREN Investment Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="Mix Énergétique Espagnol",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTES
# ============================================================================

# Capacités installées estimées Espagne fin 2025 (GW)
BASE_SOLAR_GW   = 32.0
BASE_WIND_GW    = 31.0
BASE_BESS_GW    = 0.5   # quasi-nul en 2025

MONTH_NAMES = {
    1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr",
    5:"Mai", 6:"Jun", 7:"Jul", 8:"Août",
    9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"
}
SEASON_MAP = {
    1:"Hiver",  2:"Hiver",  3:"Printemps",
    4:"Printemps", 5:"Printemps", 6:"Été",
    7:"Été",    8:"Été",    9:"Automne",
    10:"Automne",11:"Automne",12:"Hiver"
}

# Technologies de production (stackées dans le dispatch)
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
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv("spain_mix_data.csv", parse_dates=["Date"])
    # Suppression des heures avec Demand = 0 (données manquantes avril)
    df = df[df["Demand"] > 0].copy()
    df["MonthName"] = df["Month"].map(MONTH_NAMES)
    df["Season"]    = df["Month"].map(SEASON_MAP)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["VRE_share"] = (df["Solar PV"] + df["Wind"]) / df["Demand"]
    return df

# ============================================================================
# DISPATCH BATTERIE
# ============================================================================

def simulate_battery(surplus_arr: np.ndarray, cap_mw: float,
                     duration_h: float, efficiency: float = 0.90):
    """
    Dispatch simplifié : charge sur surplus VRE, décharge sur déficit.
    Retourne (charge, discharge, soc) en MW / MWh.
    """
    capacity_mwh = cap_mw * duration_h
    n = len(surplus_arr)
    soc      = np.zeros(n + 1)
    charge   = np.zeros(n)
    discharge= np.zeros(n)

    for i in range(n):
        s = surplus_arr[i]
        if s > 0:
            max_ch  = min(cap_mw, s, (capacity_mwh - soc[i]) / max(efficiency, 0.01))
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
# MOTEUR DE PROJECTION
# ============================================================================

@st.cache_data
def project_year(delta_solar_gw: float, delta_wind_gw: float,
                 bess_total_gw: float, bess_dur_h: float,
                 cum_demand_growth_pct: float):
    """
    Projette le mix énergétique pour un scénario donné.

    Hypothèses de modélisation (simplifiées) :
    - La production PV et éolienne scale proportionnellement aux capacités.
    - Le nucléaire, l'hydro et la cogénération restent constants.
    - Le CCGT remplit le résidu de demande (plafonné à sa capacité max 2025).
    - Le BESS charge sur les surplus VRE, décharge sur les déficits.
    """
    df = load_data().copy()

    solar_scale  = 1 + delta_solar_gw / BASE_SOLAR_GW
    wind_scale   = 1 + delta_wind_gw  / BASE_WIND_GW
    demand_scale = 1 + cum_demand_growth_pct / 100.0

    df["Solar PV"]    = df["Solar PV"]    * solar_scale
    df["Wind"]        = df["Wind"]        * wind_scale
    df["Solar Thermal"] = df["Solar Thermal"] * solar_scale
    df["Demand"]      = df["Demand"]      * demand_scale

    # Génération "prévisible" : VRE + must-run
    must_run = (df["Nuclear"] + df["Hydro (on-flow)"] + df["Cogeneration"]
                + df["Solar Thermal"] + df["Other Renewables"])
    surplus  = df["Solar PV"] + df["Wind"] + must_run - df["Demand"]

    # Simulation BESS
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

    net_surplus  = surplus + batt_discharge - batt_charge
    curtailment  = net_surplus.clip(lower=0)
    deficit      = (-net_surplus).clip(lower=0)
    ccgt_cap     = df["Combined Cycle GT"].max()
    ccgt_dispatch= deficit.clip(upper=ccgt_cap)

    # Métriques
    total_demand  = df["Demand"].sum()
    ren_gen       = (df["Solar PV"] + df["Wind"] + df["Hydro (on-flow)"]
                     + df["Solar Thermal"] + df["Other Renewables"]).sum()
    vre_share_s   = (df["Solar PV"] + df["Wind"]) / df["Demand"]
    bess_energy   = bess_total_gw * 1000 * bess_dur_h if bess_total_gw > 0 else 1

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
# APP PRINCIPALE
# ============================================================================

def main():
    df = load_data()

    # ── HEADER ──────────────────────────────────────────────────────────────
    st.title("⚡ Mix Énergétique Espagnol — Analyse & Projections 2025-2030")
    st.caption(
        f"Données REE • {len(df):,} heures analysées • Nota : données avril partiellement manquantes"
    )

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Paramètres")

        st.markdown("### 🔭 Scénario Projections")
        st.markdown("*Capacités additionnelles annuelles (GW)*")

        solar_annual  = st.slider("☀️ Solaire PV (GW/an)",  0.0, 8.0, 4.0, 0.5,
                                   help="Additions annuelles de capacité PV. Réf. Aurora Central : ~4 GW/an")
        wind_annual   = st.slider("💨 Éolien (GW/an)",       0.0, 5.0, 2.0, 0.5,
                                   help="Additions annuelles éolien. Réf. PNIEC : ~2-3 GW/an")
        bess_annual   = st.slider("🔋 BESS (GW/an)",         0.0, 3.0, 0.5, 0.25,
                                   help="Capacité BESS ajoutée chaque année")
        bess_duration = st.radio("Durée de stockage BESS",  [2, 4], index=1,
                                  horizontal=True,
                                  format_func=lambda x: f"{x}h",
                                  help="2h = arbitrage intraday | 4h = pic soir")
        demand_growth = st.slider("📈 Croissance demande (%/an)", -1.0, 3.0, 1.5, 0.5,
                                   help="Croissance annuelle de la demande électrique")

        st.divider()
        st.markdown("**Capacités de référence (2025)**")
        st.markdown(f"☀️ PV : **{BASE_SOLAR_GW:.0f} GW**")
        st.markdown(f"💨 Éolien : **{BASE_WIND_GW:.0f} GW**")
        st.markdown(f"🔋 BESS : **{BASE_BESS_GW:.1f} GW**")

        st.divider()
        st.info(
            "**Sources :** REE 2025 (données horaires) • "
            "Aurora Energy Research Dec-25 • Alantra BESS Nov-25"
        )

    # ── ONGLETS ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dispatch 2025",
        "📅 Analyses Mensuelles",
        "📈 KPIs du Mix",
        "🔭 Projections 2026-2030",
        "🌡️ Heatmaps",
    ])

    month_num = {v: k for k, v in MONTH_NAMES.items()}

    # ════════════════════════════════════════════════════════════════════════
    # ONGLET 1 — DISPATCH HORAIRE 2025
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Courbe de dispatch horaire 2025")

        c1, c2, c3 = st.columns(3)
        sel_month  = c1.selectbox("Mois", list(MONTH_NAMES.values()), index=0, key="t1_month")
        view_mode  = c2.radio("Plage", ["Semaine type", "Mois complet"], horizontal=True, key="t1_view")
        show_dem   = c3.checkbox("Afficher la demande", value=True, key="t1_dem")

        m = month_num[sel_month]
        df_m = df[df["Month"] == m].copy()

        if view_mode == "Semaine type":
            days = sorted(df_m["Day"].unique())[:7]
            df_v = df_m[df_m["Day"].isin(days)]
        else:
            df_v = df_m

        techs = [t for t in DISPATCH_TECHS
                 if t in df_v.columns and df_v[t].sum() > 0]

        fig = go.Figure()
        for t in techs:
            fig.add_trace(go.Scatter(
                x=df_v["Date"], y=df_v[t],
                mode="lines", stackgroup="one", name=t,
                line=dict(width=0), fillcolor=COLORS.get(t, "#AAAAAA"),
                hovertemplate=f"<b>{t}</b>: %{{y:.0f}} MW<extra></extra>"
            ))
        if show_dem:
            fig.add_trace(go.Scatter(
                x=df_v["Date"], y=df_v["Demand"],
                mode="lines", name="Demande",
                line=dict(color="black", width=2, dash="dot"),
                hovertemplate="<b>Demande</b>: %{y:.0f} MW<extra></extra>"
            ))

        fig.update_layout(
            height=480, hovermode="x unified",
            title=f"Dispatch horaire — {sel_month} 2025",
            xaxis_title="Date", yaxis_title="Production (MW)",
            plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # KPIs rapides
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Part VRE (PV+Éolien)",
                  f"{(df_m['Solar PV'].sum()+df_m['Wind'].sum())/df_m['Demand'].sum()*100:.1f}%")
        k2.metric("Demande moyenne",  f"{df_m['Demand'].mean():,.0f} MW")
        k3.metric("Pic PV",           f"{df_m['Solar PV'].max():,.0f} MW")
        k4.metric("Pic éolien",       f"{df_m['Wind'].max():,.0f} MW")

    # ════════════════════════════════════════════════════════════════════════
    # ONGLET 2 — ANALYSES MENSUELLES / SAISONNIÈRES
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Analyse mensuelle et saisonnière")

        # ---- Production mensuelle
        techs_nz = [t for t in DISPATCH_TECHS if df[t].sum() > 0]
        monthly  = df.groupby("Month")[techs_nz].sum() / 1e6   # TWh
        monthly.index = [MONTH_NAMES[i] for i in monthly.index]
        dem_mo   = df.groupby("Month")["Demand"].sum() / 1e6
        dem_mo.index = monthly.index

        fig_mo = go.Figure()
        for t in techs_nz:
            fig_mo.add_trace(go.Bar(
                name=t, x=monthly.index, y=monthly[t],
                marker_color=COLORS.get(t, "#AAA")
            ))
        fig_mo.add_trace(go.Scatter(
            x=dem_mo.index, y=dem_mo.values,
            mode="lines+markers", name="Demande",
            line=dict(color="black", width=2)
        ))
        fig_mo.update_layout(
            barmode="stack", height=400,
            title="Production mensuelle par technologie (TWh)",
            yaxis_title="TWh", plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
        )
        st.plotly_chart(fig_mo, use_container_width=True)

        # ---- Profil journalier moyen par saison
        st.subheader("Profil journalier moyen par saison")
        seasons = ["Hiver", "Printemps", "Été", "Automne"]
        sel_sea = st.radio("Saison", seasons, horizontal=True, key="t2_sea")

        df_s    = df[df["Season"] == sel_sea]
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
            mode="lines", name="Demande",
            line=dict(color="black", width=2, dash="dot")
        ))
        fig_day.update_layout(
            height=420, hovermode="x unified",
            title=f"Profil journalier moyen — {sel_sea}",
            xaxis_title="Heure", yaxis_title="MW (moyenne)",
            xaxis=dict(tickmode="linear", dtick=2),
            plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        )
        st.plotly_chart(fig_day, use_container_width=True)

        # ---- Part ENR par saison
        c_a, c_b = st.columns(2)
        seasonal = df.groupby("Season").agg(
            Solar_PV  = ("Solar PV",        "sum"),
            Wind      = ("Wind",             "sum"),
            Hydro     = ("Hydro (on-flow)",  "sum"),
            SolarTh   = ("Solar Thermal",    "sum"),
            OtherRen  = ("Other Renewables", "sum"),
            Demand    = ("Demand",           "sum"),
        )
        seasonal["REN_pct"] = (seasonal[["Solar_PV","Wind","Hydro","SolarTh","OtherRen"]].sum(axis=1)
                               / seasonal["Demand"] * 100)
        sea_order = ["Hiver","Printemps","Été","Automne"]
        seasonal  = seasonal.reindex([s for s in sea_order if s in seasonal.index])
        sea_colors= {"Hiver":"#4169E1","Printemps":"#90EE90","Été":"#FFD700","Automne":"#D2691E"}

        with c_a:
            fig_s1 = px.bar(
                x=seasonal.index, y=seasonal["REN_pct"],
                color=seasonal.index,
                color_discrete_map=sea_colors,
                title="Part ENR par saison (%)",
                labels={"x":"Saison","y":"Part ENR (%)"},
                text_auto=".1f"
            )
            fig_s1.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            fig_s1.update_layout(showlegend=False, height=300, yaxis=dict(range=[0,100]),
                                  plot_bgcolor="white", )
            st.plotly_chart(fig_s1, use_container_width=True)

        with c_b:
            fig_s2 = go.Figure()
            for col, name, clr in [("Solar_PV","☀️ PV","#FFD700"),
                                    ("Wind","💨 Éolien","#87CEEB"),
                                    ("Hydro","💧 Hydro","#4169E1")]:
                fig_s2.add_trace(go.Bar(
                    name=name, x=seasonal.index,
                    y=seasonal[col]/seasonal["Demand"]*100,
                    marker_color=clr
                ))
            fig_s2.update_layout(
                barmode="group", height=300,
                title="Détail VRE par saison (%)",
                yaxis_title="%", plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
                legend=dict(orientation="h", y=-0.35)
            )
            st.plotly_chart(fig_s2, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # ONGLET 3 — KPIs 2025
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Indicateurs clés du mix espagnol 2025")

        tot_dem   = df["Demand"].sum()     / 1e6
        tot_ren   = sum(df[t].sum() for t in RENEWABLES) / 1e6
        tot_vre   = (df["Solar PV"].sum() + df["Wind"].sum()) / 1e6
        h_vre70   = int((df["VRE_share"] > 0.70).sum())
        h_pv50    = int((df["Solar PV"] / df["Demand"] > 0.50).sum())
        h_surplus = int((df["Solar PV"] + df["Wind"] > df["Demand"]).sum())

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Demande totale",    f"{tot_dem:.0f} TWh")
        k2.metric("Part ENR",          f"{tot_ren/tot_dem*100:.1f}%")
        k3.metric("Part PV + Éolien",  f"{tot_vre/tot_dem*100:.1f}%")
        k4.metric("Heures VRE > 70%",  f"{h_vre70:,} h",
                  help="Proxy cannibalisation = heures où PV+Éolien > 70% de la demande")
        k5.metric("Heures PV > 50%",   f"{h_pv50:,} h",
                  help="Heures où le seul PV dépasse 50% de la demande")
        k6.metric("Heures PV+Vent > Demande", f"{h_surplus:,} h",
                  help="Heures où les ENR seules dépassent la demande totale")

        st.divider()
        col1, col2 = st.columns([1.2, 1])

        with col1:
            # Pie technologie
            tech_sums = {t: df[t].sum() for t in DISPATCH_TECHS if df[t].sum() > 0}
            fig_pie = px.pie(
                values=list(tech_sums.values()),
                names=list(tech_sums.keys()),
                title="Part de chaque technologie dans le mix 2025",
                color=list(tech_sums.keys()),
                color_discrete_map=COLORS,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                                   textfont_size=10)
            fig_pie.update_layout(height=460, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Distribution part PV
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
                title="Heures par part PV dans la demande",
                labels={"x":"Part PV","y":"Heures"},
                text_auto=True
            )
            fig_dist.update_layout(showlegend=False, height=230,
                                    plot_bgcolor="white", yaxis=dict(gridcolor="#eee"))
            st.plotly_chart(fig_dist, use_container_width=True)

            # CCGT selon VRE
            vre_bins = pd.cut(df["VRE_share"]*100,
                               bins=[0,25,50,75,100], labels=["0-25%","25-50%","50-75%",">75%"])
            ccgt_vre = df.groupby(vre_bins, observed=True)["Combined Cycle GT"].mean()

            fig_ccgt = px.bar(
                x=ccgt_vre.index.astype(str), y=ccgt_vre.values,
                title="CCGT moyen (MW) selon la part VRE",
                labels={"x":"Part VRE","y":"CCGT moyen (MW)"},
                color_discrete_sequence=["#FF6347"],
                text_auto=".0f"
            )
            fig_ccgt.update_layout(showlegend=False, height=210,
                                    plot_bgcolor="white", yaxis=dict(gridcolor="#eee"))
            st.plotly_chart(fig_ccgt, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # ONGLET 4 — PROJECTIONS 2026-2030
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🔭 Projections 2026-2030 — Scénario ENR + BESS")

        with st.expander("ℹ️ Méthodologie du modèle", expanded=False):
            st.markdown("""
            **Hypothèses simplifiées du modèle :**
            - La production **PV et éolienne** est scalée proportionnellement aux nouvelles capacités (profil 2025 × ratio capacité).
            - Le **nucléaire**, l'**hydro** et la **cogénération** restent constants.
            - Le **BESS** suit un dispatch simple : charge sur les surplus VRE instantanés, décharge sur les déficits residuels.
            - Le **CCGT** remplit le résidu de demande (plafonné à la capacité max observée en 2025).
            - L'**excédent non absorbé** est compté comme curtailment.

            **Limites :** Le modèle ne simule pas les prix de marché, le merit order, ni l'optimisation financière du BESS.
            Il donne une **vue structurelle** de l'évolution du mix, adaptée à l'analyse d'investissement ENR + stockage.
            """)

        years = [2025, 2026, 2027, 2028, 2029, 2030]
        res   = {}
        for yr in years:
            n = yr - 2025
            res[yr] = project_year(
                delta_solar_gw       = n * solar_annual,
                delta_wind_gw        = n * wind_annual,
                bess_total_gw        = BASE_BESS_GW + n * bess_annual,
                bess_dur_h           = bess_duration,
                cum_demand_growth_pct= n * demand_growth,
            )

        # ── Tableau de synthèse ──────────────────────────────────────────
        summary = pd.DataFrame({
            "Année":             years,
            "☀️ PV installé (GW)":   [round(BASE_SOLAR_GW + (y-2025)*solar_annual, 1) for y in years],
            "💨 Éolien (GW)":        [round(BASE_WIND_GW  + (y-2025)*wind_annual,  1) for y in years],
            "🔋 BESS (GW)":          [round(BASE_BESS_GW  + (y-2025)*bess_annual,  2) for y in years],
            "Demande (TWh)":         [round(res[y]["demand_twh"],    1) for y in years],
            "Part ENR (%)":          [round(res[y]["renewable_pct"],  1) for y in years],
            "Curtailment (TWh)":     [round(res[y]["curtailment_twh"],2) for y in years],
            "CCGT résiduel (TWh)":   [round(res[y]["ccgt_twh"],       1) for y in years],
            "Heures VRE > 70%":      [res[y]["cannib_70"]               for y in years],
            "Heures VRE > 90%":      [res[y]["cannib_90"]               for y in years],
            "Cycles BESS/an":        [round(res[y]["batt_cycles"],    0) for y in years],
        })

        st.markdown("### 📋 Tableau de synthèse des scénarios")
        st.dataframe(
            summary.style
                   .format({"Part ENR (%)": "{:.1f}%",
                             "Curtailment (TWh)": "{:.2f}",
                             "Cycles BESS/an": "{:.0f}"})
                   .background_gradient(subset=["Part ENR (%)"],      cmap="Greens", vmin=40, vmax=100)
                   .background_gradient(subset=["Curtailment (TWh)"], cmap="OrRd",   vmin=0,  vmax=10)
                   .background_gradient(subset=["Heures VRE > 70%"],  cmap="YlOrRd", vmin=0,  vmax=6000),
            use_container_width=True,
            
        )

        # ── Graphiques évolution ─────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            # Part ENR
            fig_ren = go.Figure(go.Bar(
                x=years,
                y=[res[y]["renewable_pct"] for y in years],
                marker_color=["#90EE90" if y > 2025 else "#4169E1" for y in years],
                text=[f"{res[y]['renewable_pct']:.1f}%" for y in years],
                textposition="outside",
            ))
            fig_ren.update_layout(
                title="Évolution de la part ENR (%)",
                yaxis=dict(range=[0, 100], gridcolor="#eee"),
                height=300, plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(fig_ren, use_container_width=True)

            # Curtailment & CCGT
            fig_disp = go.Figure()
            fig_disp.add_trace(go.Bar(
                x=years, y=[res[y]["curtailment_twh"] for y in years],
                name="Curtailment (TWh)", marker_color="#FF6347",
            ))
            fig_disp.add_trace(go.Bar(
                x=years, y=[res[y]["ccgt_twh"] for y in years],
                name="CCGT résiduel (TWh)", marker_color="#FFA500",
            ))
            fig_disp.update_layout(
                barmode="group", height=300,
                title="Curtailment ENR & CCGT résiduel (TWh)",
                plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
                legend=dict(orientation="h", y=-0.35),
            )
            st.plotly_chart(fig_disp, use_container_width=True)

        with col2:
            # Trajectoire capacités
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(
                x=years, y=[BASE_SOLAR_GW + (y-2025)*solar_annual for y in years],
                mode="lines+markers+text", name="☀️ PV (GW)",
                line=dict(color="#FFD700", width=2.5),
                text=[f"{BASE_SOLAR_GW+(y-2025)*solar_annual:.0f}" for y in years],
                textposition="top center", textfont=dict(color="#997700"),
            ))
            fig_cap.add_trace(go.Scatter(
                x=years, y=[BASE_WIND_GW + (y-2025)*wind_annual for y in years],
                mode="lines+markers+text", name="💨 Éolien (GW)",
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
                title="Trajectoire des capacités installées (GW)",
                yaxis=dict(title="GW (PV & Éolien)", gridcolor="#eee"),
                yaxis2=dict(title="GW (BESS)", overlaying="y", side="right", showgrid=False),
                height=300, plot_bgcolor="white",
                legend=dict(orientation="h", y=-0.35),
            )
            st.plotly_chart(fig_cap, use_container_width=True)

            # Heures de cannibalisation
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
                title="Heures de cannibalisation VRE (h/an)",
                yaxis_title="Heures", plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
                legend=dict(orientation="h", y=-0.35),
            )
            st.plotly_chart(fig_cann, use_container_width=True)

        # ── Dispatch projeté pour une année sélectionnée ─────────────────
        st.markdown("---")
        st.subheader("Dispatch horaire projeté — Zoom sur un mois")

        ca, cb = st.columns(2)
        yr_sel  = ca.select_slider("Année", options=years, value=2028, key="yr_sel")
        mo_sel2 = cb.selectbox("Mois", list(MONTH_NAMES.values()), index=3, key="mo_sel2")

        m2    = month_num[mo_sel2]
        df_p  = res[yr_sel]["df"]
        df_pm = df_p[df_p["Month"] == m2].copy()
        idx   = df_pm.index

        df_pm["Battery Discharge"] = res[yr_sel]["batt_discharge"].reindex(idx).fillna(0)
        df_pm["Battery Charging"]  = -res[yr_sel]["batt_charge"].reindex(idx).fillna(0)
        curtm = res[yr_sel]["curtailment"].reindex(idx).fillna(0)
        ccgtm = res[yr_sel]["ccgt_dispatch"].reindex(idx).fillna(0)

        techs_proj = [t for t in DISPATCH_TECHS + ["Battery Discharge"]
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
            mode="lines", name="Demande",
            line=dict(color="black", width=2, dash="dot"),
            hovertemplate="<b>Demande</b>: %{y:.0f} MW<extra></extra>"
        ))

        bess_total = BASE_BESS_GW + (yr_sel-2025)*bess_annual
        fig_proj.update_layout(
            height=500, hovermode="x unified",
            title=(f"Dispatch — {mo_sel2} {yr_sel}  |  "
                   f"+{(yr_sel-2025)*solar_annual:.0f} GW PV  •  "
                   f"+{(yr_sel-2025)*wind_annual:.0f} GW Éolien  •  "
                   f"{bess_total:.1f} GW BESS"),
            xaxis_title="Date", yaxis_title="MW",
            plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
        )
        st.plotly_chart(fig_proj, use_container_width=True)

        # ── État de charge BESS ───────────────────────────────────────────
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
                hovertemplate="SOC : %{y:.1f}%<extra></extra>",
            ))
            fig_soc.update_layout(
                title=f"État de charge BESS — {mo_sel2} {yr_sel}  |  {bess_total:.1f} GW / {bess_mwh:.0f} MWh",
                yaxis=dict(range=[0, 110], title="SOC (%)", gridcolor="#eee"),
                height=240, plot_bgcolor="white",
            )
            st.plotly_chart(fig_soc, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # ONGLET 5 — HEATMAPS
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("🌡️ Heatmaps — Intensité horaire par technologie")

        tech_opts = ["Solar PV", "Wind", "Demand", "Combined Cycle GT",
                     "Nuclear", "Hydro (on-flow)", "VRE_share"]
        tech_hm   = st.selectbox("Technologie / Variable", tech_opts, index=0, key="hm_tech")
        cscale_map = {
            "Solar PV":"YlOrRd", "Wind":"Blues", "Demand":"Viridis",
            "Combined Cycle GT":"Oranges", "Nuclear":"Purples",
            "Hydro (on-flow)":"BuPu", "VRE_share":"RdYlGn",
        }
        cscale = cscale_map.get(tech_hm, "Plasma")
        label  = "Part VRE (%)" if tech_hm == "VRE_share" else "MW"
        col    = tech_hm

        # ---- Heatmap Heure × Jour de l'année
        pivot_yr = df.pivot_table(values=col, index="Hour", columns="DayOfYear", aggfunc="mean")

        fig_hm1 = go.Figure(go.Heatmap(
            z=pivot_yr.values * (100 if tech_hm == "VRE_share" else 1),
            x=pivot_yr.columns, y=pivot_yr.index,
            colorscale=cscale, hoverongaps=False,
            colorbar=dict(title=label),
            hovertemplate="Jour %{x} · %{y}h → %{z:.1f}<extra></extra>",
        ))

        # Séparateurs de mois
        mo_starts = df.groupby("Month")["DayOfYear"].min().sort_index()
        for m_id, doy in mo_starts.items():
            fig_hm1.add_vline(x=doy, line_width=0.8, line_color="white", opacity=0.6)
            fig_hm1.add_annotation(
                x=doy+7, y=23.5, text=MONTH_NAMES[m_id],
                showarrow=False, font=dict(color="white", size=8)
            )

        fig_hm1.update_layout(
            title=f"Heatmap {tech_hm} — Heure × Jour de l'année 2025",
            xaxis_title="Jour de l'année", yaxis_title="Heure",
            yaxis=dict(tickmode="linear", dtick=2),
            height=430,
        )
        st.plotly_chart(fig_hm1, use_container_width=True)

        # ---- Heatmap Heure × Mois (moyenne)
        pivot_mo = df.pivot_table(values=col, index="Hour", columns="Month", aggfunc="mean")
        pivot_mo.columns = [MONTH_NAMES[c] for c in pivot_mo.columns]

        fig_hm2 = go.Figure(go.Heatmap(
            z=pivot_mo.values * (100 if tech_hm == "VRE_share" else 1),
            x=pivot_mo.columns, y=pivot_mo.index,
            colorscale=cscale, hoverongaps=False,
            colorbar=dict(title=label),
            hovertemplate="%{x} · %{y}h → %{z:.1f}<extra></extra>",
        ))
        fig_hm2.update_layout(
            title=f"Profil moyen {tech_hm} — Heure × Mois",
            xaxis_title="Mois", yaxis_title="Heure",
            yaxis=dict(tickmode="linear", dtick=2),
            height=370,
        )
        st.plotly_chart(fig_hm2, use_container_width=True)


# ============================================================================
if __name__ == "__main__":
    main()
