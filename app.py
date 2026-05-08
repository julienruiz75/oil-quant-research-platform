# app.py

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_oil_prices

from src.indicators import (
    compute_daily_returns,
    compute_rolling_volatility,
    compute_drawdowns,
    compute_summary_metrics,
)

from src.futures_curve import (
    load_futures_curve,
    enrich_futures_curve,
    compute_curve_summary,
)

from src.backtest import (
    run_backtest,
    compute_backtest_metrics,
)

from src.risk import (
    compute_var_table,
    compute_distribution_metrics,
    compute_stress_tests,
)

from src.data_quality import (
    remove_non_positive_prices,
    cap_extreme_returns,
)

from src.market_drivers import (
    load_market_drivers,
    compute_driver_returns,
    compute_correlation_with_oil,
    compute_rolling_correlations,
    build_market_drivers_summary,
)

from src.volatility_model import (
    build_volatility_dataset,
    train_volatility_model,
    predict_latest_volatility,
)

from src.regime_detection import (
    compute_market_regime_features,
    detect_market_regimes,
    compute_regime_summary,
    get_latest_regime,
)

from src.signal_engine import (
    compute_research_signal,
    build_signal_breakdown,
)

from src.signal_backtest import (
    build_historical_signal_backtest,
    compute_signal_backtest_metrics,
    compute_signal_distribution,
)

from src.monte_carlo_simulator import (
    run_monte_carlo_simulation,
    build_monte_carlo_percentiles,
    compute_monte_carlo_summary,
    compute_monte_carlo_risk_table,
    sample_simulation_paths,
)

from src.research_summary import (
    build_research_summary,
    build_research_summary_table,
    format_research_summary_table,
)

from src.report_generator import generate_research_report_html


# =========================
# CONFIGURATION DE LA PAGE
# =========================

st.set_page_config(
    page_title="Oil Quant Research Platform",
    layout="wide",
)


# =========================
# STYLE CSS PERSONNALISÉ
# =========================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0B0F19 0%, #111827 45%, #020617 100%);
            color: #F9FAFB;
        }

        header[data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        section[data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid rgba(212, 175, 55, 0.20);
        }

        section[data-testid="stSidebar"] label {
            color: #E5E7EB;
            font-weight: 500;
        }

        h1 {
            color: #F9FAFB;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        h2, h3 {
            color: #F9FAFB;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        p, li {
            color: #D1D5DB;
            font-size: 1rem;
            line-height: 1.6;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.95), rgba(3, 7, 18, 0.95));
            border: 1px solid rgba(212, 175, 55, 0.25);
            border-radius: 18px;
            padding: 22px 24px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
            min-height: 135px;
            margin-bottom: 14px;
        }

        .metric-label {
            color: #9CA3AF;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #F9FAFB;
            font-size: 2.05rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 4px;
        }

        .metric-caption {
            color: #D4AF37;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .info-card {
            background: rgba(17, 24, 39, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            padding: 22px 26px;
            margin-bottom: 22px;
        }

        .summary-card {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(2, 6, 23, 0.95));
            border: 1px solid rgba(212, 175, 55, 0.30);
            border-radius: 22px;
            padding: 26px 30px;
            margin-top: 18px;
            margin-bottom: 26px;
            box-shadow: 0 22px 55px rgba(0, 0, 0, 0.42);
        }

        .summary-title {
            color: #F9FAFB;
            font-size: 1.35rem;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .summary-text {
            color: #D1D5DB;
            font-size: 1.02rem;
            line-height: 1.75;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 16px;
            overflow: hidden;
        }

        div[data-testid="stMarkdownContainer"] {
            color: #F9FAFB;
        }

        div[data-testid="stTabs"] div[role="tablist"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem 0.55rem;
            overflow-x: visible !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.20);
            padding-bottom: 0.70rem;
            margin-bottom: 1.20rem;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            background: rgba(17, 24, 39, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            color: #D1D5DB;
            font-weight: 600;
            white-space: nowrap;
            margin-bottom: 0.20rem;
        }

        div[data-testid="stTabs"] button[role="tab"]:hover {
            background: rgba(212, 175, 55, 0.12);
            border: 1px solid rgba(212, 175, 55, 0.35);
            color: #F9FAFB;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(212, 175, 55, 0.22), rgba(249, 115, 22, 0.16));
            color: #F9FAFB;
            border: 1px solid rgba(212, 175, 55, 0.70);
            box-shadow: 0 0 18px rgba(212, 175, 55, 0.15);
        }

        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# FONCTIONS D'AFFICHAGE
# =========================

def metric_card(label: str, value: str, caption: str) -> None:
    """
    Affiche une carte KPI personnalisée.
    """

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_figure(fig):
    """
    Applique un thème sombre professionnel aux graphiques Plotly.
    """

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.35)",
        font=dict(
            family="Arial",
            color="#F9FAFB",
        ),
        title=dict(
            font=dict(size=20, color="#F9FAFB"),
            x=0.02,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0)",
            font=dict(color="#D1D5DB"),
        ),
        margin=dict(l=20, r=20, t=60, b=30),
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.15)",
        zerolinecolor="rgba(148,163,184,0.15)",
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.15)",
        zerolinecolor="rgba(148,163,184,0.15)",
    )

    return fig


def format_metric_value(metric_name: str, value) -> str:
    """
    Formate proprement une métrique de backtest.
    """

    if pd.isna(value):
        return "-"

    percentage_metrics = [
        "Total Return",
        "Annualized Volatility",
        "Maximum Drawdown",
        "Average Exposure",
        "Average Daily Turnover",
    ]

    if metric_name in percentage_metrics:
        return f"{value:.2%}"

    if metric_name == "Sharpe Ratio":
        return f"{value:.2f}"

    return f"{value:.2f}"


def format_percent(value) -> str:
    """
    Formate une valeur en pourcentage.
    """

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def format_price(value) -> str:
    """
    Formate une valeur en dollars.
    """

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2f} $"


def format_monte_carlo_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Formate le tableau de résumé Monte Carlo.
    """

    if summary is None or summary.empty:
        return pd.DataFrame()

    formatted_summary = summary.copy()

    price_metrics = [
        "Current Price",
        "Mean Simulated Price",
        "Median Simulated Price",
        "5th Percentile Price",
        "95th Percentile Price",
    ]

    percentage_metrics = [
        "Mean Simulated Return",
        "Median Simulated Return",
        "Probability of Loss",
        "Probability of Gain",
    ]

    def format_row(row):
        if row["Metric"] in price_metrics:
            return f"{row['Value']:.2f} $"

        if row["Metric"] in percentage_metrics:
            return f"{row['Value']:.2%}"

        return f"{row['Value']:.2f}"

    formatted_summary["Value"] = formatted_summary.apply(format_row, axis=1)

    return formatted_summary


def format_monte_carlo_risk_table(risk_table: pd.DataFrame) -> pd.DataFrame:
    """
    Formate le tableau de risque Monte Carlo.
    """

    if risk_table is None or risk_table.empty:
        return pd.DataFrame()

    formatted_risk_table = risk_table.copy()

    formatted_risk_table["Confidence Level"] = formatted_risk_table[
        "Confidence Level"
    ].map(lambda x: f"{x:.0%}")

    formatted_risk_table["Monte Carlo VaR"] = formatted_risk_table[
        "Monte Carlo VaR"
    ].map(lambda x: f"{x:.2%}")

    formatted_risk_table["Monte Carlo CVaR"] = formatted_risk_table[
        "Monte Carlo CVaR"
    ].map(lambda x: f"{x:.2%}")

    formatted_risk_table["VaR Price Impact"] = formatted_risk_table[
        "VaR Price Impact"
    ].map(lambda x: f"{x:.2f} $")

    formatted_risk_table["CVaR Price Impact"] = formatted_risk_table[
        "CVaR Price Impact"
    ].map(lambda x: f"{x:.2f} $")

    return formatted_risk_table


# =========================
# HEADER PRINCIPAL
# =========================

st.markdown(
    """
    <div style="margin-bottom: 2rem;">
        <h1>Oil Quant Research Platform</h1>
        <p style="font-size: 1.1rem; color: #D1D5DB; max-width: 900px;">
            Plateforme Python d'analyse quantitative du marché pétrolier :
            WTI, Brent, spread, rendements, volatilité, drawdowns,
            courbe futures, drivers macro, machine learning, Monte Carlo,
            régimes de marché, signal engine, executive summary,
            backtest dynamique, gestion du risque et contrôle qualité des données.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR
# =========================

st.sidebar.header("Paramètres")

start_date = st.sidebar.date_input(
    "Date de début",
    value=pd.to_datetime("2020-01-01"),
)

end_date = st.sidebar.date_input(
    "Date de fin",
    value=pd.Timestamp.today(),
)

vol_window = st.sidebar.slider(
    "Fenêtre de volatilité glissante",
    min_value=10,
    max_value=120,
    value=30,
    step=5,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Market Drivers")

rolling_corr_window = st.sidebar.slider(
    "Fenêtre de corrélation glissante",
    min_value=20,
    max_value=180,
    value=60,
    step=10,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Volatility Forecasting")

forecast_horizon = st.sidebar.slider(
    "Horizon de prédiction de volatilité",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Monte Carlo Lab")

monte_carlo_horizon = st.sidebar.slider(
    "Horizon Monte Carlo",
    min_value=10,
    max_value=180,
    value=60,
    step=10,
)

monte_carlo_simulations = st.sidebar.slider(
    "Nombre de simulations",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000,
)

monte_carlo_display_paths = st.sidebar.slider(
    "Trajectoires affichées",
    min_value=20,
    max_value=300,
    value=100,
    step=20,
)

use_ml_volatility_for_monte_carlo = st.sidebar.checkbox(
    "Utiliser la volatilité prédite",
    value=True,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Signal Engine")

target_signal_volatility = st.sidebar.slider(
    "Volatilité cible",
    min_value=0.10,
    max_value=0.60,
    value=0.25,
    step=0.05,
)

max_signal_exposure = st.sidebar.slider(
    "Exposition maximale",
    min_value=0.10,
    max_value=1.00,
    value=1.00,
    step=0.10,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Risk Data Quality")

risk_data_mode = st.sidebar.selectbox(
    "Mode de données pour le Risk Dashboard",
    options=[
        "Cleaned Data",
        "Raw Data",
    ],
    index=0,
)

st.sidebar.caption(
    "Cleaned Data supprime les prix négatifs et limite les rendements extrêmes."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Backtest")

short_ma_window = st.sidebar.slider(
    "Moyenne mobile courte",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
)

long_ma_window = st.sidebar.slider(
    "Moyenne mobile longue",
    min_value=100,
    max_value=300,
    value=200,
    step=10,
)

max_strategy_volatility = st.sidebar.slider(
    "Volatilité maximum autorisée",
    min_value=0.10,
    max_value=1.50,
    value=0.60,
    step=0.05,
)

transaction_cost = st.sidebar.slider(
    "Frais de transaction",
    min_value=0.0,
    max_value=0.005,
    value=0.0005,
    step=0.0005,
    format="%.4f",
)


# Conversion des dates au format texte pour yfinance
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")


# Chemin du fichier CSV de futures curve
FUTURES_CURVE_PATH = Path("data/raw/wti_futures_curve.csv")


# =========================
# CHARGEMENT DES DONNÉES
# =========================

try:
    prices = load_oil_prices(start_date_str, end_date_str)

    if "Brent" in prices.columns and "WTI" in prices.columns:
        prices["Brent-WTI Spread"] = prices["Brent"] - prices["WTI"]

    oil_prices = prices[["WTI", "Brent"]].copy()

    returns = compute_daily_returns(oil_prices)

    rolling_volatility = compute_rolling_volatility(
        returns=returns,
        window=vol_window,
    )

    drawdowns = compute_drawdowns(oil_prices)

    summary_metrics = compute_summary_metrics(
        prices=oil_prices,
        returns=returns,
    )

    # =========================
    # FUTURES CURVE
    # =========================

    if FUTURES_CURVE_PATH.exists():
        futures_curve = load_futures_curve(str(FUTURES_CURVE_PATH))
        futures_curve = enrich_futures_curve(futures_curve)
        curve_summary = compute_curve_summary(futures_curve)
        curve_regime = curve_summary["regime"]
    else:
        futures_curve = pd.DataFrame()
        curve_summary = {}
        curve_regime = None

    # =========================
    # MARKET DRIVERS
    # =========================

    market_drivers = load_market_drivers(start_date_str, end_date_str)

    if not market_drivers.empty:
        driver_returns = compute_driver_returns(market_drivers)

        correlation_table = compute_correlation_with_oil(
            oil_returns=returns["WTI"],
            driver_returns=driver_returns,
        )

        rolling_correlations = compute_rolling_correlations(
            oil_returns=returns["WTI"],
            driver_returns=driver_returns,
            window=rolling_corr_window,
        )

        drivers_summary = build_market_drivers_summary(
            drivers=market_drivers,
            driver_returns=driver_returns,
        )
    else:
        driver_returns = pd.DataFrame()
        correlation_table = pd.DataFrame()
        rolling_correlations = pd.DataFrame()
        drivers_summary = pd.DataFrame()

    # =========================
    # DONNÉES NETTOYÉES POUR LE RISQUE
    # =========================

    cleaned_oil_prices = remove_non_positive_prices(oil_prices)

    cleaned_returns = compute_daily_returns(cleaned_oil_prices)

    capped_cleaned_returns = cap_extreme_returns(
        returns=cleaned_returns,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )

    if risk_data_mode == "Cleaned Data":
        risk_returns = capped_cleaned_returns.copy()
        risk_mode_description = (
            "Cleaned Data : les prix inférieurs ou égaux à zéro sont supprimés "
            "et les rendements extrêmes sont limités aux quantiles 1% et 99%."
        )
    else:
        risk_returns = returns.copy()
        risk_mode_description = (
            "Raw Data : les données historiques sont utilisées telles quelles, "
            "y compris l'épisode du WTI négatif en avril 2020."
        )

    # =========================
    # RISQUE GLOBAL
    # =========================

    wti_returns = risk_returns["WTI"].dropna()
    current_wti_price = oil_prices["WTI"].dropna().iloc[-1]

    var_table = compute_var_table(wti_returns)
    distribution_metrics = compute_distribution_metrics(wti_returns)
    stress_tests = compute_stress_tests(current_wti_price)

    # =========================
    # VOLATILITY FORECASTING
    # =========================

    volatility_prices = cleaned_oil_prices["WTI"].dropna()

    volatility_dataset = build_volatility_dataset(
        prices=volatility_prices,
        horizon=forecast_horizon,
    )

    if len(volatility_dataset) > 150:
        (
            volatility_model,
            volatility_predictions,
            volatility_metrics,
            volatility_feature_importance,
            volatility_test_data,
        ) = train_volatility_model(volatility_dataset)

        latest_volatility_prediction = predict_latest_volatility(
            model=volatility_model,
            dataset=volatility_dataset,
        )

    else:
        volatility_model = None
        volatility_predictions = pd.DataFrame()
        volatility_metrics = pd.DataFrame()
        volatility_feature_importance = pd.DataFrame()
        volatility_test_data = pd.DataFrame()
        latest_volatility_prediction = None

    # =========================
    # MONTE CARLO LAB
    # =========================

    monte_carlo_current_price = cleaned_oil_prices["WTI"].dropna().iloc[-1]
    monte_carlo_returns = capped_cleaned_returns["WTI"].dropna()

    if (
        use_ml_volatility_for_monte_carlo
        and latest_volatility_prediction is not None
        and latest_volatility_prediction > 0
    ):
        monte_carlo_volatility_override = latest_volatility_prediction
        monte_carlo_volatility_source = "Volatilité prédite par machine learning"
    else:
        monte_carlo_volatility_override = None
        monte_carlo_volatility_source = "Volatilité historique réalisée"

    monte_carlo_paths = run_monte_carlo_simulation(
        current_price=monte_carlo_current_price,
        returns=monte_carlo_returns,
        horizon_days=monte_carlo_horizon,
        n_simulations=monte_carlo_simulations,
        annualized_volatility_override=monte_carlo_volatility_override,
        random_seed=42,
    )

    monte_carlo_percentiles = build_monte_carlo_percentiles(
        simulation_paths=monte_carlo_paths,
    )

    monte_carlo_summary = compute_monte_carlo_summary(
        simulation_paths=monte_carlo_paths,
        current_price=monte_carlo_current_price,
    )

    monte_carlo_risk_table = compute_monte_carlo_risk_table(
        simulation_paths=monte_carlo_paths,
        current_price=monte_carlo_current_price,
    )

    monte_carlo_sample_paths = sample_simulation_paths(
        simulation_paths=monte_carlo_paths,
        n_paths=monte_carlo_display_paths,
        random_seed=42,
    )

    # =========================
    # MARKET REGIME DETECTION
    # =========================

    if not market_drivers.empty and "VIX" in market_drivers.columns:
        vix_series = market_drivers["VIX"]
    else:
        vix_series = None

    if not driver_returns.empty and "DXY" in driver_returns.columns:
        dxy_returns = driver_returns["DXY"]
    else:
        dxy_returns = None

    if (
        "WTI" in cleaned_oil_prices.columns
        and "WTI" in cleaned_returns.columns
        and not cleaned_oil_prices["WTI"].dropna().empty
        and not cleaned_returns["WTI"].dropna().empty
    ):
        regime_features = compute_market_regime_features(
            prices=cleaned_oil_prices["WTI"].dropna(),
            returns=cleaned_returns["WTI"].dropna(),
            vix_series=vix_series,
            dxy_returns=dxy_returns,
        )

        if not regime_features.empty:
            market_regimes = detect_market_regimes(regime_features)
            regime_summary = compute_regime_summary(market_regimes)
            latest_regime_info = get_latest_regime(market_regimes)
        else:
            market_regimes = pd.DataFrame()
            regime_summary = pd.DataFrame()
            latest_regime_info = None

    else:
        regime_features = pd.DataFrame()
        market_regimes = pd.DataFrame()
        regime_summary = pd.DataFrame()
        latest_regime_info = None

    # =========================
    # SIGNAL ENGINE
    # =========================

    if latest_regime_info is not None:
        signal_summary = compute_research_signal(
            latest_regime_info=latest_regime_info,
            predicted_volatility=latest_volatility_prediction,
            curve_regime=curve_regime,
            target_volatility=target_signal_volatility,
            max_exposure=max_signal_exposure,
        )

        signal_breakdown = build_signal_breakdown(signal_summary)

    else:
        signal_summary = None
        signal_breakdown = pd.DataFrame()

    # =========================
    # SIGNAL ENGINE BACKTEST
    # =========================

    if not market_regimes.empty:
        signal_engine_backtest = build_historical_signal_backtest(
            market_regimes=market_regimes,
            curve_regime=curve_regime,
            target_volatility=target_signal_volatility,
            max_exposure=max_signal_exposure,
            transaction_cost=transaction_cost,
        )

        signal_backtest_metrics = compute_signal_backtest_metrics(
            signal_engine_backtest
        )

        signal_distribution = compute_signal_distribution(
            signal_engine_backtest
        )

    else:
        signal_engine_backtest = pd.DataFrame()
        signal_backtest_metrics = pd.DataFrame()
        signal_distribution = pd.DataFrame()

    # =========================
    # EXECUTIVE SUMMARY
    # =========================

    executive_summary = build_research_summary(
        signal_summary=signal_summary,
        latest_regime_info=latest_regime_info,
        monte_carlo_summary=monte_carlo_summary,
        monte_carlo_risk_table=monte_carlo_risk_table,
        var_table=var_table,
        distribution_metrics=distribution_metrics,
    )

    executive_summary_table = build_research_summary_table(
        executive_summary
    )

    formatted_executive_summary_table = format_research_summary_table(
        executive_summary_table
    )

    # =========================
    # ONGLETS
    # =========================

    (
        tab_executive,
        tab_market,
        tab_returns_risk,
        tab_curve,
        tab_drivers,
        tab_volatility,
        tab_monte_carlo,
        tab_regime,
        tab_signal,
        tab_backtest,
        tab_risk_dashboard,
        tab_data,
    ) = st.tabs([
        "Executive Summary",
        "Market Overview",
        "Returns & Risk",
        "Futures Curve",
        "Market Drivers",
        "Volatility Forecasting",
        "Monte Carlo Lab",
        "Market Regime",
        "Signal Engine",
        "Strategy Backtest",
        "Risk Dashboard",
        "Data",
    ])

    # =========================
    # ONGLET 0 : EXECUTIVE SUMMARY
    # =========================

    with tab_executive:

        st.markdown(
            """
            <div class="info-card">
                <h2>Executive Summary</h2>
                <p>
                    Cette page synthétise automatiquement les résultats clés du dashboard :
                    signal de recherche, régime de marché, niveau de risque,
                    scénario Monte Carlo et conclusion analytique.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        report_html = generate_research_report_html(
            executive_summary=executive_summary,
            executive_summary_table=formatted_executive_summary_table,
            signal_breakdown=signal_breakdown,
            signal_backtest_metrics=signal_backtest_metrics,
            monte_carlo_summary=format_monte_carlo_summary(monte_carlo_summary),
            monte_carlo_risk_table=format_monte_carlo_risk_table(monte_carlo_risk_table),
            project_title="Oil Quant Research Platform",
            start_date=start_date_str,
            end_date=end_date_str,
            monte_carlo_horizon=monte_carlo_horizon,
            risk_data_mode=risk_data_mode,
        )

        st.download_button(
            label="Download Research Report HTML",
            data=report_html,
            file_name="oil_quant_research_report.html",
            mime="text/html",
            use_container_width=True,
        )

        st.markdown("---")

        signal = executive_summary["signal"]
        signal_description = executive_summary["signal_description"]
        total_score = executive_summary["total_score"]
        target_exposure = executive_summary["target_exposure"]
        regime = executive_summary["regime"]
        risk_level = executive_summary["risk_level"]
        probability_of_loss = executive_summary["probability_of_loss"]
        median_simulated_price = executive_summary["median_simulated_price"]
        historical_var_95 = executive_summary["historical_var_95"]
        monte_carlo_var_95 = executive_summary["monte_carlo_var_95"]
        interpretation = executive_summary["interpretation"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                label="Research Signal",
                value=signal,
                caption=signal_description,
            )

        with col2:
            metric_card(
                label="Market Regime",
                value=regime,
                caption="Régime de marché actuel",
            )

        with col3:
            metric_card(
                label="Risk Level",
                value=risk_level,
                caption="Classification globale du risque",
            )

        with col4:
            metric_card(
                label="Target Exposure",
                value=format_percent(target_exposure),
                caption=f"Score total : {total_score}",
            )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                label="Median MC Price",
                value=format_price(median_simulated_price),
                caption=f"Prix médian simulé à {monte_carlo_horizon} jours",
            )

        with col2:
            metric_card(
                label="Probability of Loss",
                value=format_percent(probability_of_loss),
                caption="Probabilité simulée de perte",
            )

        with col3:
            metric_card(
                label="Historical VaR 95%",
                value=format_percent(historical_var_95),
                caption="Risque historique journalier",
            )

        with col4:
            metric_card(
                label="Monte Carlo VaR 95%",
                value=format_percent(monte_carlo_var_95),
                caption="Risque simulé à horizon choisi",
            )

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-title">Lecture analytique automatique</div>
                <div class="summary-text">{interpretation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Synthèse détaillée")

        st.dataframe(
            formatted_executive_summary_table,
            use_container_width=True,
        )

        st.subheader("Indicateurs clés normalisés")

        summary_chart_data = pd.DataFrame({
            "Metric": [
                "Momentum 20D",
                "Drawdown",
                "Historical VaR 95%",
                "Monte Carlo VaR 95%",
                "Annualized Volatility",
                "Probability of Loss",
            ],
            "Value": [
                executive_summary["momentum_20d"],
                executive_summary["drawdown"],
                executive_summary["historical_var_95"],
                executive_summary["monte_carlo_var_95"],
                executive_summary["annualized_volatility"],
                executive_summary["probability_of_loss"],
            ],
        })

        summary_chart_data = summary_chart_data.dropna()

        if not summary_chart_data.empty:
            fig_summary = px.bar(
                summary_chart_data,
                x="Metric",
                y="Value",
                title="Vue synthétique des métriques clés",
                labels={
                    "Metric": "Métrique",
                    "Value": "Valeur",
                },
            )

            fig_summary.update_traces(
                marker_color="#D4AF37",
                opacity=0.9,
            )

            fig_summary = style_plotly_figure(fig_summary)

            st.plotly_chart(fig_summary, use_container_width=True)

    # =========================
    # ONGLET 1 : MARKET OVERVIEW
    # =========================

    with tab_market:

        st.markdown(
            """
            <div class="info-card">
                <h2>Market Overview</h2>
                <p>
                    Cette section présente les prix du WTI, du Brent et le spread Brent-WTI.
                    Le spread mesure l'écart entre les deux références majeures du marché pétrolier.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""
        Le spread Brent-WTI est défini par :

        $Spread_t = Brent_t - WTI_t$

        où :

        - $Brent_t$ est le prix du Brent au jour $t$ ;
        - $WTI_t$ est le prix du WTI au jour $t$.
        """)

        latest_wti = prices["WTI"].dropna().iloc[-1]
        latest_brent = prices["Brent"].dropna().iloc[-1]
        latest_spread = prices["Brent-WTI Spread"].dropna().iloc[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card(
                label="WTI",
                value=f"{latest_wti:.2f} $",
                caption="Dernier prix disponible",
            )

        with col2:
            metric_card(
                label="Brent",
                value=f"{latest_brent:.2f} $",
                caption="Dernier prix disponible",
            )

        with col3:
            metric_card(
                label="Spread Brent-WTI",
                value=f"{latest_spread:.2f} $",
                caption="Brent moins WTI",
            )

        st.subheader("Prix historiques du WTI et du Brent")

        fig_prices = px.line(
            oil_prices,
            x=oil_prices.index,
            y=["WTI", "Brent"],
            title="Évolution des prix du WTI et du Brent",
            labels={
                "value": "Prix",
                "index": "Date",
                "variable": "Actif",
            },
            color_discrete_map={
                "WTI": "#D4AF37",
                "Brent": "#38BDF8",
            },
        )

        fig_prices = style_plotly_figure(fig_prices)

        st.plotly_chart(fig_prices, use_container_width=True)

        st.subheader("Spread Brent-WTI")

        fig_spread = px.line(
            prices,
            x=prices.index,
            y="Brent-WTI Spread",
            title="Évolution du spread Brent-WTI",
            labels={
                "Brent-WTI Spread": "Spread",
                "index": "Date",
            },
        )

        fig_spread.update_traces(line_color="#F97316")
        fig_spread = style_plotly_figure(fig_spread)

        st.plotly_chart(fig_spread, use_container_width=True)

    # =========================
    # ONGLET 2 : RETURNS & RISK
    # =========================

    with tab_returns_risk:

        st.markdown(
            """
            <div class="info-card">
                <h2>Returns & Risk</h2>
                <p>
                    Cette section analyse les rendements, la volatilité annualisée et les drawdowns.
                    Elle permet d'évaluer le risque historique du WTI et du Brent.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""
        Le rendement journalier est défini par :

        $r_t = \\frac{P_t}{P_{t-1}} - 1$

        La volatilité annualisée est définie par :

        $\\sigma_{annuelle} = \\sigma_{journalière} \\times \\sqrt{252}$

        Le drawdown est défini par :

        $DD_t = \\frac{P_t}{\\max(P_0, ..., P_t)} - 1$
        """)

        st.subheader("Résumé quantitatif")

        formatted_metrics = summary_metrics.copy()

        formatted_metrics["Prix actuel"] = formatted_metrics["Prix actuel"].map(lambda x: f"{x:.2f} $")
        formatted_metrics["Performance totale"] = formatted_metrics["Performance totale"].map(lambda x: f"{x:.2%}")
        formatted_metrics["Volatilité annualisée"] = formatted_metrics["Volatilité annualisée"].map(lambda x: f"{x:.2%}")
        formatted_metrics["Maximum drawdown"] = formatted_metrics["Maximum drawdown"].map(lambda x: f"{x:.2%}")

        st.dataframe(formatted_metrics, use_container_width=True)

        st.subheader("Rendements journaliers")

        fig_returns = px.line(
            returns,
            x=returns.index,
            y=["WTI", "Brent"],
            title="Rendements journaliers du WTI et du Brent",
            labels={
                "value": "Rendement journalier",
                "index": "Date",
                "variable": "Actif",
            },
            color_discrete_map={
                "WTI": "#D4AF37",
                "Brent": "#38BDF8",
            },
        )

        fig_returns = style_plotly_figure(fig_returns)

        st.plotly_chart(fig_returns, use_container_width=True)

        st.subheader("Volatilité glissante annualisée")

        fig_vol = px.line(
            rolling_volatility,
            x=rolling_volatility.index,
            y=["WTI", "Brent"],
            title=f"Volatilité glissante annualisée sur {vol_window} jours",
            labels={
                "value": "Volatilité annualisée",
                "index": "Date",
                "variable": "Actif",
            },
            color_discrete_map={
                "WTI": "#D4AF37",
                "Brent": "#38BDF8",
            },
        )

        fig_vol = style_plotly_figure(fig_vol)

        st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("Drawdowns")

        fig_drawdowns = px.line(
            drawdowns,
            x=drawdowns.index,
            y=["WTI", "Brent"],
            title="Drawdowns du WTI et du Brent",
            labels={
                "value": "Drawdown",
                "index": "Date",
                "variable": "Actif",
            },
            color_discrete_map={
                "WTI": "#D4AF37",
                "Brent": "#38BDF8",
            },
        )

        fig_drawdowns = style_plotly_figure(fig_drawdowns)

        st.plotly_chart(fig_drawdowns, use_container_width=True)

    # =========================
    # ONGLET 3 : FUTURES CURVE
    # =========================

    with tab_curve:

        st.markdown(
            """
            <div class="info-card">
                <h2>Futures Curve</h2>
                <p>
                    Cette section analyse la courbe des contrats futures sur le pétrole WTI.
                    La forme de la courbe donne une information importante sur les conditions de marché :
                    stockage, rareté du pétrole disponible, anticipations et coût de portage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""
        Une courbe de futures peut être en **contango** ou en **backwardation**.

        **Contango** :

        $F_{long} > F_{front}$

        Cela signifie que les contrats longs sont plus chers que le contrat proche.

        **Backwardation** :

        $F_{long} < F_{front}$

        Cela signifie que les contrats longs sont moins chers que le contrat proche.

        Le roll yield annualisé simplifié est défini par :

        $RollYield = -Slope \\times \\frac{12}{N}$

        avec :

        $Slope = \\frac{F_{long}}{F_{front}} - 1$
        """)

        if futures_curve.empty:
            st.warning(
                "Le fichier data/raw/wti_futures_curve.csv est introuvable. "
                "Crée ce fichier avant d'afficher la Futures Curve."
            )

        else:
            regime = curve_summary["regime"]
            slope = curve_summary["slope"]
            roll_yield = curve_summary["roll_yield"]

            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card(
                    label="Curve Regime",
                    value=regime,
                    caption="Contango ou backwardation",
                )

            with col2:
                metric_card(
                    label="Curve Slope",
                    value=f"{slope:.2%}",
                    caption="Écart entre contrat long et front",
                )

            with col3:
                metric_card(
                    label="Annualized Roll Yield",
                    value=f"{roll_yield:.2%}",
                    caption="Estimation simplifiée",
                )

            st.subheader("Courbe des futures WTI")

            fig_curve = px.line(
                futures_curve,
                x="maturity",
                y="price",
                markers=True,
                text="contract",
                title="WTI Futures Curve",
                labels={
                    "maturity": "Maturité",
                    "price": "Prix future",
                    "contract": "Contrat",
                },
            )

            fig_curve.update_traces(
                line_color="#D4AF37",
                marker=dict(size=9),
                textposition="top center",
            )

            fig_curve = style_plotly_figure(fig_curve)

            st.plotly_chart(fig_curve, use_container_width=True)

            st.subheader("Tableau de la courbe futures")

            formatted_curve = futures_curve.copy()

            formatted_curve["maturity"] = formatted_curve["maturity"].dt.strftime("%Y-%m-%d")
            formatted_curve["price"] = formatted_curve["price"].map(lambda x: f"{x:.2f} $")
            formatted_curve["price_change_vs_front"] = formatted_curve["price_change_vs_front"].map(lambda x: f"{x:.2f} $")
            formatted_curve["pct_change_vs_front"] = formatted_curve["pct_change_vs_front"].map(lambda x: f"{x:.2%}")

            st.dataframe(formatted_curve, use_container_width=True)

    # =========================
    # ONGLET 4 : MARKET DRIVERS
    # =========================

    with tab_drivers:

        st.markdown(
            """
            <div class="info-card">
                <h2>Market Drivers</h2>
                <p>
                    Cette section relie le pétrole à plusieurs facteurs de marché :
                    volatilité, dollar américain, actions, taux et secteur énergie.
                    L'objectif est de comprendre quels drivers sont historiquement
                    les plus corrélés aux mouvements du WTI.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""
        Les corrélations sont calculées sur les rendements journaliers :

        $Corr(WTI, Driver) = \\frac{Cov(r_{WTI}, r_{Driver})}{\\sigma_{WTI}\\sigma_{Driver}}$

        Une corrélation positive signifie que le driver a tendance à monter lorsque le WTI monte.

        Une corrélation négative signifie que le driver a tendance à baisser lorsque le WTI monte.
        """)

        if market_drivers.empty:
            st.warning(
                "Aucune donnée de market drivers n'a été récupérée. "
                "Vérifie ta connexion internet ou les tickers Yahoo Finance."
            )

        else:
            st.subheader("Résumé des drivers de marché")

            formatted_drivers_summary = drivers_summary.copy()

            formatted_drivers_summary["Latest Value"] = formatted_drivers_summary["Latest Value"].map(lambda x: f"{x:.2f}")
            formatted_drivers_summary["Total Change"] = formatted_drivers_summary["Total Change"].map(lambda x: f"{x:.2%}")
            formatted_drivers_summary["Annualized Volatility"] = formatted_drivers_summary["Annualized Volatility"].map(lambda x: f"{x:.2%}")

            st.dataframe(formatted_drivers_summary, use_container_width=True)

            st.subheader("Corrélation avec le WTI")

            formatted_correlation_table = correlation_table.copy()
            formatted_correlation_table["Correlation with WTI"] = formatted_correlation_table["Correlation with WTI"].map(lambda x: f"{x:.2f}")

            st.dataframe(formatted_correlation_table, use_container_width=True)

            fig_corr = px.bar(
                correlation_table,
                x="Driver",
                y="Correlation with WTI",
                title="Corrélations historiques avec le WTI",
                labels={
                    "Driver": "Driver",
                    "Correlation with WTI": "Corrélation avec le WTI",
                },
            )

            fig_corr.update_traces(
                marker_color="#D4AF37",
                opacity=0.9,
            )

            fig_corr = style_plotly_figure(fig_corr)

            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Niveaux historiques des drivers")

            selected_drivers = st.multiselect(
                "Choisis les drivers à afficher",
                options=list(market_drivers.columns),
                default=list(market_drivers.columns[:3]),
            )

            if selected_drivers:
                normalized_drivers = market_drivers[selected_drivers].copy()
                normalized_drivers = normalized_drivers / normalized_drivers.iloc[0] * 100

                fig_drivers = px.line(
                    normalized_drivers,
                    x=normalized_drivers.index,
                    y=selected_drivers,
                    title="Drivers normalisés base 100",
                    labels={
                        "value": "Base 100",
                        "index": "Date",
                        "variable": "Driver",
                    },
                )

                fig_drivers = style_plotly_figure(fig_drivers)

                st.plotly_chart(fig_drivers, use_container_width=True)

            st.subheader("Corrélations glissantes avec le WTI")

            if rolling_correlations.empty:
                st.warning(
                    "Pas assez de données pour calculer les corrélations glissantes avec la fenêtre sélectionnée."
                )

            else:
                selected_rolling_driver = st.selectbox(
                    "Driver pour la corrélation glissante",
                    options=list(rolling_correlations.columns),
                    index=0,
                )

                rolling_corr_series = rolling_correlations[[selected_rolling_driver]].copy()

                fig_rolling_corr = px.line(
                    rolling_corr_series,
                    x=rolling_corr_series.index,
                    y=selected_rolling_driver,
                    title=f"Corrélation glissante WTI / {selected_rolling_driver} sur {rolling_corr_window} jours",
                    labels={
                        selected_rolling_driver: "Corrélation",
                        "index": "Date",
                    },
                )

                fig_rolling_corr.update_traces(line_color="#F97316")

                fig_rolling_corr = style_plotly_figure(fig_rolling_corr)

                st.plotly_chart(fig_rolling_corr, use_container_width=True)

    # =========================
    # ONGLET 5 : VOLATILITY FORECASTING
    # =========================

    with tab_volatility:

        st.markdown(
            """
            <div class="info-card">
                <h2>Volatility Forecasting</h2>
                <p>
                    Cette section utilise un modèle de machine learning pour prédire
                    la volatilité future du WTI. Le modèle utilise des variables comme
                    les rendements passés, la volatilité réalisée historique et le momentum.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"""
        L'objectif est de prédire la volatilité réalisée future sur `{forecast_horizon}` jours.

        La volatilité réalisée future est définie par :

        $\\sigma_{{future,t}} = std(r_{{t+1}}, ..., r_{{t+h}}) \\times \\sqrt{{252}}$

        où :

        - $h$ est l'horizon de prédiction ;
        - $r_t$ représente le rendement journalier du WTI ;
        - $\\sqrt{{252}}$ annualise la volatilité.

        Le modèle utilisé ici est un **Random Forest Regressor**.
        """)

        if volatility_model is None or volatility_predictions.empty:
            st.warning(
                "Pas assez de données pour entraîner le modèle de volatilité. "
                "Augmente la période d'analyse."
            )

        else:
            latest_realized_volatility = volatility_predictions[
                "Realized Future Volatility"
            ].dropna().iloc[-1]

            r2_score_value = volatility_metrics.loc[
                volatility_metrics["Metric"] == "R2 Score",
                "Value",
            ].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card(
                    label="Predicted Volatility",
                    value=f"{latest_volatility_prediction:.2%}",
                    caption=f"Prévision à {forecast_horizon} jours",
                )

            with col2:
                metric_card(
                    label="Last Realized Volatility",
                    value=f"{latest_realized_volatility:.2%}",
                    caption="Volatilité réalisée future observée",
                )

            with col3:
                metric_card(
                    label="Model R2 Score",
                    value=f"{r2_score_value:.2f}",
                    caption="Qualité explicative sur test set",
                )

            st.subheader("Volatilité prédite vs volatilité réalisée")

            fig_vol_forecast = px.line(
                volatility_predictions,
                x=volatility_predictions.index,
                y=[
                    "Realized Future Volatility",
                    "Predicted Future Volatility",
                ],
                title="Prévision de volatilité future du WTI",
                labels={
                    "value": "Volatilité annualisée",
                    "index": "Date",
                    "variable": "Série",
                },
                color_discrete_map={
                    "Realized Future Volatility": "#38BDF8",
                    "Predicted Future Volatility": "#D4AF37",
                },
            )

            fig_vol_forecast = style_plotly_figure(fig_vol_forecast)

            st.plotly_chart(fig_vol_forecast, use_container_width=True)

            st.subheader("Métriques du modèle")

            formatted_volatility_metrics = volatility_metrics.copy()

            formatted_volatility_metrics["Value"] = formatted_volatility_metrics.apply(
                lambda row: (
                    f"{row['Value']:.2%}"
                    if row["Metric"] in ["MAE", "RMSE"]
                    else f"{row['Value']:.2f}"
                ),
                axis=1,
            )

            st.dataframe(formatted_volatility_metrics, use_container_width=True)

            st.subheader("Importance des variables")

            fig_feature_importance = px.bar(
                volatility_feature_importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Variables les plus importantes pour prédire la volatilité",
                labels={
                    "Importance": "Importance",
                    "Feature": "Variable",
                },
            )

            fig_feature_importance.update_traces(
                marker_color="#F97316",
                opacity=0.9,
            )

            fig_feature_importance.update_layout(
                yaxis=dict(autorange="reversed")
            )

            fig_feature_importance = style_plotly_figure(fig_feature_importance)

            st.plotly_chart(fig_feature_importance, use_container_width=True)

            st.subheader("Dernières prédictions")

            st.dataframe(volatility_predictions.tail(20), use_container_width=True)

    # =========================
    # ONGLET 6 : MONTE CARLO LAB
    # =========================

    with tab_monte_carlo:

        st.markdown(
            """
            <div class="info-card">
                <h2>Monte Carlo Scenario Lab</h2>
                <p>
                    Cette section simule plusieurs milliers de trajectoires futures possibles du WTI.
                    L'objectif est d'obtenir une distribution de scénarios, une probabilité de perte,
                    ainsi qu'une VaR et une CVaR simulées à l'horizon choisi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"""
        Le modèle utilisé est un **Geometric Brownian Motion** :

        $S_{{t+1}} = S_t \\times exp((\\mu - 0.5\\sigma^2)\\Delta t + \\sigma\\sqrt{{\\Delta t}}\\epsilon_t)$

        Paramètres utilisés :

        - Horizon : `{monte_carlo_horizon}` jours ;
        - Nombre de simulations : `{monte_carlo_simulations}` ;
        - Source de volatilité : `{monte_carlo_volatility_source}`.
        """)

        if monte_carlo_paths.empty or monte_carlo_summary.empty:
            st.warning(
                "Impossible de générer les simulations Monte Carlo. "
                "Vérifie que les données nettoyées du WTI sont disponibles."
            )

        else:
            final_prices = monte_carlo_paths.iloc[-1]
            final_returns = final_prices / monte_carlo_current_price - 1

            median_price = final_prices.median()
            probability_of_loss = (final_returns < 0).mean()

            mc_var_95 = monte_carlo_risk_table.loc[
                monte_carlo_risk_table["Confidence Level"] == 0.95,
                "Monte Carlo VaR",
            ].iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                metric_card(
                    label="Current WTI",
                    value=f"{monte_carlo_current_price:.2f} $",
                    caption="Prix initial de simulation",
                )

            with col2:
                metric_card(
                    label="Median Scenario",
                    value=f"{median_price:.2f} $",
                    caption=f"Prix médian à {monte_carlo_horizon} jours",
                )

            with col3:
                metric_card(
                    label="Probability of Loss",
                    value=f"{probability_of_loss:.2%}",
                    caption="Probabilité de finir sous le prix actuel",
                )

            with col4:
                metric_card(
                    label="Monte Carlo VaR 95%",
                    value=f"{mc_var_95:.2%}",
                    caption="Perte simulée au seuil 95%",
                )

            st.subheader("Trajectoires Monte Carlo simulées")

            fig_mc_paths = px.line(
                monte_carlo_sample_paths,
                x=monte_carlo_sample_paths.index,
                y=list(monte_carlo_sample_paths.columns),
                title="Échantillon de trajectoires simulées du WTI",
                labels={
                    "value": "Prix simulé",
                    "Day": "Jour",
                    "variable": "Simulation",
                },
            )

            fig_mc_paths.update_traces(
                opacity=0.18,
                line=dict(width=1),
                showlegend=False,
            )

            fig_mc_paths = style_plotly_figure(fig_mc_paths)

            st.plotly_chart(fig_mc_paths, use_container_width=True)

            st.subheader("Couloir de scénarios simulés")

            fig_mc_percentiles = px.line(
                monte_carlo_percentiles,
                x=monte_carlo_percentiles.index,
                y=["P5", "P25", "P50", "P75", "P95"],
                title="Percentiles des prix simulés du WTI",
                labels={
                    "value": "Prix simulé",
                    "Day": "Jour",
                    "variable": "Percentile",
                },
                color_discrete_map={
                    "P5": "#EF4444",
                    "P25": "#F97316",
                    "P50": "#D4AF37",
                    "P75": "#38BDF8",
                    "P95": "#22C55E",
                },
            )

            fig_mc_percentiles = style_plotly_figure(fig_mc_percentiles)

            st.plotly_chart(fig_mc_percentiles, use_container_width=True)

            st.subheader("Résumé des scénarios Monte Carlo")

            formatted_monte_carlo_summary = format_monte_carlo_summary(
                monte_carlo_summary
            )

            st.dataframe(formatted_monte_carlo_summary, use_container_width=True)

            st.subheader("Tableau de risque Monte Carlo")

            formatted_monte_carlo_risk_table = format_monte_carlo_risk_table(
                monte_carlo_risk_table
            )

            st.dataframe(formatted_monte_carlo_risk_table, use_container_width=True)

            st.subheader("Distribution des prix finaux simulés")

            final_prices_df = pd.DataFrame({
                "Final Simulated Price": final_prices,
            })

            fig_final_price_distribution = px.histogram(
                final_prices_df,
                x="Final Simulated Price",
                nbins=80,
                title=f"Distribution des prix simulés à {monte_carlo_horizon} jours",
                labels={
                    "Final Simulated Price": "Prix final simulé",
                    "count": "Fréquence",
                },
            )

            fig_final_price_distribution.update_traces(
                marker_color="#D4AF37",
                opacity=0.85,
            )

            fig_final_price_distribution = style_plotly_figure(
                fig_final_price_distribution
            )

            st.plotly_chart(fig_final_price_distribution, use_container_width=True)

    # =========================
    # ONGLET 7 : MARKET REGIME
    # =========================

    with tab_regime:

        st.markdown(
            """
            <div class="info-card">
                <h2>Market Regime Detection</h2>
                <p>
                    Cette section classe automatiquement le marché du pétrole en plusieurs régimes :
                    tendance haussière, tendance baissière, forte volatilité, stress de marché ou régime neutre.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if market_regimes.empty or latest_regime_info is None:
            st.warning(
                "Pas assez de données pour détecter les régimes de marché. "
                "Augmente la période d'analyse."
            )

        else:
            latest_regime = latest_regime_info["regime"]
            latest_date = pd.to_datetime(latest_regime_info["date"]).strftime("%Y-%m-%d")
            latest_momentum_20d = latest_regime_info["momentum_20d"]
            latest_realized_vol_20d = latest_regime_info["realized_vol_20d"]
            latest_drawdown = latest_regime_info["drawdown"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                metric_card("Current Regime", latest_regime, f"Dernière date : {latest_date}")

            with col2:
                metric_card("Momentum 20D", f"{latest_momentum_20d:.2%}", "Variation du WTI sur 20 jours")

            with col3:
                metric_card("Realized Vol 20D", f"{latest_realized_vol_20d:.2%}", "Volatilité annualisée 20 jours")

            with col4:
                metric_card("Current Drawdown", f"{latest_drawdown:.2%}", "Perte depuis le plus haut historique")

            if "vix_level" in latest_regime_info:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <p>
                            <strong>VIX actuel utilisé par le modèle :</strong>
                            {latest_regime_info["vix_level"]:.2f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("Résumé statistique par régime")

            formatted_regime_summary = regime_summary.copy()
            formatted_regime_summary["Frequency"] = formatted_regime_summary["Frequency"].map(lambda x: f"{x:.2%}")
            formatted_regime_summary["Mean Daily Return"] = formatted_regime_summary["Mean Daily Return"].map(lambda x: f"{x:.2%}")
            formatted_regime_summary["Annualized Volatility"] = formatted_regime_summary["Annualized Volatility"].map(lambda x: f"{x:.2%}")
            formatted_regime_summary["Average Drawdown"] = formatted_regime_summary["Average Drawdown"].map(lambda x: f"{x:.2%}")

            st.dataframe(formatted_regime_summary, use_container_width=True)

            st.subheader("Prix du WTI coloré par régime")

            regime_chart = market_regimes.reset_index()
            regime_chart.columns = ["Date"] + list(regime_chart.columns[1:])

            regime_color_map = {
                "Bullish Trend": "#22C55E",
                "Bearish Trend": "#EF4444",
                "High Volatility": "#F97316",
                "Stress Regime": "#A855F7",
                "Neutral": "#94A3B8",
            }

            fig_regime_price = px.scatter(
                regime_chart,
                x="Date",
                y="price",
                color="market_regime",
                title="WTI Market Regimes",
                labels={
                    "Date": "Date",
                    "price": "Prix WTI",
                    "market_regime": "Régime",
                },
                color_discrete_map=regime_color_map,
            )

            fig_regime_price.update_traces(marker=dict(size=5, opacity=0.80))
            fig_regime_price = style_plotly_figure(fig_regime_price)

            st.plotly_chart(fig_regime_price, use_container_width=True)

            st.subheader("Répartition des régimes")

            fig_regime_distribution = px.pie(
                regime_summary,
                names="Regime",
                values="Number of Days",
                title="Répartition historique des régimes",
                hole=0.45,
                color="Regime",
                color_discrete_map=regime_color_map,
            )

            fig_regime_distribution.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )

            fig_regime_distribution = style_plotly_figure(fig_regime_distribution)

            st.plotly_chart(fig_regime_distribution, use_container_width=True)

            st.subheader("Variables de régime")

            st.dataframe(market_regimes.tail(20), use_container_width=True)

    # =========================
    # ONGLET 8 : SIGNAL ENGINE
    # =========================

    with tab_signal:

        st.markdown(
            """
            <div class="info-card">
                <h2>Signal Engine</h2>
                <p>
                    Cette section transforme les analyses précédentes en signal de recherche.
                    Le moteur combine le régime de marché, le momentum, la volatilité prédite,
                    la courbe futures et le drawdown pour produire une exposition cible.
                    Elle backteste aussi historiquement cette logique contre un simple Buy & Hold WTI.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""
        Le Signal Engine combine plusieurs composantes :

        - **Market Regime** : tendance, stress ou neutralité ;
        - **Momentum** : direction récente du WTI ;
        - **Predicted Volatility** : volatilité future estimée par machine learning ;
        - **Futures Curve** : contango ou backwardation ;
        - **Drawdown** : distance au plus haut historique.

        Le résultat final est un signal de recherche, une exposition cible au WTI,
        puis un backtest historique de cette exposition dynamique.
        """)

        if signal_summary is None or signal_breakdown.empty:
            st.warning(
                "Le Signal Engine n'est pas disponible. "
                "Il faut d'abord disposer d'un régime de marché valide."
            )

        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card(
                    label="Research Signal",
                    value=signal_summary["signal"],
                    caption=signal_summary["signal_description"],
                )

            with col2:
                metric_card(
                    label="Total Score",
                    value=str(signal_summary["total_score"]),
                    caption="Score agrégé multi-facteurs",
                )

            with col3:
                metric_card(
                    label="Target Exposure",
                    value=f"{signal_summary['target_exposure']:.2%}",
                    caption="Exposition théorique au WTI",
                )

            st.subheader("Décomposition du signal actuel")

            st.dataframe(signal_breakdown, use_container_width=True)

            fig_signal_breakdown = px.bar(
                signal_breakdown,
                x="Component",
                y="Score",
                title="Contribution de chaque composante au signal actuel",
                labels={
                    "Component": "Composante",
                    "Score": "Score",
                },
            )

            fig_signal_breakdown.update_traces(
                marker_color="#D4AF37",
                opacity=0.9,
            )

            fig_signal_breakdown = style_plotly_figure(fig_signal_breakdown)

            st.plotly_chart(fig_signal_breakdown, use_container_width=True)

            st.subheader("Résumé du signal actuel")

            signal_table = pd.DataFrame({
                "Metric": [
                    "Signal",
                    "Signal Description",
                    "Target Exposure",
                    "Current Regime",
                    "Predicted Volatility",
                    "Futures Curve Regime",
                    "Momentum 20D",
                    "Momentum 60D",
                    "Drawdown",
                ],
                "Value": [
                    signal_summary["signal"],
                    signal_summary["signal_description"],
                    f"{signal_summary['target_exposure']:.2%}",
                    signal_summary["current_regime"],
                    (
                        f"{signal_summary['predicted_volatility']:.2%}"
                        if signal_summary["predicted_volatility"] is not None
                        else "Not available"
                    ),
                    signal_summary["curve_regime"],
                    f"{signal_summary['momentum_20d']:.2%}",
                    f"{signal_summary['momentum_60d']:.2%}",
                    f"{signal_summary['drawdown']:.2%}",
                ],
            })

            st.dataframe(signal_table, use_container_width=True)

        st.markdown("---")

        st.subheader("Backtest historique du Signal Engine")

        if signal_engine_backtest.empty or signal_backtest_metrics.empty:
            st.warning(
                "Pas assez de données pour backtester le Signal Engine."
            )

        else:
            signal_total_return = (
                signal_engine_backtest["signal_strategy_equity"].iloc[-1] - 1
            )

            buy_hold_total_return = (
                signal_engine_backtest["buy_hold_equity"].iloc[-1] - 1
            )

            average_exposure = signal_engine_backtest["position"].mean()

            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card(
                    label="Signal Strategy Return",
                    value=f"{signal_total_return:.2%}",
                    caption="Performance cumulée Signal Engine",
                )

            with col2:
                metric_card(
                    label="WTI Buy & Hold",
                    value=f"{buy_hold_total_return:.2%}",
                    caption="Performance cumulée WTI",
                )

            with col3:
                metric_card(
                    label="Average Exposure",
                    value=f"{average_exposure:.2%}",
                    caption="Exposition moyenne historique",
                )

            st.subheader("Performance cumulée : Signal Engine vs Buy & Hold")

            signal_equity_curve = signal_engine_backtest[[
                "buy_hold_equity",
                "signal_strategy_equity",
            ]].rename(
                columns={
                    "buy_hold_equity": "WTI Buy & Hold",
                    "signal_strategy_equity": "Signal Engine Strategy",
                }
            )

            fig_signal_equity = px.line(
                signal_equity_curve,
                x=signal_equity_curve.index,
                y=["WTI Buy & Hold", "Signal Engine Strategy"],
                title="Signal Engine Backtest : Buy & Hold vs Signal Strategy",
                labels={
                    "value": "Capital indexé",
                    "index": "Date",
                    "variable": "Stratégie",
                },
                color_discrete_map={
                    "WTI Buy & Hold": "#38BDF8",
                    "Signal Engine Strategy": "#D4AF37",
                },
            )

            fig_signal_equity = style_plotly_figure(fig_signal_equity)

            st.plotly_chart(fig_signal_equity, use_container_width=True)

            st.subheader("Métriques du backtest Signal Engine")

            formatted_signal_backtest_metrics = signal_backtest_metrics.copy()

            for column in ["WTI Buy & Hold", "Signal Engine Strategy"]:
                formatted_signal_backtest_metrics[column] = formatted_signal_backtest_metrics.apply(
                    lambda row: format_metric_value(row["Metric"], row[column]),
                    axis=1,
                )

            st.dataframe(formatted_signal_backtest_metrics, use_container_width=True)

            st.subheader("Exposition cible dans le temps")

            fig_exposure = px.line(
                signal_engine_backtest,
                x=signal_engine_backtest.index,
                y="position",
                title="Exposition historique du Signal Engine",
                labels={
                    "position": "Exposition",
                    "index": "Date",
                },
            )

            fig_exposure.update_traces(
                line_color="#F97316"
            )

            fig_exposure = style_plotly_figure(fig_exposure)

            st.plotly_chart(fig_exposure, use_container_width=True)

            st.subheader("Distribution historique des signaux")

            if not signal_distribution.empty:

                formatted_signal_distribution = signal_distribution.copy()
                formatted_signal_distribution["Frequency"] = formatted_signal_distribution["Frequency"].map(
                    lambda x: f"{x:.2%}"
                )

                st.dataframe(formatted_signal_distribution, use_container_width=True)

                fig_signal_distribution = px.pie(
                    signal_distribution,
                    names="Research Signal",
                    values="Number of Days",
                    title="Répartition historique des signaux",
                    hole=0.45,
                )

                fig_signal_distribution.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                )

                fig_signal_distribution = style_plotly_figure(fig_signal_distribution)

                st.plotly_chart(fig_signal_distribution, use_container_width=True)

            st.subheader("Dernières données du backtest Signal Engine")

            st.dataframe(signal_engine_backtest.tail(20), use_container_width=True)

    # =========================
    # ONGLET 9 : STRATEGY BACKTEST
    # =========================

    with tab_backtest:

        st.markdown(
            """
            <div class="info-card">
                <h2>Strategy Backtest</h2>
                <p>
                    Cette section teste une stratégie systématique simple sur le WTI.
                    La stratégie combine un signal de tendance basé sur moyennes mobiles
                    avec un filtre de volatilité.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        backtest_results = run_backtest(
            prices=oil_prices["WTI"],
            short_window=short_ma_window,
            long_window=long_ma_window,
            vol_window=vol_window,
            max_volatility=max_strategy_volatility,
            transaction_cost=transaction_cost,
        )

        if backtest_results.empty:
            st.warning(
                "Pas assez de données pour lancer le backtest avec les paramètres actuels. "
                "Augmente la période d'analyse ou réduis les fenêtres de moyennes mobiles."
            )

        else:
            backtest_metrics = compute_backtest_metrics(backtest_results)

            strategy_total_return = backtest_results["strategy_equity"].iloc[-1] - 1
            asset_total_return = backtest_results["asset_equity"].iloc[-1] - 1
            latest_position = backtest_results["position"].iloc[-1]

            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card("Strategy Return", f"{strategy_total_return:.2%}", "Performance cumulée stratégie")

            with col2:
                metric_card("WTI Buy & Hold", f"{asset_total_return:.2%}", "Performance cumulée WTI")

            with col3:
                metric_card(
                    "Current Position",
                    "Long" if latest_position == 1 else "Cash",
                    "Position actuelle du modèle",
                )

            st.subheader("Performance cumulée")

            equity_curve = backtest_results[["asset_equity", "strategy_equity"]].rename(
                columns={
                    "asset_equity": "WTI Buy & Hold",
                    "strategy_equity": "Trend Strategy",
                }
            )

            fig_equity = px.line(
                equity_curve,
                x=equity_curve.index,
                y=["WTI Buy & Hold", "Trend Strategy"],
                title="Backtest : WTI Buy & Hold vs Trend Strategy",
                labels={
                    "value": "Capital indexé",
                    "index": "Date",
                    "variable": "Stratégie",
                },
                color_discrete_map={
                    "WTI Buy & Hold": "#38BDF8",
                    "Trend Strategy": "#D4AF37",
                },
            )

            fig_equity = style_plotly_figure(fig_equity)

            st.plotly_chart(fig_equity, use_container_width=True)

            st.subheader("Signal de trading")

            signal_chart = backtest_results[[
                "price",
                "short_ma",
                "long_ma",
            ]].rename(
                columns={
                    "price": "WTI Price",
                    "short_ma": "Short Moving Average",
                    "long_ma": "Long Moving Average",
                }
            )

            fig_signal = px.line(
                signal_chart,
                x=signal_chart.index,
                y=[
                    "WTI Price",
                    "Short Moving Average",
                    "Long Moving Average",
                ],
                title="Signal de tendance sur le WTI",
                labels={
                    "value": "Prix",
                    "index": "Date",
                    "variable": "Série",
                },
                color_discrete_map={
                    "WTI Price": "#F9FAFB",
                    "Short Moving Average": "#D4AF37",
                    "Long Moving Average": "#38BDF8",
                },
            )

            fig_signal = style_plotly_figure(fig_signal)

            st.plotly_chart(fig_signal, use_container_width=True)

            st.subheader("Métriques du backtest")

            formatted_backtest_metrics = backtest_metrics.copy()

            for column in ["WTI Buy & Hold", "Trend Strategy"]:
                formatted_backtest_metrics[column] = formatted_backtest_metrics.apply(
                    lambda row: format_metric_value(row["Metric"], row[column]),
                    axis=1,
                )

            st.dataframe(formatted_backtest_metrics, use_container_width=True)

            st.subheader("Données du backtest")

            st.dataframe(backtest_results.tail(20), use_container_width=True)

    # =========================
    # ONGLET 10 : RISK DASHBOARD
    # =========================

    with tab_risk_dashboard:

        st.markdown(
            """
            <div class="info-card">
                <h2>Risk Dashboard</h2>
                <p>
                    Cette section analyse le risque de marché du WTI à travers la VaR,
                    la CVaR, les stress tests et la distribution historique des rendements.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"""
        **Mode de données utilisé :** `{risk_data_mode}`

        {risk_mode_description}

        La **Value-at-Risk** historique mesure une perte potentielle à un niveau de confiance donné.

        $VaR_{{\\alpha}} = Quantile_{{1-\\alpha}}(R)$

        La **Conditional Value-at-Risk** mesure la perte moyenne dans les pires scénarios :

        $CVaR_{{\\alpha}} = E[R \\mid R \\leq VaR_{{\\alpha}}]$

        Dans ce dashboard, on applique ces mesures aux rendements journaliers du WTI.
        """)

        var_95 = var_table.loc[
            var_table["Confidence Level"] == 0.95,
            "Historical VaR",
        ].iloc[0]

        cvar_95 = var_table.loc[
            var_table["Confidence Level"] == 0.95,
            "Historical CVaR",
        ].iloc[0]

        worst_day = distribution_metrics["worst_day"]

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card("Historical VaR 95%", f"{var_95:.2%}", "Perte journalière seuil")

        with col2:
            metric_card("Historical CVaR 95%", f"{cvar_95:.2%}", "Perte moyenne extrême")

        with col3:
            metric_card("Worst Daily Return", f"{worst_day:.2%}", "Pire rendement journalier observé")

        st.subheader("Tableau VaR / CVaR")

        formatted_var_table = var_table.copy()

        formatted_var_table["Confidence Level"] = formatted_var_table["Confidence Level"].map(lambda x: f"{x:.0%}")
        formatted_var_table["Historical VaR"] = formatted_var_table["Historical VaR"].map(lambda x: f"{x:.2%}")
        formatted_var_table["Historical CVaR"] = formatted_var_table["Historical CVaR"].map(lambda x: f"{x:.2%}")

        st.dataframe(formatted_var_table, use_container_width=True)

        st.subheader("Distribution des rendements journaliers du WTI")

        returns_distribution = pd.DataFrame({
            "WTI Daily Returns": wti_returns,
        })

        fig_distribution = px.histogram(
            returns_distribution,
            x="WTI Daily Returns",
            nbins=80,
            title="Distribution historique des rendements journaliers du WTI",
            labels={
                "WTI Daily Returns": "Rendement journalier",
                "count": "Fréquence",
            },
        )

        fig_distribution.update_traces(
            marker_color="#D4AF37",
            opacity=0.85,
        )

        fig_distribution = style_plotly_figure(fig_distribution)

        st.plotly_chart(fig_distribution, use_container_width=True)

        st.subheader("Statistiques de distribution")

        distribution_table = pd.DataFrame({
            "Metric": [
                "Mean Daily Return",
                "Annualized Volatility",
                "Skewness",
                "Kurtosis",
                "Best Daily Return",
                "Worst Daily Return",
            ],
            "Value": [
                distribution_metrics["mean_return"],
                distribution_metrics["annualized_volatility"],
                distribution_metrics["skewness"],
                distribution_metrics["kurtosis"],
                distribution_metrics["best_day"],
                distribution_metrics["worst_day"],
            ],
        })

        formatted_distribution_table = distribution_table.copy()

        formatted_distribution_table["Value"] = formatted_distribution_table.apply(
            lambda row: (
                f"{row['Value']:.2%}"
                if row["Metric"] in [
                    "Mean Daily Return",
                    "Annualized Volatility",
                    "Best Daily Return",
                    "Worst Daily Return",
                ]
                else f"{row['Value']:.2f}"
            ),
            axis=1,
        )

        st.dataframe(formatted_distribution_table, use_container_width=True)

        st.subheader("Stress Tests sur le prix du WTI")

        formatted_stress_tests = stress_tests.copy()

        formatted_stress_tests["Shock"] = formatted_stress_tests["Shock"].map(lambda x: f"{x:.0%}")
        formatted_stress_tests["Stressed Price"] = formatted_stress_tests["Stressed Price"].map(lambda x: f"{x:.2f} $")
        formatted_stress_tests["PnL per Unit"] = formatted_stress_tests["PnL per Unit"].map(lambda x: f"{x:.2f} $")

        st.dataframe(formatted_stress_tests, use_container_width=True)

        st.subheader("Visualisation des scénarios de stress")

        fig_stress = px.bar(
            stress_tests,
            x="Scenario",
            y="PnL per Unit",
            title="PnL théorique par scénario de stress",
            labels={
                "Scenario": "Scénario",
                "PnL per Unit": "PnL par unité",
            },
        )

        fig_stress.update_traces(
            marker_color="#F97316",
            opacity=0.9,
        )

        fig_stress = style_plotly_figure(fig_stress)

        st.plotly_chart(fig_stress, use_container_width=True)

    # =========================
    # ONGLET 11 : DATA
    # =========================

    with tab_data:

        st.markdown(
            """
            <div class="info-card">
                <h2>Données utilisées</h2>
                <p>
                    Cette section affiche les données utilisées par le dashboard :
                    prix historiques, rendements bruts, rendements nettoyés,
                    drivers de marché, dataset machine learning, simulations Monte Carlo,
                    executive summary, régimes de marché, signal engine,
                    backtest Signal Engine, courbe futures et données de backtest.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Executive Summary")

        st.dataframe(pd.DataFrame([executive_summary]), use_container_width=True)

        st.subheader("Executive Summary Table")

        st.dataframe(executive_summary_table, use_container_width=True)

        st.subheader("Prix historiques")

        st.dataframe(prices.tail(20), use_container_width=True)

        st.subheader("Rendements journaliers bruts")

        st.dataframe(returns.tail(20), use_container_width=True)

        st.subheader("Rendements nettoyés pour le Risk Dashboard")

        st.dataframe(capped_cleaned_returns.tail(20), use_container_width=True)

        if not volatility_dataset.empty:
            st.subheader("Dataset du modèle de volatilité")

            st.dataframe(volatility_dataset.tail(20), use_container_width=True)

        if not volatility_predictions.empty:
            st.subheader("Prédictions du modèle de volatilité")

            st.dataframe(volatility_predictions.tail(20), use_container_width=True)

        if not monte_carlo_summary.empty:
            st.subheader("Monte Carlo Summary")

            st.dataframe(monte_carlo_summary, use_container_width=True)

        if not monte_carlo_risk_table.empty:
            st.subheader("Monte Carlo Risk Table")

            st.dataframe(monte_carlo_risk_table, use_container_width=True)

        if not monte_carlo_percentiles.empty:
            st.subheader("Monte Carlo Percentiles")

            st.dataframe(monte_carlo_percentiles.tail(20), use_container_width=True)

        if not market_regimes.empty:
            st.subheader("Market Regimes")

            st.dataframe(market_regimes.tail(20), use_container_width=True)

        if not regime_summary.empty:
            st.subheader("Résumé des régimes")

            st.dataframe(regime_summary, use_container_width=True)

        if signal_summary is not None:
            st.subheader("Signal Engine Summary")

            st.dataframe(pd.DataFrame([signal_summary]), use_container_width=True)

        if not signal_breakdown.empty:
            st.subheader("Signal Engine Breakdown")

            st.dataframe(signal_breakdown, use_container_width=True)

        if not signal_engine_backtest.empty:
            st.subheader("Signal Engine Backtest")

            st.dataframe(signal_engine_backtest.tail(20), use_container_width=True)

        if not signal_backtest_metrics.empty:
            st.subheader("Signal Engine Backtest Metrics")

            st.dataframe(signal_backtest_metrics, use_container_width=True)

        if not signal_distribution.empty:
            st.subheader("Signal Distribution")

            st.dataframe(signal_distribution, use_container_width=True)

        if not market_drivers.empty:
            st.subheader("Market Drivers")

            st.dataframe(market_drivers.tail(20), use_container_width=True)

            st.subheader("Rendements des Market Drivers")

            st.dataframe(driver_returns.tail(20), use_container_width=True)

        if not futures_curve.empty:
            st.subheader("Courbe futures WTI")

            st.dataframe(futures_curve, use_container_width=True)


except Exception as e:
    st.error("Une erreur est survenue pendant le chargement des données.")
    st.exception(e)