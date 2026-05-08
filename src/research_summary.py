# src/research_summary.py

import pandas as pd


def get_table_value(
    table: pd.DataFrame,
    metric_name: str,
    metric_column: str = "Metric",
    value_column: str = "Value",
):
    """
    Récupère une valeur dans un tableau de métriques.

    Exemple :
    - metric_name = "Probability of Loss"
    - value_column = "Value"
    """

    if table is None or table.empty:
        return None

    if metric_column not in table.columns or value_column not in table.columns:
        return None

    selected_rows = table.loc[table[metric_column] == metric_name]

    if selected_rows.empty:
        return None

    return selected_rows[value_column].iloc[0]


def classify_risk_level(var_95: float | None, volatility: float | None) -> str:
    """
    Classe le niveau de risque global du WTI.
    """

    if var_95 is None or volatility is None:
        return "Unknown"

    abs_var = abs(var_95)

    if abs_var >= 0.08 or volatility >= 0.70:
        return "Very High Risk"

    if abs_var >= 0.05 or volatility >= 0.45:
        return "High Risk"

    if abs_var >= 0.03 or volatility >= 0.30:
        return "Moderate Risk"

    return "Low Risk"


def build_market_interpretation(
    signal: str,
    regime: str,
    target_exposure: float,
    probability_of_loss: float | None,
    monte_carlo_var_95: float | None,
) -> str:
    """
    Génère une interprétation textuelle simple du marché.
    """

    if signal in ["Strong Bullish", "Bullish"] and target_exposure > 0.50:
        return (
            "Le modèle indique un environnement favorable au WTI. "
            "Le signal est constructif, l'exposition cible est élevée, "
            "et le marché peut être abordé avec une logique directionnelle positive."
        )

    if signal == "Neutral":
        return (
            "Le modèle indique un environnement équilibré. "
            "Aucun avantage directionnel fort ne ressort actuellement. "
            "Une exposition réduite ou progressive est préférable."
        )

    if signal in ["Defensive", "Stress Defensive"]:
        return (
            "Le modèle indique un environnement fragile. "
            "La priorité est la conservation du capital plutôt que la recherche agressive de rendement. "
            "Une exposition faible au WTI est cohérente avec ce régime."
        )

    if probability_of_loss is not None and probability_of_loss > 0.60:
        return (
            "Les simulations Monte Carlo indiquent une probabilité de perte élevée. "
            "Le scénario central doit être interprété avec prudence."
        )

    if monte_carlo_var_95 is not None and monte_carlo_var_95 < -0.08:
        return (
            "La VaR Monte Carlo montre un risque baissier important. "
            "Le potentiel de perte extrême domine la lecture du marché."
        )

    return (
        "Le marché ne présente pas de signal extrême. "
        "La situation doit être suivie à travers le régime de marché, la volatilité et le momentum."
    )


def build_research_summary(
    signal_summary: dict | None,
    latest_regime_info: dict | None,
    monte_carlo_summary: pd.DataFrame,
    monte_carlo_risk_table: pd.DataFrame,
    var_table: pd.DataFrame,
    distribution_metrics: dict,
) -> dict:
    """
    Construit une synthèse complète du dashboard.

    Cette fonction utilise :
    - le Signal Engine ;
    - le régime de marché ;
    - les simulations Monte Carlo ;
    - la VaR historique ;
    - les statistiques de distribution.
    """

    if signal_summary is None:
        signal = "Not Available"
        signal_description = "Signal Engine unavailable"
        target_exposure = 0.0
        total_score = 0
    else:
        signal = signal_summary.get("signal", "Not Available")
        signal_description = signal_summary.get("signal_description", "")
        target_exposure = signal_summary.get("target_exposure", 0.0)
        total_score = signal_summary.get("total_score", 0)

    if latest_regime_info is None:
        regime = "Not Available"
        momentum_20d = None
        realized_vol_20d = None
        drawdown = None
    else:
        regime = latest_regime_info.get("regime", "Not Available")
        momentum_20d = latest_regime_info.get("momentum_20d", None)
        realized_vol_20d = latest_regime_info.get("realized_vol_20d", None)
        drawdown = latest_regime_info.get("drawdown", None)

    median_simulated_price = get_table_value(
        monte_carlo_summary,
        metric_name="Median Simulated Price",
    )

    probability_of_loss = get_table_value(
        monte_carlo_summary,
        metric_name="Probability of Loss",
    )

    monte_carlo_var_95 = None

    if monte_carlo_risk_table is not None and not monte_carlo_risk_table.empty:
        selected_var = monte_carlo_risk_table.loc[
            monte_carlo_risk_table["Confidence Level"] == 0.95,
            "Monte Carlo VaR",
        ]

        if not selected_var.empty:
            monte_carlo_var_95 = selected_var.iloc[0]

    historical_var_95 = None

    if var_table is not None and not var_table.empty:
        selected_historical_var = var_table.loc[
            var_table["Confidence Level"] == 0.95,
            "Historical VaR",
        ]

        if not selected_historical_var.empty:
            historical_var_95 = selected_historical_var.iloc[0]

    annualized_volatility = None

    if distribution_metrics is not None:
        annualized_volatility = distribution_metrics.get(
            "annualized_volatility",
            None,
        )

    risk_level = classify_risk_level(
        var_95=historical_var_95,
        volatility=annualized_volatility,
    )

    interpretation = build_market_interpretation(
        signal=signal,
        regime=regime,
        target_exposure=target_exposure,
        probability_of_loss=probability_of_loss,
        monte_carlo_var_95=monte_carlo_var_95,
    )

    summary = {
        "signal": signal,
        "signal_description": signal_description,
        "total_score": total_score,
        "target_exposure": target_exposure,
        "regime": regime,
        "momentum_20d": momentum_20d,
        "realized_vol_20d": realized_vol_20d,
        "drawdown": drawdown,
        "median_simulated_price": median_simulated_price,
        "probability_of_loss": probability_of_loss,
        "monte_carlo_var_95": monte_carlo_var_95,
        "historical_var_95": historical_var_95,
        "annualized_volatility": annualized_volatility,
        "risk_level": risk_level,
        "interpretation": interpretation,
    }

    return summary


def build_research_summary_table(summary: dict) -> pd.DataFrame:
    """
    Convertit la synthèse en tableau affichable dans Streamlit.
    """

    if summary is None:
        return pd.DataFrame()

    rows = [
        {
            "Category": "Signal Engine",
            "Metric": "Research Signal",
            "Value": summary["signal"],
        },
        {
            "Category": "Signal Engine",
            "Metric": "Signal Description",
            "Value": summary["signal_description"],
        },
        {
            "Category": "Signal Engine",
            "Metric": "Total Score",
            "Value": summary["total_score"],
        },
        {
            "Category": "Signal Engine",
            "Metric": "Target Exposure",
            "Value": summary["target_exposure"],
        },
        {
            "Category": "Market Regime",
            "Metric": "Current Regime",
            "Value": summary["regime"],
        },
        {
            "Category": "Market Regime",
            "Metric": "Momentum 20D",
            "Value": summary["momentum_20d"],
        },
        {
            "Category": "Market Regime",
            "Metric": "Realized Volatility 20D",
            "Value": summary["realized_vol_20d"],
        },
        {
            "Category": "Market Regime",
            "Metric": "Current Drawdown",
            "Value": summary["drawdown"],
        },
        {
            "Category": "Monte Carlo",
            "Metric": "Median Simulated Price",
            "Value": summary["median_simulated_price"],
        },
        {
            "Category": "Monte Carlo",
            "Metric": "Probability of Loss",
            "Value": summary["probability_of_loss"],
        },
        {
            "Category": "Risk",
            "Metric": "Historical VaR 95%",
            "Value": summary["historical_var_95"],
        },
        {
            "Category": "Risk",
            "Metric": "Monte Carlo VaR 95%",
            "Value": summary["monte_carlo_var_95"],
        },
        {
            "Category": "Risk",
            "Metric": "Annualized Volatility",
            "Value": summary["annualized_volatility"],
        },
        {
            "Category": "Risk",
            "Metric": "Risk Level",
            "Value": summary["risk_level"],
        },
    ]

    return pd.DataFrame(rows)


def format_research_summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Formate les valeurs du tableau de synthèse.
    """

    if table.empty:
        return table

    formatted_table = table.copy()

    percentage_metrics = [
        "Target Exposure",
        "Momentum 20D",
        "Realized Volatility 20D",
        "Current Drawdown",
        "Probability of Loss",
        "Historical VaR 95%",
        "Monte Carlo VaR 95%",
        "Annualized Volatility",
    ]

    price_metrics = [
        "Median Simulated Price",
    ]

    def format_value(row):
        metric = row["Metric"]
        value = row["Value"]

        if pd.isna(value):
            return "Not available"

        if metric in percentage_metrics:
            return f"{value:.2%}"

        if metric in price_metrics:
            return f"{value:.2f} $"

        return str(value)

    formatted_table["Value"] = formatted_table.apply(format_value, axis=1)

    return formatted_table