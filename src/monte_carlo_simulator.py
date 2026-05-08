# src/monte_carlo_simulator.py

# Importation de numpy pour les simulations numériques
import numpy as np

# Importation de pandas pour manipuler les tableaux de données
import pandas as pd


def estimate_annualized_parameters(returns: pd.Series) -> dict:
    """
    Estime les paramètres annualisés à partir des rendements journaliers.

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers.

    Retour
    ------
    dict
        Dictionnaire contenant le rendement moyen annualisé et la volatilité annualisée.
    """

    # Suppression des valeurs manquantes
    clean_returns = returns.dropna()

    # Si aucune donnée n'est disponible, on renvoie des paramètres neutres
    if clean_returns.empty:
        return {
            "annualized_drift": 0.0,
            "annualized_volatility": 0.0,
        }

    # Rendement moyen annualisé
    annualized_drift = clean_returns.mean() * 252

    # Volatilité annualisée
    annualized_volatility = clean_returns.std() * np.sqrt(252)

    return {
        "annualized_drift": annualized_drift,
        "annualized_volatility": annualized_volatility,
    }


def run_monte_carlo_simulation(
    current_price: float,
    returns: pd.Series,
    horizon_days: int = 30,
    n_simulations: int = 5000,
    annualized_volatility_override: float | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simule des trajectoires futures du prix du WTI avec un modèle de type Geometric Brownian Motion.

    Formule utilisée :

    S_{t+1} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * epsilon)

    où :
    - S_t est le prix actuel ;
    - mu est le drift annualisé ;
    - sigma est la volatilité annualisée ;
    - dt = 1 / 252 ;
    - epsilon suit une loi normale standard.

    Paramètres
    ----------
    current_price : float
        Prix actuel du WTI.

    returns : pd.Series
        Rendements journaliers historiques.

    horizon_days : int
        Horizon de simulation en jours de marché.

    n_simulations : int
        Nombre de trajectoires simulées.

    annualized_volatility_override : float | None
        Volatilité annualisée imposée.
        Si None, on utilise la volatilité historique estimée.

    random_seed : int
        Graine aléatoire pour rendre les résultats reproductibles.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les trajectoires simulées.
    """

    # Sécurité sur le prix actuel
    if current_price <= 0:
        return pd.DataFrame()

    # Estimation des paramètres historiques
    parameters = estimate_annualized_parameters(returns)

    annualized_drift = parameters["annualized_drift"]
    annualized_volatility = parameters["annualized_volatility"]

    # Si une volatilité prédite est fournie, on l'utilise à la place de la volatilité historique
    if annualized_volatility_override is not None and annualized_volatility_override > 0:
        annualized_volatility = annualized_volatility_override

    # Sécurité si la volatilité est nulle ou indisponible
    if annualized_volatility <= 0 or np.isnan(annualized_volatility):
        return pd.DataFrame()

    # Initialisation du générateur aléatoire
    np.random.seed(random_seed)

    # Pas de temps journalier
    dt = 1 / 252

    # Création du tableau des trajectoires
    simulations = np.zeros((horizon_days + 1, n_simulations))

    # Le jour 0 correspond au prix actuel
    simulations[0, :] = current_price

    # Simulation jour par jour
    for day in range(1, horizon_days + 1):

        # Chocs aléatoires
        random_shocks = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=n_simulations,
        )

        # Rendements simulés
        simulated_returns = np.exp(
            (annualized_drift - 0.5 * annualized_volatility ** 2) * dt
            + annualized_volatility * np.sqrt(dt) * random_shocks
        )

        # Prix simulés
        simulations[day, :] = simulations[day - 1, :] * simulated_returns

    # Conversion en DataFrame
    simulation_paths = pd.DataFrame(
        simulations,
        index=range(horizon_days + 1),
        columns=[f"Simulation {i + 1}" for i in range(n_simulations)],
    )

    simulation_paths.index.name = "Day"

    return simulation_paths


def build_monte_carlo_percentiles(simulation_paths: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les percentiles des trajectoires simulées jour par jour.

    Paramètres
    ----------
    simulation_paths : pd.DataFrame
        Trajectoires simulées.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les percentiles 5%, 25%, 50%, 75% et 95%.
    """

    if simulation_paths.empty:
        return pd.DataFrame()

    percentiles = pd.DataFrame(index=simulation_paths.index)

    percentiles["P5"] = simulation_paths.quantile(0.05, axis=1)
    percentiles["P25"] = simulation_paths.quantile(0.25, axis=1)
    percentiles["P50"] = simulation_paths.quantile(0.50, axis=1)
    percentiles["P75"] = simulation_paths.quantile(0.75, axis=1)
    percentiles["P95"] = simulation_paths.quantile(0.95, axis=1)

    return percentiles


def compute_monte_carlo_summary(
    simulation_paths: pd.DataFrame,
    current_price: float,
) -> pd.DataFrame:
    """
    Calcule un résumé des scénarios Monte Carlo à l'horizon final.

    Paramètres
    ----------
    simulation_paths : pd.DataFrame
        Trajectoires simulées.

    current_price : float
        Prix actuel du WTI.

    Retour
    ------
    pd.DataFrame
        Tableau résumé des scénarios.
    """

    if simulation_paths.empty or current_price <= 0:
        return pd.DataFrame()

    # Prix finaux simulés
    final_prices = simulation_paths.iloc[-1]

    # Rendements finaux simulés
    final_returns = final_prices / current_price - 1

    summary = pd.DataFrame({
        "Metric": [
            "Current Price",
            "Mean Simulated Price",
            "Median Simulated Price",
            "5th Percentile Price",
            "95th Percentile Price",
            "Mean Simulated Return",
            "Median Simulated Return",
            "Probability of Loss",
            "Probability of Gain",
        ],
        "Value": [
            current_price,
            final_prices.mean(),
            final_prices.median(),
            final_prices.quantile(0.05),
            final_prices.quantile(0.95),
            final_returns.mean(),
            final_returns.median(),
            (final_returns < 0).mean(),
            (final_returns > 0).mean(),
        ],
    })

    return summary


def compute_monte_carlo_risk_table(
    simulation_paths: pd.DataFrame,
    current_price: float,
    confidence_levels: list[float] | None = None,
) -> pd.DataFrame:
    """
    Calcule une VaR et une CVaR simulées à partir des scénarios Monte Carlo.

    Paramètres
    ----------
    simulation_paths : pd.DataFrame
        Trajectoires simulées.

    current_price : float
        Prix actuel du WTI.

    confidence_levels : list[float] | None
        Niveaux de confiance.

    Retour
    ------
    pd.DataFrame
        Tableau de risque Monte Carlo.
    """

    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    if simulation_paths.empty or current_price <= 0:
        return pd.DataFrame()

    final_prices = simulation_paths.iloc[-1]
    final_returns = final_prices / current_price - 1

    rows = []

    for confidence_level in confidence_levels:

        # Quantile de perte
        var_return = final_returns.quantile(1 - confidence_level)

        # Moyenne des pertes au-delà de la VaR
        cvar_return = final_returns[final_returns <= var_return].mean()

        rows.append({
            "Confidence Level": confidence_level,
            "Monte Carlo VaR": var_return,
            "Monte Carlo CVaR": cvar_return,
            "VaR Price Impact": current_price * var_return,
            "CVaR Price Impact": current_price * cvar_return,
        })

    risk_table = pd.DataFrame(rows)

    return risk_table


def sample_simulation_paths(
    simulation_paths: pd.DataFrame,
    n_paths: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Sélectionne un échantillon de trajectoires pour l'affichage graphique.

    Afficher 5000 trajectoires dans Plotly serait trop lourd.
    On en sélectionne donc un nombre limité.

    Paramètres
    ----------
    simulation_paths : pd.DataFrame
        Trajectoires simulées.

    n_paths : int
        Nombre de trajectoires à conserver.

    random_seed : int
        Graine aléatoire.

    Retour
    ------
    pd.DataFrame
        Échantillon de trajectoires simulées.
    """

    if simulation_paths.empty:
        return pd.DataFrame()

    np.random.seed(random_seed)

    available_columns = list(simulation_paths.columns)

    selected_columns = np.random.choice(
        available_columns,
        size=min(n_paths, len(available_columns)),
        replace=False,
    )

    sampled_paths = simulation_paths[selected_columns].copy()

    return sampled_paths