# src/risk.py

# Importation de numpy pour les calculs numériques
import numpy as np

# Importation de pandas pour manipuler les séries temporelles
import pandas as pd


def compute_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calcule la Value-at-Risk historique.

    La VaR historique mesure la perte maximale attendue sur un horizon donné
    avec un certain niveau de confiance.

    Exemple :
    Une VaR 95% de -3% signifie que, historiquement, les pertes journalières
    ont dépassé 3% dans environ 5% des cas.

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers.

    confidence_level : float
        Niveau de confiance, par exemple 0.95 ou 0.99.

    Retour
    ------
    float
        VaR historique.
    """

    # On retire les valeurs manquantes
    returns = returns.dropna()

    # Niveau de quantile correspondant à la queue gauche
    alpha = 1 - confidence_level

    # VaR historique : quantile de la distribution des rendements
    var = returns.quantile(alpha)

    return var


def compute_historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calcule la Conditional Value-at-Risk historique.

    La CVaR mesure la perte moyenne lorsque la perte dépasse la VaR.

    Elle est souvent plus informative que la VaR car elle regarde
    la gravité des scénarios extrêmes.

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers.

    confidence_level : float
        Niveau de confiance, par exemple 0.95 ou 0.99.

    Retour
    ------
    float
        CVaR historique.
    """

    # On retire les valeurs manquantes
    returns = returns.dropna()

    # Calcul de la VaR
    var = compute_historical_var(
        returns=returns,
        confidence_level=confidence_level
    )

    # CVaR : moyenne des rendements inférieurs ou égaux à la VaR
    cvar = returns[returns <= var].mean()

    return cvar


def compute_distribution_metrics(returns: pd.Series) -> dict:
    """
    Calcule des statistiques descriptives sur les rendements.

    Métriques :
    - rendement moyen ;
    - volatilité ;
    - skewness ;
    - kurtosis ;
    - meilleur jour ;
    - pire jour.

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers.

    Retour
    ------
    dict
        Dictionnaire de métriques.
    """

    # On retire les valeurs manquantes
    returns = returns.dropna()

    metrics = {
        "mean_return": returns.mean(),
        "annualized_volatility": returns.std() * np.sqrt(252),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "best_day": returns.max(),
        "worst_day": returns.min()
    }

    return metrics


def compute_stress_tests(current_price: float) -> pd.DataFrame:
    """
    Calcule des scénarios de stress sur le prix actuel du pétrole.

    Les scénarios représentent des chocs hypothétiques sur le prix du WTI.

    Paramètres
    ----------
    current_price : float
        Dernier prix disponible du WTI.

    Retour
    ------
    pd.DataFrame
        Tableau des scénarios de stress.
    """

    # Liste de scénarios de marché
    scenarios = [
        {
            "Scenario": "Mild bearish shock",
            "Shock": -0.05,
            "Description": "Baisse modérée du pétrole"
        },
        {
            "Scenario": "Severe bearish shock",
            "Shock": -0.10,
            "Description": "Forte baisse du pétrole"
        },
        {
            "Scenario": "Crisis shock",
            "Shock": -0.20,
            "Description": "Scénario de crise sur le pétrole"
        },
        {
            "Scenario": "Supply disruption",
            "Shock": 0.10,
            "Description": "Hausse liée à une perturbation de l'offre"
        },
        {
            "Scenario": "Geopolitical spike",
            "Shock": 0.20,
            "Description": "Hausse brutale liée à un choc géopolitique"
        }
    ]

    # Conversion en DataFrame
    stress_df = pd.DataFrame(scenarios)

    # Calcul du prix stressé
    stress_df["Stressed Price"] = current_price * (1 + stress_df["Shock"])

    # Calcul du PnL théorique pour une exposition de 1 unité
    stress_df["PnL per Unit"] = stress_df["Stressed Price"] - current_price

    return stress_df


def compute_var_table(returns: pd.Series) -> pd.DataFrame:
    """
    Calcule un tableau de VaR et CVaR pour plusieurs niveaux de confiance.

    Paramètres
    ----------
    returns : pd.Series
        Série des rendements journaliers.

    Retour
    ------
    pd.DataFrame
        Tableau de VaR / CVaR.
    """

    confidence_levels = [0.95, 0.99]

    rows = []

    for confidence_level in confidence_levels:

        var = compute_historical_var(
            returns=returns,
            confidence_level=confidence_level
        )

        cvar = compute_historical_cvar(
            returns=returns,
            confidence_level=confidence_level
        )

        rows.append({
            "Confidence Level": confidence_level,
            "Historical VaR": var,
            "Historical CVaR": cvar
        })

    var_table = pd.DataFrame(rows)

    return var_table