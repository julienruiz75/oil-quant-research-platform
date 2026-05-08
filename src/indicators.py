# src/indicators.py

# Importation de numpy pour les calculs numériques
import numpy as np

# Importation de pandas pour manipuler les séries temporelles
import pandas as pd


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements journaliers simples.

    Formule :
    r_t = P_t / P_{t-1} - 1

    Paramètres
    ----------
    prices : pd.DataFrame
        Tableau contenant les prix des actifs.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les rendements journaliers.
    """

    # pct_change calcule directement P_t / P_{t-1} - 1
    returns = prices.pct_change()

    # On supprime les lignes vides au début
    returns = returns.dropna(how="all")

    return returns


def compute_rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Calcule la volatilité glissante annualisée.

    Formule :
    sigma_annuelle = sigma_journaliere * sqrt(252)

    Paramètres
    ----------
    returns : pd.DataFrame
        Tableau contenant les rendements journaliers.

    window : int
        Fenêtre glissante utilisée pour calculer la volatilité.

    Retour
    ------
    pd.DataFrame
        Tableau contenant la volatilité annualisée.
    """

    # On calcule l'écart-type des rendements sur une fenêtre glissante
    rolling_std = returns.rolling(window=window).std()

    # On annualise la volatilité avec sqrt(252), car il y a environ 252 jours de trading par an
    rolling_vol = rolling_std * np.sqrt(252)

    return rolling_vol


def compute_drawdowns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les drawdowns des prix.

    Formule :
    DD_t = P_t / max(P_0, ..., P_t) - 1

    Paramètres
    ----------
    prices : pd.DataFrame
        Tableau contenant les prix des actifs.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les drawdowns.
    """

    # Pour chaque date, on calcule le maximum historique observé jusque-là
    running_max = prices.cummax()

    # Le drawdown mesure la perte par rapport au plus haut historique précédent
    drawdowns = prices / running_max - 1

    return drawdowns


def compute_summary_metrics(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un tableau récapitulatif des métriques principales.

    Métriques :
    - prix actuel ;
    - performance totale ;
    - volatilité annualisée ;
    - maximum drawdown.

    Paramètres
    ----------
    prices : pd.DataFrame
        Tableau contenant les prix.

    returns : pd.DataFrame
        Tableau contenant les rendements.

    Retour
    ------
    pd.DataFrame
        Tableau récapitulatif.
    """

    # Liste qui contiendra les résultats actif par actif
    metrics = []

    # On parcourt chaque actif
    for asset in prices.columns:

        # On ignore les colonnes qui n'ont pas de données suffisantes
        if prices[asset].dropna().empty:
            continue

        # Prix de début et prix final
        first_price = prices[asset].dropna().iloc[0]
        last_price = prices[asset].dropna().iloc[-1]

        # Performance totale sur la période
        total_return = last_price / first_price - 1

        # Volatilité annualisée
        annualized_volatility = returns[asset].std() * np.sqrt(252)

        # Maximum drawdown
        drawdowns = compute_drawdowns(prices[[asset]])
        max_drawdown = drawdowns[asset].min()

        # Ajout des résultats dans la liste
        metrics.append({
            "Actif": asset,
            "Prix actuel": last_price,
            "Performance totale": total_return,
            "Volatilité annualisée": annualized_volatility,
            "Maximum drawdown": max_drawdown
        })

    # Conversion en DataFrame
    metrics_df = pd.DataFrame(metrics)

    return metrics_df