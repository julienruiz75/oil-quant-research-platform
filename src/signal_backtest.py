# src/signal_backtest.py

# Importation de numpy pour les calculs numériques
import numpy as np

# Importation de pandas pour manipuler les séries temporelles
import pandas as pd

# Importation des fonctions de scoring déjà utilisées par le Signal Engine
from src.signal_engine import (
    score_regime,
    score_momentum,
    score_volatility,
    score_futures_curve,
    score_drawdown,
)


def classify_signal_from_score(total_score: int) -> str:
    """
    Transforme un score total en signal de recherche.

    Paramètres
    ----------
    total_score : int
        Score agrégé du Signal Engine.

    Retour
    ------
    str
        Signal de recherche.
    """

    if total_score >= 4:
        return "Strong Bullish"

    if total_score >= 2:
        return "Bullish"

    if total_score >= 0:
        return "Neutral"

    if total_score >= -3:
        return "Defensive"

    return "Stress Defensive"


def compute_target_exposure(
    total_score: int,
    volatility: float,
    target_volatility: float = 0.25,
    max_exposure: float = 1.00
) -> float:
    """
    Calcule l'exposition cible à partir du score et de la volatilité.

    Idée :
    - plus la volatilité est élevée, plus l'exposition doit être réduite ;
    - plus le score est faible, plus l'exposition doit être réduite.

    Formule de base :

    target_exposure = target_volatility / realized_volatility

    Puis on ajuste selon le score.

    Paramètres
    ----------
    total_score : int
        Score total du Signal Engine.

    volatility : float
        Volatilité annualisée utilisée pour dimensionner l'exposition.

    target_volatility : float
        Volatilité cible du portefeuille.

    max_exposure : float
        Exposition maximale autorisée.

    Retour
    ------
    float
        Exposition cible.
    """

    # Sécurité si la volatilité n'est pas disponible
    if volatility is None or np.isnan(volatility) or volatility <= 0:
        return 0.0

    # Exposition basée sur la volatilité
    volatility_based_exposure = target_volatility / volatility

    # Limite de l'exposition maximale
    volatility_based_exposure = min(volatility_based_exposure, max_exposure)

    # Ajustement selon le score
    if total_score >= 4:
        return volatility_based_exposure

    if total_score >= 2:
        return 0.75 * volatility_based_exposure

    if total_score >= 0:
        return 0.35 * volatility_based_exposure

    if total_score >= -3:
        return 0.10 * volatility_based_exposure

    return 0.0


def build_historical_signal_backtest(
    market_regimes: pd.DataFrame,
    curve_regime: str | None = None,
    target_volatility: float = 0.25,
    max_exposure: float = 1.00,
    transaction_cost: float = 0.0005
) -> pd.DataFrame:
    """
    Construit un backtest historique du Signal Engine.

    Le backtest utilise les données déjà calculées dans market_regimes :
    - régime de marché ;
    - momentum 20 jours ;
    - momentum 60 jours ;
    - volatilité réalisée 20 jours ;
    - drawdown ;
    - rendement journalier.

    Remarque :
    Pour le backtest historique, on utilise la volatilité réalisée 20 jours
    comme approximation disponible de la volatilité prévue.

    Paramètres
    ----------
    market_regimes : pd.DataFrame
        Tableau contenant les variables de régime de marché.

    curve_regime : str | None
        Régime de la courbe futures : Backwardation, Contango ou Flat.

    target_volatility : float
        Volatilité cible du portefeuille.

    max_exposure : float
        Exposition maximale autorisée.

    transaction_cost : float
        Frais de transaction appliqués à chaque changement d'exposition.

    Retour
    ------
    pd.DataFrame
        Résultats du backtest Signal Engine.
    """

    if market_regimes.empty:
        return pd.DataFrame()

    # Copie du tableau pour éviter de modifier l'original
    data = market_regimes.copy()

    # Sécurité : on vérifie les colonnes nécessaires
    required_columns = [
        "price",
        "return_1d",
        "market_regime",
        "momentum_20d",
        "momentum_60d",
        "realized_vol_20d",
        "drawdown",
    ]

    for column in required_columns:
        if column not in data.columns:
            return pd.DataFrame()

    # Score du régime de marché
    data["regime_score"] = data["market_regime"].apply(score_regime)

    # Score du momentum
    data["momentum_score"] = data.apply(
        lambda row: score_momentum(
            momentum_20d=row["momentum_20d"],
            momentum_60d=row["momentum_60d"],
        ),
        axis=1,
    )

    # Score de la volatilité
    data["volatility_score"] = data["realized_vol_20d"].apply(score_volatility)

    # Score de la courbe futures
    data["curve_score"] = score_futures_curve(curve_regime)

    # Score du drawdown
    data["drawdown_score"] = data["drawdown"].apply(score_drawdown)

    # Score total
    data["total_score"] = (
        data["regime_score"]
        + data["momentum_score"]
        + data["volatility_score"]
        + data["curve_score"]
        + data["drawdown_score"]
    )

    # Signal textuel
    data["research_signal"] = data["total_score"].apply(classify_signal_from_score)

    # Exposition cible
    data["target_exposure"] = data.apply(
        lambda row: compute_target_exposure(
            total_score=row["total_score"],
            volatility=row["realized_vol_20d"],
            target_volatility=target_volatility,
            max_exposure=max_exposure,
        ),
        axis=1,
    )

    # Position utilisée pour le rendement du jour
    # On décale d'un jour pour éviter le biais d'anticipation.
    data["position"] = data["target_exposure"].shift(1).fillna(0.0)

    # Changement d'exposition
    data["position_change"] = data["position"].diff().abs().fillna(0.0)

    # Frais de transaction
    data["transaction_cost"] = data["position_change"] * transaction_cost

    # Rendement Buy & Hold WTI
    data["buy_hold_return"] = data["return_1d"]

    # Rendement de la stratégie Signal Engine
    data["signal_strategy_return"] = (
        data["position"] * data["return_1d"]
        - data["transaction_cost"]
    )

    # Courbe de capital Buy & Hold
    data["buy_hold_equity"] = (1 + data["buy_hold_return"]).cumprod()

    # Courbe de capital Signal Engine
    data["signal_strategy_equity"] = (1 + data["signal_strategy_return"]).cumprod()

    # Nettoyage final
    data = data.dropna()

    return data


def compute_signal_backtest_metrics(signal_backtest: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les métriques principales du backtest Signal Engine.

    Métriques :
    - performance totale ;
    - volatilité annualisée ;
    - Sharpe ratio ;
    - maximum drawdown ;
    - exposition moyenne ;
    - turnover moyen.

    Paramètres
    ----------
    signal_backtest : pd.DataFrame
        Résultats du backtest Signal Engine.

    Retour
    ------
    pd.DataFrame
        Tableau de métriques.
    """

    if signal_backtest.empty:
        return pd.DataFrame()

    # Performance totale
    buy_hold_total_return = signal_backtest["buy_hold_equity"].iloc[-1] - 1
    signal_total_return = signal_backtest["signal_strategy_equity"].iloc[-1] - 1

    # Volatilité annualisée
    buy_hold_volatility = signal_backtest["buy_hold_return"].std() * np.sqrt(252)
    signal_volatility = signal_backtest["signal_strategy_return"].std() * np.sqrt(252)

    # Sharpe ratio simplifié avec taux sans risque supposé égal à 0
    buy_hold_sharpe = (
        signal_backtest["buy_hold_return"].mean()
        / signal_backtest["buy_hold_return"].std()
        * np.sqrt(252)
        if signal_backtest["buy_hold_return"].std() != 0
        else np.nan
    )

    signal_sharpe = (
        signal_backtest["signal_strategy_return"].mean()
        / signal_backtest["signal_strategy_return"].std()
        * np.sqrt(252)
        if signal_backtest["signal_strategy_return"].std() != 0
        else np.nan
    )

    # Maximum drawdown Buy & Hold
    buy_hold_running_max = signal_backtest["buy_hold_equity"].cummax()
    buy_hold_drawdown = signal_backtest["buy_hold_equity"] / buy_hold_running_max - 1
    buy_hold_max_drawdown = buy_hold_drawdown.min()

    # Maximum drawdown Signal Engine
    signal_running_max = signal_backtest["signal_strategy_equity"].cummax()
    signal_drawdown = signal_backtest["signal_strategy_equity"] / signal_running_max - 1
    signal_max_drawdown = signal_drawdown.min()

    # Exposition moyenne
    average_exposure = signal_backtest["position"].mean()

    # Turnover moyen
    average_turnover = signal_backtest["position_change"].mean()

    metrics = pd.DataFrame({
        "Metric": [
            "Total Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Average Exposure",
            "Average Daily Turnover",
        ],
        "WTI Buy & Hold": [
            buy_hold_total_return,
            buy_hold_volatility,
            buy_hold_sharpe,
            buy_hold_max_drawdown,
            1.0,
            np.nan,
        ],
        "Signal Engine Strategy": [
            signal_total_return,
            signal_volatility,
            signal_sharpe,
            signal_max_drawdown,
            average_exposure,
            average_turnover,
        ],
    })

    return metrics


def compute_signal_distribution(signal_backtest: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la distribution historique des signaux.

    Paramètres
    ----------
    signal_backtest : pd.DataFrame
        Résultats du backtest Signal Engine.

    Retour
    ------
    pd.DataFrame
        Répartition des signaux.
    """

    if signal_backtest.empty or "research_signal" not in signal_backtest.columns:
        return pd.DataFrame()

    distribution = (
        signal_backtest["research_signal"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = ["Research Signal", "Number of Days"]

    distribution["Frequency"] = (
        distribution["Number of Days"]
        / distribution["Number of Days"].sum()
    )

    return distribution