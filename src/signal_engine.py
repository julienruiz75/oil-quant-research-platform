# src/signal_engine.py

# Importation de numpy pour gérer les valeurs numériques
import numpy as np

# Importation de pandas pour construire des tableaux propres
import pandas as pd


def score_regime(regime: str) -> int:
    """
    Attribue un score au régime de marché.

    Logique :
    - Bullish Trend : favorable au long ;
    - Bearish Trend : défavorable ;
    - High Volatility : prudence ;
    - Stress Regime : très défensif ;
    - Neutral : neutre.

    Paramètres
    ----------
    regime : str
        Régime de marché détecté.

    Retour
    ------
    int
        Score associé au régime.
    """

    regime_scores = {
        "Bullish Trend": 2,
        "Neutral": 0,
        "High Volatility": -1,
        "Bearish Trend": -2,
        "Stress Regime": -3,
    }

    return regime_scores.get(regime, 0)


def score_momentum(momentum_20d: float, momentum_60d: float) -> int:
    """
    Attribue un score au momentum du WTI.

    Paramètres
    ----------
    momentum_20d : float
        Momentum du WTI sur 20 jours.

    momentum_60d : float
        Momentum du WTI sur 60 jours.

    Retour
    ------
    int
        Score de momentum.
    """

    # Momentum clairement positif
    if momentum_20d > 0.03 and momentum_60d > 0:
        return 2

    # Momentum légèrement positif
    if momentum_20d > 0 and momentum_60d > -0.03:
        return 1

    # Momentum clairement négatif
    if momentum_20d < -0.03 and momentum_60d < 0:
        return -2

    # Momentum légèrement négatif
    if momentum_20d < 0:
        return -1

    return 0


def score_volatility(predicted_volatility: float | None) -> int:
    """
    Attribue un score à la volatilité prédite.

    Paramètres
    ----------
    predicted_volatility : float | None
        Volatilité future prédite annualisée.

    Retour
    ------
    int
        Score de volatilité.
    """

    # Si le modèle n'est pas disponible, on reste neutre
    if predicted_volatility is None or np.isnan(predicted_volatility):
        return 0

    # Volatilité extrêmement élevée
    if predicted_volatility >= 0.80:
        return -3

    # Volatilité élevée
    if predicted_volatility >= 0.60:
        return -2

    # Volatilité modérée
    if predicted_volatility >= 0.40:
        return -1

    # Volatilité acceptable
    return 1


def score_futures_curve(curve_regime: str | None) -> int:
    """
    Attribue un score à la forme de la courbe futures.

    En commodities :
    - Backwardation est souvent favorable au portage long ;
    - Contango est souvent défavorable au portage long.

    Paramètres
    ----------
    curve_regime : str | None
        Régime de courbe : Backwardation, Contango ou Flat.

    Retour
    ------
    int
        Score de courbe.
    """

    if curve_regime == "Backwardation":
        return 1

    if curve_regime == "Contango":
        return -1

    return 0


def score_drawdown(drawdown: float) -> int:
    """
    Attribue un score au drawdown actuel.

    Paramètres
    ----------
    drawdown : float
        Drawdown courant.

    Retour
    ------
    int
        Score de drawdown.
    """

    # Drawdown très important
    if drawdown <= -0.25:
        return -2

    # Drawdown notable
    if drawdown <= -0.15:
        return -1

    # Proche des plus hauts
    if drawdown > -0.05:
        return 1

    return 0


def compute_research_signal(
    latest_regime_info: dict,
    predicted_volatility: float | None,
    curve_regime: str | None,
    target_volatility: float = 0.25,
    max_exposure: float = 1.00
) -> dict:
    """
    Calcule un signal de recherche global sur le WTI.

    Le signal combine :
    - régime de marché ;
    - momentum ;
    - volatilité prédite ;
    - courbe futures ;
    - drawdown.

    Paramètres
    ----------
    latest_regime_info : dict
        Informations sur le dernier régime de marché.

    predicted_volatility : float | None
        Volatilité prédite par le modèle de machine learning.

    curve_regime : str | None
        Régime de courbe futures.

    target_volatility : float
        Volatilité cible du portefeuille.

    max_exposure : float
        Exposition maximale autorisée.

    Retour
    ------
    dict
        Résumé du signal.
    """

    # Récupération des variables principales
    current_regime = latest_regime_info.get("regime", "Neutral")
    momentum_20d = latest_regime_info.get("momentum_20d", 0.0)
    momentum_60d = latest_regime_info.get("momentum_60d", 0.0)
    drawdown = latest_regime_info.get("drawdown", 0.0)

    # Scores individuels
    regime_score = score_regime(current_regime)
    momentum_score = score_momentum(momentum_20d, momentum_60d)
    volatility_score = score_volatility(predicted_volatility)
    curve_score = score_futures_curve(curve_regime)
    drawdown_score = score_drawdown(drawdown)

    # Score total
    total_score = (
        regime_score
        + momentum_score
        + volatility_score
        + curve_score
        + drawdown_score
    )

    # Détermination du signal textuel
    if total_score >= 4:
        signal = "Strong Bullish"
        signal_description = "Conditions favorables pour une exposition longue élevée."

    elif total_score >= 2:
        signal = "Bullish"
        signal_description = "Conditions globalement favorables, mais avec prudence."

    elif total_score >= 0:
        signal = "Neutral"
        signal_description = "Signal équilibré, exposition modérée ou attente."

    elif total_score >= -3:
        signal = "Defensive"
        signal_description = "Conditions fragiles, exposition réduite recommandée."

    else:
        signal = "Stress Defensive"
        signal_description = "Risque élevé, conservation du capital prioritaire."

    # Calcul d'une exposition basée sur la volatilité cible
    if predicted_volatility is None or predicted_volatility <= 0 or np.isnan(predicted_volatility):
        volatility_based_exposure = 0.0
    else:
        volatility_based_exposure = target_volatility / predicted_volatility

    # Limite d'exposition maximale
    volatility_based_exposure = min(volatility_based_exposure, max_exposure)

    # Ajustement selon le score
    if total_score >= 4:
        target_exposure = volatility_based_exposure

    elif total_score >= 2:
        target_exposure = 0.75 * volatility_based_exposure

    elif total_score >= 0:
        target_exposure = 0.35 * volatility_based_exposure

    elif total_score >= -3:
        target_exposure = 0.10 * volatility_based_exposure

    else:
        target_exposure = 0.0

    # Construction du résumé
    signal_summary = {
        "signal": signal,
        "signal_description": signal_description,
        "total_score": total_score,
        "target_exposure": target_exposure,
        "regime_score": regime_score,
        "momentum_score": momentum_score,
        "volatility_score": volatility_score,
        "curve_score": curve_score,
        "drawdown_score": drawdown_score,
        "current_regime": current_regime,
        "momentum_20d": momentum_20d,
        "momentum_60d": momentum_60d,
        "drawdown": drawdown,
        "predicted_volatility": predicted_volatility,
        "curve_regime": curve_regime,
    }

    return signal_summary


def build_signal_breakdown(signal_summary: dict) -> pd.DataFrame:
    """
    Construit un tableau détaillant les composantes du signal.

    Paramètres
    ----------
    signal_summary : dict
        Résumé du signal.

    Retour
    ------
    pd.DataFrame
        Tableau de décomposition du signal.
    """

    rows = [
        {
            "Component": "Market Regime",
            "Value": signal_summary["current_regime"],
            "Score": signal_summary["regime_score"],
        },
        {
            "Component": "Momentum",
            "Value": (
                f"20D: {signal_summary['momentum_20d']:.2%} | "
                f"60D: {signal_summary['momentum_60d']:.2%}"
            ),
            "Score": signal_summary["momentum_score"],
        },
        {
            "Component": "Predicted Volatility",
            "Value": (
                f"{signal_summary['predicted_volatility']:.2%}"
                if signal_summary["predicted_volatility"] is not None
                else "Not available"
            ),
            "Score": signal_summary["volatility_score"],
        },
        {
            "Component": "Futures Curve",
            "Value": signal_summary["curve_regime"],
            "Score": signal_summary["curve_score"],
        },
        {
            "Component": "Drawdown",
            "Value": f"{signal_summary['drawdown']:.2%}",
            "Score": signal_summary["drawdown_score"],
        },
    ]

    breakdown = pd.DataFrame(rows)

    return breakdown