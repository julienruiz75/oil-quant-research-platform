# src/regime_detection.py

# Importation de numpy pour les calculs numériques
import numpy as np

# Importation de pandas pour manipuler les séries temporelles
import pandas as pd


def compute_market_regime_features(
    prices: pd.Series,
    returns: pd.Series,
    vix_series: pd.Series | None = None,
    dxy_returns: pd.Series | None = None
) -> pd.DataFrame:
    """
    Calcule les variables nécessaires à la détection de régime de marché.

    Variables calculées :
    - momentum 20 jours ;
    - momentum 60 jours ;
    - volatilité réalisée 20 jours ;
    - volatilité réalisée 60 jours ;
    - drawdown ;
    - niveau et variation du VIX si disponible ;
    - variation du dollar si disponible.

    Paramètres
    ----------
    prices : pd.Series
        Série des prix du WTI.

    returns : pd.Series
        Série des rendements journaliers du WTI.

    vix_series : pd.Series | None
        Série du VIX si disponible.

    dxy_returns : pd.Series | None
        Rendements du dollar index si disponible.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les variables de régime.
    """

    # Création du tableau principal
    features = pd.DataFrame(index=prices.index)

    # Prix du WTI
    features["price"] = prices

    # Rendement journalier du WTI
    features["return_1d"] = returns.reindex(features.index)

    # Momentum court terme
    features["momentum_20d"] = prices / prices.shift(20) - 1

    # Momentum moyen terme
    features["momentum_60d"] = prices / prices.shift(60) - 1

    # Volatilité réalisée annualisée sur 20 jours
    features["realized_vol_20d"] = features["return_1d"].rolling(20).std() * np.sqrt(252)

    # Volatilité réalisée annualisée sur 60 jours
    features["realized_vol_60d"] = features["return_1d"].rolling(60).std() * np.sqrt(252)

    # Drawdown par rapport au plus haut historique
    running_max = prices.cummax()
    features["drawdown"] = prices / running_max - 1

    # Si le VIX est disponible, on ajoute son niveau et sa variation sur 20 jours
    if vix_series is not None and not vix_series.dropna().empty:
        features["vix_level"] = vix_series.reindex(features.index)
        features["vix_change_20d"] = features["vix_level"] / features["vix_level"].shift(20) - 1

    # Si le DXY est disponible, on ajoute ses rendements et son momentum sur 20 jours
    if dxy_returns is not None and not dxy_returns.dropna().empty:
        features["dxy_return_1d"] = dxy_returns.reindex(features.index)
        features["dxy_momentum_20d"] = (
            1 + features["dxy_return_1d"]
        ).rolling(20).apply(np.prod, raw=True) - 1

    # On supprime seulement les lignes où les variables obligatoires sont manquantes
    required_columns = [
        "price",
        "return_1d",
        "momentum_20d",
        "momentum_60d",
        "realized_vol_20d",
        "realized_vol_60d",
        "drawdown",
    ]

    features = features.dropna(subset=required_columns)

    return features


def classify_regime(row: pd.Series) -> str:
    """
    Classe une ligne de données dans un régime de marché.

    Nouvelle logique plus réaliste pour le pétrole :
    - Stress Regime : VIX très élevé OU drawdown fort combiné à une volatilité élevée ;
    - High Volatility : volatilité réalisée élevée sans stress extrême ;
    - Bullish Trend : momentum positif et volatilité non excessive ;
    - Bearish Trend : momentum négatif ;
    - Neutral : sinon.

    Paramètres
    ----------
    row : pd.Series
        Ligne contenant les variables de régime.

    Retour
    ------
    str
        Nom du régime de marché.
    """

    # Récupération des variables principales
    momentum_20d = row.get("momentum_20d", np.nan)
    momentum_60d = row.get("momentum_60d", np.nan)
    realized_vol_20d = row.get("realized_vol_20d", np.nan)
    realized_vol_60d = row.get("realized_vol_60d", np.nan)
    drawdown = row.get("drawdown", np.nan)
    vix_level = row.get("vix_level", np.nan)

    # =========================
    # 1. STRESS REGIME
    # =========================

    # Stress macro clair : VIX très élevé
    if not np.isnan(vix_level) and vix_level >= 40:
        return "Stress Regime"

    # Stress pétrole : drawdown très fort ET volatilité élevée
    if drawdown <= -0.30 and realized_vol_20d >= 0.55:
        return "Stress Regime"

    # Stress violent : drawdown important ET volatilité extrême
    if drawdown <= -0.20 and realized_vol_20d >= 0.75:
        return "Stress Regime"

    # =========================
    # 2. HIGH VOLATILITY
    # =========================

    # Volatilité très élevée, mais sans signal de stress complet
    if realized_vol_20d >= 0.60:
        return "High Volatility"

    # VIX élevé + volatilité pétrole élevée
    if not np.isnan(vix_level) and vix_level >= 30 and realized_vol_20d >= 0.45:
        return "High Volatility"

    # =========================
    # 3. BULLISH TREND
    # =========================

    # Momentum positif sur 20 jours et 60 jours, avec volatilité raisonnable
    if momentum_20d > 0.03 and momentum_60d > 0 and realized_vol_20d < 0.60:
        return "Bullish Trend"

    # =========================
    # 4. BEARISH TREND
    # =========================

    # Momentum négatif sur 20 jours et 60 jours
    if momentum_20d < -0.03 and momentum_60d < 0:
        return "Bearish Trend"

    # Momentum 60 jours négatif et drawdown déjà marqué
    if momentum_60d < -0.05 and drawdown <= -0.10:
        return "Bearish Trend"

    # =========================
    # 5. NEUTRAL
    # =========================

    return "Neutral"


def detect_market_regimes(features: pd.DataFrame) -> pd.DataFrame:
    """
    Applique la classification de régime sur l'ensemble du dataset.

    Paramètres
    ----------
    features : pd.DataFrame
        Tableau contenant les variables de régime.

    Retour
    ------
    pd.DataFrame
        Tableau enrichi avec une colonne market_regime.
    """

    # Copie du dataset
    regimes = features.copy()

    # Application de la classification ligne par ligne
    regimes["market_regime"] = regimes.apply(classify_regime, axis=1)

    return regimes


def compute_regime_summary(regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un résumé statistique par régime de marché.

    Métriques :
    - nombre de jours ;
    - fréquence ;
    - rendement moyen ;
    - volatilité annualisée ;
    - drawdown moyen.

    Paramètres
    ----------
    regimes : pd.DataFrame
        Tableau contenant les régimes de marché.

    Retour
    ------
    pd.DataFrame
        Tableau résumé par régime.
    """

    rows = []

    total_days = len(regimes)

    if total_days == 0:
        return pd.DataFrame()

    for regime_name, group in regimes.groupby("market_regime"):

        number_of_days = len(group)
        frequency = number_of_days / total_days

        mean_daily_return = group["return_1d"].mean()
        annualized_volatility = group["return_1d"].std() * np.sqrt(252)
        average_drawdown = group["drawdown"].mean()

        rows.append({
            "Regime": regime_name,
            "Number of Days": number_of_days,
            "Frequency": frequency,
            "Mean Daily Return": mean_daily_return,
            "Annualized Volatility": annualized_volatility,
            "Average Drawdown": average_drawdown
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by="Frequency",
        ascending=False
    )

    return summary


def get_latest_regime(regimes: pd.DataFrame) -> dict | None:
    """
    Récupère le dernier régime de marché disponible.

    Paramètres
    ----------
    regimes : pd.DataFrame
        Tableau contenant les régimes de marché.

    Retour
    ------
    dict | None
        Informations sur le régime actuel.
    """

    if regimes.empty:
        return None

    latest_row = regimes.iloc[-1]

    latest_info = {
        "date": regimes.index[-1],
        "regime": latest_row["market_regime"],
        "momentum_20d": latest_row["momentum_20d"],
        "momentum_60d": latest_row["momentum_60d"],
        "realized_vol_20d": latest_row["realized_vol_20d"],
        "realized_vol_60d": latest_row["realized_vol_60d"],
        "drawdown": latest_row["drawdown"]
    }

    if "vix_level" in latest_row.index and not pd.isna(latest_row["vix_level"]):
        latest_info["vix_level"] = latest_row["vix_level"]

    if "vix_change_20d" in latest_row.index and not pd.isna(latest_row["vix_change_20d"]):
        latest_info["vix_change_20d"] = latest_row["vix_change_20d"]

    if "dxy_momentum_20d" in latest_row.index and not pd.isna(latest_row["dxy_momentum_20d"]):
        latest_info["dxy_momentum_20d"] = latest_row["dxy_momentum_20d"]

    return latest_info