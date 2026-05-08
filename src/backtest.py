# src/backtest.py

# Importation de numpy pour les calculs numériques
import numpy as np

# Importation de pandas pour manipuler les séries temporelles
import pandas as pd


def build_trend_signal(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200
) -> pd.DataFrame:
    """
    Construit un signal de tendance basé sur deux moyennes mobiles.

    Règle :
    - signal = 1 si la moyenne mobile courte est supérieure à la moyenne mobile longue ;
    - signal = 0 sinon.

    Paramètres
    ----------
    prices : pd.Series
        Série des prix de l'actif.

    short_window : int
        Fenêtre de la moyenne mobile courte.

    long_window : int
        Fenêtre de la moyenne mobile longue.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les prix, les moyennes mobiles et le signal.
    """

    # Création d'un tableau de travail
    data = pd.DataFrame(index=prices.index)

    # Ajout du prix
    data["price"] = prices

    # Calcul de la moyenne mobile courte
    data["short_ma"] = data["price"].rolling(window=short_window).mean()

    # Calcul de la moyenne mobile longue
    data["long_ma"] = data["price"].rolling(window=long_window).mean()

    # Signal de tendance : 1 si tendance positive, 0 sinon
    data["trend_signal"] = np.where(data["short_ma"] > data["long_ma"], 1, 0)

    # On retire les lignes où les moyennes mobiles ne sont pas encore disponibles
    data = data.dropna()

    return data


def apply_volatility_filter(
    signal_data: pd.DataFrame,
    returns: pd.Series,
    vol_window: int = 30,
    max_volatility: float = 0.60
) -> pd.DataFrame:
    """
    Applique un filtre de volatilité à la stratégie.

    Règle :
    - si la volatilité annualisée est inférieure au seuil, on garde le signal ;
    - si la volatilité annualisée est supérieure au seuil, on coupe la position.

    Paramètres
    ----------
    signal_data : pd.DataFrame
        Tableau contenant au moins une colonne trend_signal.

    returns : pd.Series
        Rendements journaliers de l'actif.

    vol_window : int
        Fenêtre de calcul de volatilité.

    max_volatility : float
        Seuil maximum de volatilité annualisée accepté.

    Retour
    ------
    pd.DataFrame
        Tableau contenant le signal filtré.
    """

    # Copie du tableau de signal
    data = signal_data.copy()

    # Alignement des rendements sur les dates du signal
    aligned_returns = returns.reindex(data.index)

    # Calcul de la volatilité annualisée glissante
    data["realized_volatility"] = aligned_returns.rolling(vol_window).std() * np.sqrt(252)

    # Si la volatilité est trop élevée, on coupe la position
    data["volatility_filter"] = np.where(
        data["realized_volatility"] <= max_volatility,
        1,
        0
    )

    # Signal final = signal de tendance multiplié par le filtre de volatilité
    data["final_signal"] = data["trend_signal"] * data["volatility_filter"]

    # On supprime les lignes incomplètes
    data = data.dropna()

    return data


def run_backtest(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
    vol_window: int = 30,
    max_volatility: float = 0.60,
    transaction_cost: float = 0.0005
) -> pd.DataFrame:
    """
    Lance un backtest simple sur un actif.

    La stratégie :
    - utilise un signal de tendance basé sur deux moyennes mobiles ;
    - applique un filtre de volatilité ;
    - prend en compte des frais de transaction.

    Paramètres
    ----------
    prices : pd.Series
        Série des prix de l'actif.

    short_window : int
        Fenêtre de la moyenne mobile courte.

    long_window : int
        Fenêtre de la moyenne mobile longue.

    vol_window : int
        Fenêtre de volatilité.

    max_volatility : float
        Seuil maximum de volatilité annualisée.

    transaction_cost : float
        Coût de transaction proportionnel à chaque changement de position.

    Retour
    ------
    pd.DataFrame
        Résultats du backtest.
    """

    # Calcul des rendements journaliers
    returns = prices.pct_change()

    # Construction du signal de tendance
    signal_data = build_trend_signal(
        prices=prices,
        short_window=short_window,
        long_window=long_window
    )

    # Application du filtre de volatilité
    data = apply_volatility_filter(
        signal_data=signal_data,
        returns=returns,
        vol_window=vol_window,
        max_volatility=max_volatility
    )

    # Alignement des rendements sur les dates du backtest
    data["asset_return"] = returns.reindex(data.index)

    # Position utilisée pour le rendement du jour :
    # on décale le signal d'un jour pour éviter le biais d'anticipation.
    data["position"] = data["final_signal"].shift(1).fillna(0)

    # Changement de position
    data["position_change"] = data["position"].diff().abs().fillna(0)

    # Frais de transaction
    data["transaction_cost"] = data["position_change"] * transaction_cost

    # Rendement de la stratégie après frais
    data["strategy_return"] = data["position"] * data["asset_return"] - data["transaction_cost"]

    # Performance cumulée de l'actif
    data["asset_equity"] = (1 + data["asset_return"]).cumprod()

    # Performance cumulée de la stratégie
    data["strategy_equity"] = (1 + data["strategy_return"]).cumprod()

    return data.dropna()


def compute_backtest_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les métriques principales du backtest.

    Métriques :
    - performance totale ;
    - volatilité annualisée ;
    - Sharpe ratio ;
    - maximum drawdown ;
    - nombre de trades.

    Paramètres
    ----------
    backtest : pd.DataFrame
        Résultats du backtest.

    Retour
    ------
    pd.DataFrame
        Tableau de métriques.
    """

    # Performance totale de l'actif
    asset_total_return = backtest["asset_equity"].iloc[-1] - 1

    # Performance totale de la stratégie
    strategy_total_return = backtest["strategy_equity"].iloc[-1] - 1

    # Volatilité annualisée de l'actif
    asset_volatility = backtest["asset_return"].std() * np.sqrt(252)

    # Volatilité annualisée de la stratégie
    strategy_volatility = backtest["strategy_return"].std() * np.sqrt(252)

    # Sharpe ratio simplifié avec taux sans risque supposé égal à 0
    asset_sharpe = (
        backtest["asset_return"].mean() / backtest["asset_return"].std() * np.sqrt(252)
        if backtest["asset_return"].std() != 0 else np.nan
    )

    strategy_sharpe = (
        backtest["strategy_return"].mean() / backtest["strategy_return"].std() * np.sqrt(252)
        if backtest["strategy_return"].std() != 0 else np.nan
    )

    # Drawdowns de l'actif
    asset_running_max = backtest["asset_equity"].cummax()
    asset_drawdown = backtest["asset_equity"] / asset_running_max - 1
    asset_max_drawdown = asset_drawdown.min()

    # Drawdowns de la stratégie
    strategy_running_max = backtest["strategy_equity"].cummax()
    strategy_drawdown = backtest["strategy_equity"] / strategy_running_max - 1
    strategy_max_drawdown = strategy_drawdown.min()

    # Nombre de trades
    number_of_trades = int(backtest["position_change"].sum())

    # Tableau final
    metrics = pd.DataFrame({
        "Metric": [
            "Total Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Number of Trades"
        ],
        "WTI Buy & Hold": [
            asset_total_return,
            asset_volatility,
            asset_sharpe,
            asset_max_drawdown,
            np.nan
        ],
        "Trend Strategy": [
            strategy_total_return,
            strategy_volatility,
            strategy_sharpe,
            strategy_max_drawdown,
            number_of_trades
        ]
    })

    return metrics