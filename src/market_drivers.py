# src/market_drivers.py

# Importation de pandas pour manipuler les données
import pandas as pd

# Importation de yfinance pour télécharger les données de marché
import yfinance as yf


# Dictionnaire des indicateurs de marché utilisés comme drivers du pétrole
MARKET_DRIVER_TICKERS = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "S&P 500": "^GSPC",
    "US 10Y Yield": "^TNX",
    "Energy ETF XLE": "XLE",
    "Oil ETF USO": "USO"
}


def load_market_drivers(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Télécharge les principaux drivers macro et marché liés au pétrole.

    Paramètres
    ----------
    start_date : str
        Date de début au format 'YYYY-MM-DD'.

    end_date : str
        Date de fin au format 'YYYY-MM-DD'.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les niveaux des drivers de marché.
    """

    # Liste des tickers Yahoo Finance
    tickers = list(MARKET_DRIVER_TICKERS.values())

    # Téléchargement des données avec yfinance
    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    # Si aucune donnée n'est récupérée, on retourne un tableau vide
    if raw_data.empty:
        return pd.DataFrame()

    # Avec plusieurs tickers, yfinance renvoie des colonnes multi-niveaux
    if isinstance(raw_data.columns, pd.MultiIndex):

        # On prend les prix ajustés si disponibles
        if "Adj Close" in raw_data.columns.get_level_values(0):
            drivers = raw_data["Adj Close"].copy()
        else:
            drivers = raw_data["Close"].copy()

    else:
        drivers = raw_data[["Close"]].copy()

    # Renommage des colonnes pour avoir des noms lisibles
    reverse_mapping = {
        ticker: name
        for name, ticker in MARKET_DRIVER_TICKERS.items()
    }

    drivers = drivers.rename(columns=reverse_mapping)

    # Suppression des colonnes entièrement vides
    drivers = drivers.dropna(axis=1, how="all")

    # Suppression des lignes entièrement vides
    drivers = drivers.dropna(how="all")

    # Conversion de l'index en date
    drivers.index = pd.to_datetime(drivers.index)

    return drivers


def compute_driver_returns(drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les variations journalières des drivers.

    Pour les indices comme le VIX, le DXY ou le S&P 500,
    on utilise un rendement simple :

    r_t = P_t / P_{t-1} - 1

    Paramètres
    ----------
    drivers : pd.DataFrame
        Tableau contenant les niveaux des drivers.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les variations journalières.
    """

    # Calcul des rendements journaliers
    driver_returns = drivers.pct_change()

    # Suppression des lignes vides
    driver_returns = driver_returns.dropna(how="all")

    return driver_returns


def compute_correlation_with_oil(
    oil_returns: pd.Series,
    driver_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcule la corrélation entre les rendements du WTI et les drivers de marché.

    Paramètres
    ----------
    oil_returns : pd.Series
        Rendements journaliers du WTI.

    driver_returns : pd.DataFrame
        Rendements journaliers des drivers.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les corrélations avec le WTI.
    """

    # On aligne les dates entre le WTI et les drivers
    aligned_data = driver_returns.copy()
    aligned_data["WTI"] = oil_returns

    # Suppression des lignes incomplètes
    aligned_data = aligned_data.dropna()

    # Calcul des corrélations avec le WTI
    correlations = aligned_data.corr()["WTI"].drop("WTI")

    # Conversion en DataFrame
    correlation_table = correlations.reset_index()
    correlation_table.columns = ["Driver", "Correlation with WTI"]

    # Tri par corrélation absolue décroissante
    correlation_table["Abs Correlation"] = correlation_table["Correlation with WTI"].abs()
    correlation_table = correlation_table.sort_values(
        by="Abs Correlation",
        ascending=False
    )

    # On retire la colonne technique
    correlation_table = correlation_table.drop(columns=["Abs Correlation"])

    return correlation_table


def compute_rolling_correlations(
    oil_returns: pd.Series,
    driver_returns: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Calcule les corrélations glissantes entre le WTI et chaque driver.

    Paramètres
    ----------
    oil_returns : pd.Series
        Rendements journaliers du WTI.

    driver_returns : pd.DataFrame
        Rendements journaliers des drivers.

    window : int
        Fenêtre de corrélation glissante.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les corrélations glissantes.
    """

    # Tableau qui contiendra les corrélations
    rolling_corrs = pd.DataFrame(index=driver_returns.index)

    # On calcule la corrélation glissante pour chaque driver
    for column in driver_returns.columns:

        rolling_corrs[column] = oil_returns.rolling(window).corr(
            driver_returns[column]
        )

    # Suppression des lignes entièrement vides
    rolling_corrs = rolling_corrs.dropna(how="all")

    return rolling_corrs


def build_market_drivers_summary(
    drivers: pd.DataFrame,
    driver_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Construit un résumé des drivers de marché.

    Métriques :
    - dernier niveau ;
    - variation sur la période ;
    - volatilité annualisée.

    Paramètres
    ----------
    drivers : pd.DataFrame
        Niveaux des drivers.

    driver_returns : pd.DataFrame
        Rendements des drivers.

    Retour
    ------
    pd.DataFrame
        Tableau résumé.
    """

    rows = []

    for column in drivers.columns:

        # On ignore les colonnes sans données
        if drivers[column].dropna().empty:
            continue

        first_value = drivers[column].dropna().iloc[0]
        last_value = drivers[column].dropna().iloc[-1]

        total_change = last_value / first_value - 1
        annualized_volatility = driver_returns[column].std() * (252 ** 0.5)

        rows.append({
            "Driver": column,
            "Latest Value": last_value,
            "Total Change": total_change,
            "Annualized Volatility": annualized_volatility
        })

    summary = pd.DataFrame(rows)

    return summary