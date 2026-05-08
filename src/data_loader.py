# src/data_loader.py

# Importation de pandas pour manipuler les tableaux de données
import pandas as pd

# Importation de yfinance pour télécharger les données de marché
import yfinance as yf


# Dictionnaire qui associe un nom clair à chaque ticker Yahoo Finance
OIL_TICKERS = {
    "WTI": "CL=F",
    "Brent": "BZ=F"
}


def load_oil_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Télécharge les prix historiques du WTI et du Brent depuis Yahoo Finance.

    Paramètres
    ----------
    start_date : str
        Date de début au format 'YYYY-MM-DD'.

    end_date : str
        Date de fin au format 'YYYY-MM-DD'.

    Retour
    ------
    pd.DataFrame
        Tableau contenant les prix du WTI et du Brent.
    """

    # On récupère uniquement les tickers Yahoo Finance
    tickers = list(OIL_TICKERS.values())

    # Téléchargement des données depuis Yahoo Finance
    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    # Si aucune donnée n'est récupérée, on arrête proprement
    if raw_data.empty:
        raise ValueError("Aucune donnée n'a été récupérée. Vérifie les dates ou la connexion internet.")

    # Avec plusieurs tickers, yfinance renvoie souvent des colonnes multi-niveaux
    # On récupère ici les prix de clôture ajustés si disponibles, sinon les prix de clôture classiques
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Adj Close" in raw_data.columns.get_level_values(0):
            prices = raw_data["Adj Close"].copy()
        else:
            prices = raw_data["Close"].copy()
    else:
        prices = raw_data[["Close"]].copy()

    # On renomme les colonnes pour avoir des noms plus lisibles dans le dashboard
    prices = prices.rename(columns={
        "CL=F": "WTI",
        "BZ=F": "Brent"
    })

    # On supprime les lignes où toutes les valeurs sont manquantes
    prices = prices.dropna(how="all")

    # On s'assure que l'index est bien au format date
    prices.index = pd.to_datetime(prices.index)

    return prices