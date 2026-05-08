# src/data_quality.py

# Importation de pandas pour manipuler les données
import pandas as pd


def remove_non_positive_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les prix inférieurs ou égaux à zéro.

    Cette fonction est utile pour le WTI, car le contrat front-month
    est devenu négatif en avril 2020. Les rendements simples deviennent
    alors difficiles à interpréter pour la VaR, la CVaR et les mesures de risque.

    Paramètres
    ----------
    prices : pd.DataFrame
        Tableau contenant les prix.

    Retour
    ------
    pd.DataFrame
        Tableau avec les prix strictement positifs uniquement.
    """

    # Copie du tableau pour éviter de modifier les données originales
    cleaned_prices = prices.copy()

    # On remplace les prix inférieurs ou égaux à zéro par des valeurs manquantes
    cleaned_prices = cleaned_prices.where(cleaned_prices > 0)

    # On supprime les lignes où toutes les colonnes sont manquantes
    cleaned_prices = cleaned_prices.dropna(how="all")

    return cleaned_prices


def cap_extreme_returns(
    returns: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99
) -> pd.DataFrame:
    """
    Limite les rendements extrêmes en utilisant des quantiles.

    Cette technique s'appelle la winsorisation.
    Elle permet de réduire l'impact de valeurs extrêmes sur les métriques de risque.

    Paramètres
    ----------
    returns : pd.DataFrame
        Tableau contenant les rendements.

    lower_quantile : float
        Quantile inférieur.

    upper_quantile : float
        Quantile supérieur.

    Retour
    ------
    pd.DataFrame
        Tableau de rendements winsorisés.
    """

    # Copie des rendements
    capped_returns = returns.copy()

    # On applique la méthode colonne par colonne
    for column in capped_returns.columns:

        # Calcul des bornes
        lower_bound = capped_returns[column].quantile(lower_quantile)
        upper_bound = capped_returns[column].quantile(upper_quantile)

        # Limitation des valeurs extrêmes
        capped_returns[column] = capped_returns[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    return capped_returns