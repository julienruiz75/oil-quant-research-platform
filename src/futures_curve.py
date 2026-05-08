# src/futures_curve.py

# Importation de numpy pour les calculs numériques
import numpy as np

# Importation de pandas pour manipuler les données
import pandas as pd


def load_futures_curve(file_path: str) -> pd.DataFrame:
    """
    Charge une courbe de futures depuis un fichier CSV.

    Le fichier doit contenir les colonnes :
    - contract : nom du contrat, par exemple CL1, CL2, CL3 ;
    - maturity : date de maturité du contrat ;
    - price : prix du contrat future.

    Paramètres
    ----------
    file_path : str
        Chemin du fichier CSV.

    Retour
    ------
    pd.DataFrame
        Tableau contenant la courbe de futures.
    """

    # Lecture du fichier CSV
    curve = pd.read_csv(file_path)

    # Conversion de la colonne maturity en date
    curve["maturity"] = pd.to_datetime(curve["maturity"])

    # Tri par date de maturité pour être sûr que la courbe est dans le bon ordre
    curve = curve.sort_values("maturity").reset_index(drop=True)

    return curve


def enrich_futures_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des colonnes utiles à la courbe de futures.

    Colonnes ajoutées :
    - months_to_maturity ;
    - price_change_vs_front ;
    - pct_change_vs_front.

    Paramètres
    ----------
    curve : pd.DataFrame
        Courbe de futures avec les colonnes maturity et price.

    Retour
    ------
    pd.DataFrame
        Courbe enrichie.
    """

    # Copie pour éviter de modifier le tableau original
    curve = curve.copy()

    # Date de maturité du premier contrat
    first_maturity = curve["maturity"].iloc[0]

    # Prix du premier contrat
    front_price = curve["price"].iloc[0]

    # Nombre de mois approximatif entre chaque maturité et la première maturité
    curve["months_to_maturity"] = (
        (curve["maturity"].dt.year - first_maturity.year) * 12
        + (curve["maturity"].dt.month - first_maturity.month)
    )

    # Écart en dollars par rapport au premier contrat
    curve["price_change_vs_front"] = curve["price"] - front_price

    # Écart en pourcentage par rapport au premier contrat
    curve["pct_change_vs_front"] = curve["price"] / front_price - 1

    return curve


def detect_curve_regime(curve: pd.DataFrame) -> str:
    """
    Détecte le régime de la courbe de futures.

    Si le dernier contrat est plus cher que le premier, la courbe est en contango.
    Si le dernier contrat est moins cher que le premier, la courbe est en backwardation.

    Paramètres
    ----------
    curve : pd.DataFrame
        Courbe de futures contenant une colonne price.

    Retour
    ------
    str
        Régime de marché : Contango, Backwardation ou Flat.
    """

    # Prix du premier contrat
    front_price = curve["price"].iloc[0]

    # Prix du dernier contrat
    last_price = curve["price"].iloc[-1]

    # Comparaison entre le contrat court et le contrat long
    if last_price > front_price:
        return "Contango"
    elif last_price < front_price:
        return "Backwardation"
    else:
        return "Flat"


def compute_curve_slope(curve: pd.DataFrame) -> float:
    """
    Calcule la pente de la courbe entre le premier et le dernier contrat.

    Formule :
    slope = F_long / F_front - 1

    Paramètres
    ----------
    curve : pd.DataFrame
        Courbe de futures contenant une colonne price.

    Retour
    ------
    float
        Pente totale de la courbe.
    """

    # Prix du premier contrat
    front_price = curve["price"].iloc[0]

    # Prix du dernier contrat
    last_price = curve["price"].iloc[-1]

    # Pente relative
    slope = last_price / front_price - 1

    return slope


def compute_annualized_roll_yield(curve: pd.DataFrame) -> float:
    """
    Estime le roll yield annualisé entre le premier et le dernier contrat.

    Idée :
    Si la courbe est en backwardation, le roll yield est généralement positif.
    Si la courbe est en contango, le roll yield est généralement négatif.

    Formule simplifiée :
    roll_yield = - slope * 12 / number_of_months

    Paramètres
    ----------
    curve : pd.DataFrame
        Courbe de futures enrichie avec months_to_maturity.

    Retour
    ------
    float
        Roll yield annualisé estimé.
    """

    # Pente entre le premier et le dernier contrat
    slope = compute_curve_slope(curve)

    # Nombre de mois entre le premier et le dernier contrat
    number_of_months = curve["months_to_maturity"].iloc[-1]

    # Sécurité si le nombre de mois vaut 0
    if number_of_months == 0:
        return np.nan

    # Roll yield annualisé simplifié
    roll_yield = -slope * 12 / number_of_months

    return roll_yield


def compute_curve_summary(curve: pd.DataFrame) -> dict:
    """
    Calcule un résumé de la courbe de futures.

    Paramètres
    ----------
    curve : pd.DataFrame
        Courbe de futures.

    Retour
    ------
    dict
        Résumé contenant le régime, la pente et le roll yield.
    """

    # Courbe enrichie
    curve = enrich_futures_curve(curve)

    # Régime de marché
    regime = detect_curve_regime(curve)

    # Pente de la courbe
    slope = compute_curve_slope(curve)

    # Roll yield annualisé estimé
    roll_yield = compute_annualized_roll_yield(curve)

    return {
        "regime": regime,
        "slope": slope,
        "roll_yield": roll_yield
    }