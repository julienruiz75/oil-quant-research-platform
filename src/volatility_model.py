# src/volatility_model.py

# Importation de numpy pour les calculs numériques
import numpy as np

# Importation de pandas pour manipuler les séries temporelles
import pandas as pd

# Modèle de machine learning utilisé pour prédire la volatilité
from sklearn.ensemble import RandomForestRegressor

# Métriques d'évaluation du modèle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_volatility_dataset(
    prices: pd.Series,
    horizon: int = 10
) -> pd.DataFrame:
    """
    Construit un dataset de machine learning pour prédire la volatilité future du WTI.

    L'objectif est de prédire la volatilité réalisée future sur un horizon donné.

    Formule de la volatilité réalisée future :

    volatility_future_t = std(r_{t+1}, ..., r_{t+horizon}) * sqrt(252)

    Paramètres
    ----------
    prices : pd.Series
        Série des prix du WTI.

    horizon : int
        Horizon de prédiction en jours.

    Retour
    ------
    pd.DataFrame
        Dataset contenant les variables explicatives et la cible.
    """

    # Création du DataFrame principal
    data = pd.DataFrame(index=prices.index)

    # Prix du WTI
    data["price"] = prices

    # Rendements journaliers
    data["return_1d"] = prices.pct_change()

    # Rendements retardés pour capturer la dynamique récente
    data["return_lag_1"] = data["return_1d"].shift(1)
    data["return_lag_2"] = data["return_1d"].shift(2)
    data["return_lag_3"] = data["return_1d"].shift(3)

    # Rendements absolus, utiles car la volatilité dépend de l'amplitude des mouvements
    data["abs_return_1d"] = data["return_1d"].abs()

    # Volatilités réalisées historiques sur différentes fenêtres
    data["vol_5d"] = data["return_1d"].rolling(window=5).std() * np.sqrt(252)
    data["vol_10d"] = data["return_1d"].rolling(window=10).std() * np.sqrt(252)
    data["vol_20d"] = data["return_1d"].rolling(window=20).std() * np.sqrt(252)
    data["vol_60d"] = data["return_1d"].rolling(window=60).std() * np.sqrt(252)

    # Momentum du prix sur différentes fenêtres
    data["momentum_5d"] = data["price"] / data["price"].shift(5) - 1
    data["momentum_20d"] = data["price"] / data["price"].shift(20) - 1
    data["momentum_60d"] = data["price"] / data["price"].shift(60) - 1

    # Cible : volatilité future réalisée sur horizon jours
    future_volatility = (
        data["return_1d"]
        .shift(-1)
        .rolling(window=horizon)
        .std()
        .shift(-(horizon - 1))
        * np.sqrt(252)
    )

    data["target_future_volatility"] = future_volatility

    # Suppression des lignes incomplètes
    data = data.dropna()

    return data


def split_train_test(
    dataset: pd.DataFrame,
    target_column: str = "target_future_volatility",
    test_size: float = 0.25
):
    """
    Sépare les données en train et test sans mélanger les dates.

    En finance, on évite de faire un split aléatoire, car cela créerait
    un biais temporel. On entraîne sur le passé et on teste sur le futur.

    Paramètres
    ----------
    dataset : pd.DataFrame
        Dataset complet.

    target_column : str
        Nom de la colonne cible.

    test_size : float
        Proportion des données utilisées pour le test.

    Retour
    ------
    tuple
        X_train, X_test, y_train, y_test.
    """

    # Variables explicatives
    X = dataset.drop(columns=[target_column])

    # Variable cible
    y = dataset[target_column]

    # Point de séparation temporel
    split_index = int(len(dataset) * (1 - test_size))

    # Train = ancienne période
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    # Test = période récente
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_volatility_model(
    dataset: pd.DataFrame,
    target_column: str = "target_future_volatility"
):
    """
    Entraîne un modèle Random Forest pour prédire la volatilité future.

    Paramètres
    ----------
    dataset : pd.DataFrame
        Dataset de machine learning.

    target_column : str
        Nom de la colonne cible.

    Retour
    ------
    tuple
        model, predictions, metrics, feature_importance, test_data.
    """

    # Séparation train / test
    X_train, X_test, y_train, y_test = split_train_test(
        dataset=dataset,
        target_column=target_column,
        test_size=0.25
    )

    # Modèle Random Forest
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédictions sur la période de test
    y_pred = model.predict(X_test)

    # Tableau des prédictions
    predictions = pd.DataFrame(index=X_test.index)
    predictions["Realized Future Volatility"] = y_test
    predictions["Predicted Future Volatility"] = y_pred

    # Métriques d'évaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "R2 Score"
        ],
        "Value": [
            mae,
            rmse,
            r2
        ]
    })

    # Importance des variables
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    return model, predictions, metrics, feature_importance, X_test


def predict_latest_volatility(
    model,
    dataset: pd.DataFrame,
    target_column: str = "target_future_volatility"
) -> float:
    """
    Prédit la volatilité future à partir de la dernière ligne disponible.

    Paramètres
    ----------
    model :
        Modèle entraîné.

    dataset : pd.DataFrame
        Dataset complet.

    target_column : str
        Nom de la colonne cible.

    Retour
    ------
    float
        Dernière prédiction de volatilité future.
    """

    # Dernière ligne disponible sans la cible
    latest_features = dataset.drop(columns=[target_column]).iloc[[-1]]

    # Prédiction
    latest_prediction = model.predict(latest_features)[0]

    return latest_prediction