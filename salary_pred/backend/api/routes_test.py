from fastapi import APIRouter, Request
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from backend.metrics.metrics import regression_metrics
from backend.schemas.regression_metrics_schemas import RegressionMetrics
from backend.manager.model_manager import ModelManager
from backend.utils.config import config

router = APIRouter(tags=["Test"])

@router.get("/test")
async def root(request: Request):
    return {"message": config.server.title, "version": config.server.version, "res": test_train_models()}

# fonction tets pour l'instant
def test_train_models():
    df = pd.read_csv(config.data.raw_path)
    
    X = df[['YearsExperience']].values  # dataframe -> numpy
    y = df['Salary'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.data.test_size,
        random_state=config.data.random_state
    )

    params = getattr(config.model, "params", {}) or {}
    manager = ModelManager(
        model_type=config.model.model_type,
        model_path=config.model.model_path,
        **params
    )
    train_metrics = manager.train_model(X_train, y_train)
    y_pred_my = manager.predict(X_test)

    # === Entraînement du modèle sklearn ===
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_sklearn = lr.predict(X_test)

    return "MY", RegressionMetrics(**regression_metrics(y_true=y_test, y_pred=y_pred_my)), \
            "SKL", RegressionMetrics(**regression_metrics(y_true=y_test, y_pred=y_pred_sklearn))

    # # === Comparaison ===
    # print("=== Evaluation My Linear Regression (test) ===")
    # print("MSE :", mean_squared_error(y_test, y_pred_my))
    # print("R2  :", r2_score(y_test, y_pred_my))

    # print("=== Evaluation Sklearn Linear Regression (test) ===")
    # print("MSE :", mean_squared_error(y_test, y_pred_sklearn))
    # print("R2  :", r2_score(y_test, y_pred_sklearn))