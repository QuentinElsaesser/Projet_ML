from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from typing import Optional, Tuple
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from backend.models.sklearn_model_wrapper import SKLearnModelWrapper
from backend.models.basic_model import BasicModel
from backend.models.image_classifier import ImageClassifier

class RidgeModel(SKLearnModelWrapper):
    def __init__(self, alpha: float = 1.0, use_scaler: bool = True, **kwargs):
        super().__init__(Ridge(alpha=alpha, **kwargs), use_scaler=use_scaler)

class LassoModel(SKLearnModelWrapper):
    def __init__(self, alpha: float = 0.1, use_scaler: bool = True, **kwargs):
        super().__init__(Lasso(alpha=alpha, **kwargs), use_scaler=use_scaler)

class RandomForestModel(SKLearnModelWrapper):
    def __init__(self, n_estimators: int = 100, random_state: Optional[int] = 42, **kwargs):
        super().__init__(RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, **kwargs), use_scaler=False)
        
class XGBoostModel(SKLearnModelWrapper):
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.1, random_state: Optional[int] = 42, **kwargs):
        super().__init__(XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, **kwargs), use_scaler=False)

class LightGBMModel(SKLearnModelWrapper):
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.1, random_state: Optional[int] = 42, **kwargs):
        super().__init__(LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, **kwargs), use_scaler=False)

class MLPModel(SKLearnModelWrapper):
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (64, 32), max_iter: int = 300, random_state: Optional[int] = 42, **kwargs):
        super().__init__(MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state, **kwargs), use_scaler=True)
        
def create_regressor(name: str, **kwargs) -> BasicModel:
    """
    Factory to instantiate one of the supported model wrappers.
    name: one of 'ridge', 'lasso', 'random_forest', 'xgboost', 'lightgbm', 'mlp', ...
    kwargs: forwarded to model constructors.
    """
    name = name.lower()
    if name == "ridge":
        return RidgeModel(**kwargs)
    if name == "lasso":
        return LassoModel(**kwargs)
    if name == "random_forest":
        return RandomForestModel(**kwargs)
    if name == "xgboost":
        return XGBoostModel(**kwargs)
    if name == "lightgbm":
        return LightGBMModel(**kwargs)
    if name == "mlp":
        return MLPModel(**kwargs)
    if name == "image_classifier":
        return ImageClassifier(**kwargs)