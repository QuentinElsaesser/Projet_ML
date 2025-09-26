from typing import Union, List
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, List[List[float]]]

def my_to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convertit en numpy array 2D si possible.
    Accepts: np.ndarray, pd.DataFrame, pd.Series, list-of-lists
    Returns: np.ndarray (2D)
    """
    if isinstance(x, pd.DataFrame):
        return x.values
    if isinstance(x, pd.Series):
        arr = x.values
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x
    # assume list-like
    arr = np.array(x)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr