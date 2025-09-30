import numpy as np

def parse_text_input(input_text: str) -> np.ndarray:
    """
    Parse un texte CSV-like (une ligne par exemple) en tableau numpy.
    Exemple :
        "1,2,3\n4,5,6" -> np.array([[1,2,3],[4,5,6]])
    Args:
        input_text: chaîne de caractères
    Returns:
        np.ndarray
    """
    return np.array([list(map(float, line.split(","))) for line in input_text.strip().split("\n")])
