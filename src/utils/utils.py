import yaml
import sys
from sklearn.preprocessing import normalize
import numpy as np


def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise Exception(e,sys) from e

def cosine_similarities_sparse(tfidf_matrix, user_profile: np.ndarray):
    """
    Compute cosine similarities between a dense user profile (1 x D)
    and all books using sparse, row-normalized TF-IDF (N x D).
    Returns a dense 1-D numpy array of length N.
    """
    tfidf_norm = normalize(tfidf_matrix, norm='l2', axis=1, copy=True)
    up = user_profile.astype(np.float32, copy=True)
    norm = np.linalg.norm(up)
    if norm > 0:
        up = up / norm
    sims = tfidf_norm @ up  # (N, D) @ (D,) -> (N,)
    return np.asarray(sims).ravel()