import numpy as np
from numpy.linalg import norm


def a_factor(mean_img: np.ndarray, target_img: np.ndarray) -> float:
    """
    Compute the a = x1 . x2 / |x1|^2
    :param mean_img: reference image;
    :param target_img: target image;
    :return: a / n.
    """
    ref_img = mean_img
    n = norm(ref_img)**2
    a = np.dot(ref_img.flatten(), target_img.flatten())
    return a / n


def b_factor(mean_img: np.ndarray, target_img: np.ndarray) -> float:
    """
    Compute the b = \sum_{i}^{N} = (x1i * y2i - y1i * x2i) / |x1|^2
    :param mean_img: reference image;
    :param target_img: target image;
    :return: b / n.
    """
    ref_img = mean_img
    n = norm(ref_img)**2
    b = np.sum(ref_img[:, 0] * target_img[:, 1] - ref_img[:, 1] * target_img[:, 0])
    return b / n


def R(s: float, theta: float) -> np.ndarray:
    """
    Compute the Rotation Matrix to apply to single shapes.
    :param s: scaling factor;
    :param theta: rotation angle;
    :return: rotation_mtx.
    """
    rotation_mtx = np.array([
        [s * np.cos(theta), -s * np.sin(theta)],
        [s * np.sin(theta), s * np.cos(theta)]
    ])
    return rotation_mtx