import math
import numpy as np
import itertools


def sigmoid(x: float):
    return 1.0 / (1.0 + np.exp(-x))


data_patterns = np.array(
    [
        [-1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, -1],
    ],
    dtype=int,
)


def get_data_distribution():
    return {tuple(p): 0.25 for p in data_patterns}


ALL_VISIBLE = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=int)


@dataclass
class RBMConfig:
    n_visible: int
    n_hidden: int
    lr: float = 0.05
    weight_scale: float = 0.1
    k: int = 1
    l2: float = 0.0
    seed: int = 0


class RBMPlusMinus:
    def __init__(self, cfg: RBMConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.W = rng.normal(0, cfg.weight_scale, size=(cfg.n_visible, cfg.n_hidden))
        self.a = np.zeros(cfg.n_visible)
        self.b = np.zeros(cfg.n_hidden)
        self.rng = rng
