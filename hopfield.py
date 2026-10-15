import random
import numpy as np


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    def train_hebb(self, patterns):
        self.W = np.zeros((self.size, self.size))
        N = self.size
        for p in patterns:
            v = p.flatten().astype(float)
            self.W += np.outer(v, v) / N
        np.fill_diagonal(self.W, 0)

    def train_pseudo_inverse(self, patterns):
        X = np.array([p.flatten() for p in patterns])
        G = X @ X.T
        G_inv = np.linalg.pinv(G)
        self.W = X.T @ G_inv @ X
        np.fill_diagonal(self.W, 0)

    def recall(self, x, max_iter=200):
        s = x.flatten().copy()
        for _ in range(max_iter):
            idx = list(range(self.size))
            random.shuffle(idx)
            changed = False
            for i in idx:
                h = np.dot(self.W[i], s)
                new_val = 1.0 if h > 0 else (-1.0 if h < 0 else s[i])
                if new_val != s[i]:
                    s[i] = new_val
                    changed = True
            if not changed:
                break
        return s.reshape(x.shape)

    def energy(self, x):
        v = x.flatten()
        return float(-0.5 * v @ self.W @ v)


class NoiseGenerator:
    @staticmethod
    def apply(flat_pattern, percent):
        flat = flat_pattern.flatten().copy()
        n = len(flat)
        k = int(round(n * percent / 100.0))
        if k == 0:
            return flat.reshape(flat_pattern.shape)
        idx = np.random.choice(n, size=min(k, n), replace=False)
        flat[idx] = -flat[idx]
        return flat.reshape(flat_pattern.shape)


class Recognizer:
    def __init__(self, templates):
        self.templates = templates

    def best_match(self, result):
        best_name = None
        best_dist = 10 ** 9
        for name, t in self.templates.items():
            d = int(np.sum(result != t))
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name, best_dist

    def exact_match(self, result):
        for name, t in self.templates.items():
            if np.array_equal(result, t):
                return name
        return None