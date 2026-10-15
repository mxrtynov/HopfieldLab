import numpy as np

LETTER_M = [
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
]

LETTER_A = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
]

LETTER_R = [
    [1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1],
]


class Patterns:
    def __init__(self):
        self._data = {
            "М": np.array(LETTER_M, dtype=float),
            "А": np.array(LETTER_A, dtype=float),
            "Р": np.array(LETTER_R, dtype=float),
        }

    @property
    def names(self):
        return list(self._data.keys())

    def get(self, name):
        return self._data[name].copy()

    def as_dict(self):
        return {k: v.copy() for k, v in self._data.items()}

    @staticmethod
    def to_bipolar(grid):
        return np.where(grid > 0.5, 1.0, -1.0)

    def pretty_print(self):
        for name in self.names:
            print(f"\nБуква '{name}':")
            for row in self._data[name]:
                print("  " + "".join("█" if v > 0.5 else "·" for v in row))

    def pairwise_hamming(self):
        names = self.names
        result = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = self._data[names[i]].flatten()
                b = self._data[names[j]].flatten()
                d = int(np.sum(a != b))
                result.append((names[i], names[j], d, len(a)))
        return result