import tkinter as tk

from patterns import Patterns
from hopfield import HopfieldNetwork, Recognizer
from ui import MainWindow


def main():
    patterns = Patterns()
    patterns.pretty_print()
    for a, b, d, total in patterns.pairwise_hamming():
        print(f"  {a} <-> {b}: {d} пикселей из {total}")

    templates = {name: Patterns.to_bipolar(patterns.get(name))
                 for name in patterns.names}

    size = next(iter(templates.values())).size
    net = HopfieldNetwork(size=size)
    net.train_hebb(list(templates.values()))

    recognizer = Recognizer(templates)

    root = tk.Tk()
    MainWindow(root, patterns, templates, net, recognizer)
    root.mainloop()


if __name__ == "__main__":
    main()