import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from tkinter import font as tkfont
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from hopfield import HopfieldNetwork, NoiseGenerator, Recognizer


class ImageCanvas:
    def __init__(self, parent, grid, cell=28):
        self.cell = cell
        self.h, self.w = grid.shape
        self.canvas = tk.Canvas(
            parent,
            width=self.w * cell,
            height=self.h * cell,
            highlightthickness=1,
            highlightbackground="#888",
            bg="white",
        )
        self.draw(grid)

    def draw(self, grid):
        self.canvas.delete("all")
        for i in range(self.h):
            for j in range(self.w):
                color = "black" if grid[i, j] > 0 else "white"
                x0 = j * self.cell
                y0 = i * self.cell
                self.canvas.create_rectangle(
                    x0, y0, x0 + self.cell, y0 + self.cell,
                    fill=color, outline=color
                )

    def pack(self, **kw):
        self.canvas.pack(**kw)


class MainWindow:
    def __init__(self, root, patterns, templates, net, recognizer):
        self.root = root
        self.patterns = patterns
        self.templates = templates
        self.net = net
        self.recognizer = recognizer
        self.noise = NoiseGenerator()

        self.last_analysis = None

        self.root.title("Сеть Хопфилда — Мартынов (М, А, Р)")
        self.root.geometry("1280x860")
        self.root.configure(bg="#f0f0f0")

        self._build_ui()
        self.log("Сеть обучена на образы: М, А, Р (5×7, 35 нейронов)")
        self.log("Метод обучения: правило Хебба\n")
        self.log("Попарные расстояния Хэмминга между эталонами:")
        for a, b, d, total in self.patterns.pairwise_hamming():
            self.log(f"  {a} <-> {b}: {d} из {total}")
        self.log("")

    def _build_ui(self):
        title_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")

        controls = tk.Frame(self.root, bg="#f0f0f0", pady=8)
        controls.pack(fill="x", padx=10)

        tk.Label(controls, text="Буква:", bg="#f0f0f0",
                 font=title_font).pack(side="left")
        self.combo = ttk.Combobox(controls, values=self.patterns.names,
                                  state="readonly", width=5)
        self.combo.current(0)
        self.combo.pack(side="left", padx=(4, 14))

        tk.Label(controls, text="Зашумление (%):", bg="#f0f0f0",
                 font=title_font).pack(side="left")
        self.percent_var = tk.IntVar(value=20)
        tk.Spinbox(controls, from_=0, to=100, textvariable=self.percent_var,
                   width=5).pack(side="left", padx=(4, 14))

        tk.Button(controls, text="Подать зашумлённый образ",
                  command=self.on_test_single,
                  bg="#4a90d9", fg="white", padx=8, pady=4).pack(side="left", padx=4)

        tk.Button(controls, text="100 тестов для этого %",
                  command=self.on_run_stats,
                  bg="#5cb85c", fg="white", padx=8, pady=4).pack(side="left", padx=4)

        tk.Button(controls, text="Полный анализ 10–100%",
                  command=self.on_full_analysis,
                  bg="#d9534f", fg="white", padx=8, pady=4).pack(side="left", padx=4)

        tk.Button(controls, text="Показать график",
                  command=self.on_show_plot,
                  bg="#f0ad4e", fg="white", padx=8, pady=4).pack(side="left", padx=4)

        tk.Button(controls, text="Сохранить график",
                  command=self.on_save_plot,
                  bg="#8e44ad", fg="white", padx=8, pady=4).pack(side="left", padx=4)

        vis_frame = tk.LabelFrame(self.root, text="Результат распознавания",
                                  bg="#f0f0f0", padx=10, pady=10)
        vis_frame.pack(fill="x", padx=10, pady=6)

        titles = ["Эталон", "Зашумлённый", "Восстановленный"]
        self.views = []
        for idx, t in enumerate(titles):
            col = tk.Frame(vis_frame, bg="#f0f0f0")
            col.grid(row=0, column=idx, padx=20)
            tk.Label(col, text=t, bg="#f0f0f0",
                     font=title_font).pack(pady=(0, 4))
            view = ImageCanvas(col, self.templates["М"])
            view.pack()
            self.views.append(view)

        bottom = tk.Frame(self.root, bg="#f0f0f0")
        bottom.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        log_frame = tk.LabelFrame(bottom, text="Журнал",
                                  bg="#f0f0f0", padx=6, pady=6)
        log_frame.pack(side="left", fill="both", expand=True)

        self.log_box = scrolledtext.ScrolledText(
            log_frame, height=18, font=("Consolas", 10), wrap="word"
        )
        self.log_box.pack(fill="both", expand=True)

        plot_frame = tk.LabelFrame(bottom, text="График точности",
                                   bg="#f0f0f0", padx=6, pady=6)
        plot_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

        self.fig = Figure(figsize=(5.4, 4.2), dpi=100, facecolor="#f0f0f0")
        self.ax = self.fig.add_subplot(111)
        self._draw_empty_plot()

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plot_canvas.draw()

        toolbar = NavigationToolbar2Tk(self.plot_canvas, plot_frame)
        toolbar.update()

    def _draw_empty_plot(self):
        self.ax.clear()
        self.ax.set_title("Точность распознавания vs. зашумление",
                          fontsize=11, fontweight="bold")
        self.ax.set_xlabel("Зашумление, %")
        self.ax.set_ylabel("Точность, %")
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 105)
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.text(50, 50, "Нажмите «Полный анализ 10–100%»",
                     ha="center", va="center", fontsize=11, color="#888")

    def log(self, text):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.root.update_idletasks()

    def on_test_single(self):
        name = self.combo.get()
        percent = self.percent_var.get()

        template = self.templates[name]
        noisy = self.noise.apply(template, percent)
        recalled = self.net.recall(noisy)
        matched, dist = self.recognizer.best_match(recalled)

        self.views[0].draw(template)
        self.views[1].draw(noisy)
        self.views[2].draw(recalled)

        exact = "точное" if np.array_equal(recalled, template) else "неточное"
        ok = "OK" if matched == name else "FAIL"
        self.log(f"[{name}] шум {percent}% -> распознано как '{matched}' "
                 f"(Хэмминг={dist}, {exact}) [{ok}]")

    def on_run_stats(self):
        percent = self.percent_var.get()
        trials = 100
        self.log(f"\n=== 100 тестов на каждый класс, шум {percent}% ===")
        total_correct = 0
        total_exact = 0
        total_trials = 0
        for name in self.patterns.names:
            correct = 0
            correct_exact = 0
            template = self.templates[name]
            for _ in range(trials):
                noisy = self.noise.apply(template, percent)
                recalled = self.net.recall(noisy)
                if np.array_equal(recalled, template):
                    correct_exact += 1
                    correct += 1
                else:
                    matched, _ = self.recognizer.best_match(recalled)
                    if matched == name:
                        correct += 1
            acc = 100.0 * correct / trials
            acc_exact = 100.0 * correct_exact / trials
            total_correct += correct
            total_exact += correct_exact
            total_trials += trials
            self.log(f"  Буква '{name}': распознано {correct}/{trials} "
                     f"({acc:.1f}%), точно совпало: "
                     f"{correct_exact}/{trials} ({acc_exact:.1f}%)")
        self.log(f"  ИТОГО по всем классам: распознано "
                 f"{total_correct}/{total_trials} "
                 f"({100.0 * total_correct / total_trials:.1f}%), "
                 f"точно {total_exact}/{total_trials} "
                 f"({100.0 * total_exact / total_trials:.1f}%)")

    def on_full_analysis(self):
        percents = [10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
        trials = 50
        self.log("\n=== Полный анализ по всем уровням шума "
                 "(50 тестов на класс) ===")
        header = (f"{'Шум,%':>6} | {'М точн':>7} | {'А точн':>7} | {'Р точн':>7} "
                  f"| {'М узнан':>8} | {'А узнан':>8} | {'Р узнан':>8} "
                  f"| {'Сред.узнан':>10}")
        self.log(header)
        self.log("-" * len(header))

        names = self.patterns.names
        exact_results = {n: [] for n in names}
        recog_results = {n: [] for n in names}
        averages_exact = []
        averages_recog = []

        for p in percents:
            row_exact = []
            row_recog = []
            for name in names:
                exact_cnt = 0
                recog_cnt = 0
                template = self.templates[name]
                for _ in range(trials):
                    noisy = self.noise.apply(template, p)
                    recalled = self.net.recall(noisy)
                    if np.array_equal(recalled, template):
                        exact_cnt += 1
                        recog_cnt += 1
                    else:
                        matched, _ = self.recognizer.best_match(recalled)
                        if matched == name:
                            recog_cnt += 1
                ve = 100.0 * exact_cnt / trials
                vr = 100.0 * recog_cnt / trials
                row_exact.append(ve)
                row_recog.append(vr)
                exact_results[name].append(ve)
                recog_results[name].append(vr)
            avg_e = sum(row_exact) / len(row_exact)
            avg_r = sum(row_recog) / len(row_recog)
            averages_exact.append(avg_e)
            averages_recog.append(avg_r)
            self.log(f"{p:>6} | {row_exact[0]:>6.0f}% | {row_exact[1]:>6.0f}% | "
                     f"{row_exact[2]:>6.0f}% | {row_recog[0]:>7.0f}% | "
                     f"{row_recog[1]:>7.0f}% | {row_recog[2]:>7.0f}% | "
                     f"{avg_r:>9.1f}%")

        self.last_analysis = {
            "percents": percents,
            "results": recog_results,
            "results_exact": exact_results,
            "averages": averages_recog,
            "averages_exact": averages_exact,
        }

        self._draw_plot()
        self.log("\nАнализ завершён. График обновлён.")

    def _draw_plot(self):
        if not self.last_analysis:
            return

        data = self.last_analysis
        p = data["percents"]
        colors = {"М": "#e74c3c", "А": "#3498db", "Р": "#27ae60"}

        self.ax.clear()
        self.ax.set_title("Точность распознавания vs. зашумление",
                          fontsize=12, fontweight="bold")
        self.ax.set_xlabel("Зашумление, %", fontsize=10)
        self.ax.set_ylabel("Точность распознавания, %", fontsize=10)

        for name in self.patterns.names:
            self.ax.plot(p, data["results"][name], marker="o",
                         linewidth=2, markersize=6,
                         color=colors.get(name, None),
                         label=f"Буква {name} (узнан.)")
            self.ax.plot(p, data["results_exact"][name], marker="^",
                         linewidth=1.4, markersize=5, linestyle=":",
                         color=colors.get(name, None), alpha=0.7,
                         label=f"Буква {name} (точно)")

        self.ax.plot(p, data["averages"], marker="s", linewidth=2.5,
                     markersize=7, linestyle="--", color="#2c3e50",
                     label="Среднее (узнан.)")

        self.ax.axhline(50, color="#888", linestyle=":", alpha=0.7)
        self.ax.text(2, 52, "порог 50%", fontsize=8, color="#666")

        self.ax.set_xlim(0, 105)
        self.ax.set_ylim(-3, 108)
        self.ax.set_xticks([10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

        self.fig.tight_layout()
        self.plot_canvas.draw()

    def on_show_plot(self):
        if not self.last_analysis:
            self.log("Сначала выполните «Полный анализ 10–100%».")
            return

        win = tk.Toplevel(self.root)
        win.title("График точности распознавания")
        win.geometry("900x650")

        fig = Figure(figsize=(8.5, 6), dpi=100)
        ax = fig.add_subplot(111)
        data = self.last_analysis
        p = data["percents"]
        colors = {"М": "#e74c3c", "А": "#3498db", "Р": "#27ae60"}

        for name in self.patterns.names:
            ax.plot(p, data["results"][name], marker="o",
                    linewidth=2, markersize=7,
                    color=colors.get(name, None),
                    label=f"Буква {name} (узнан.)")
            ax.plot(p, data["results_exact"][name], marker="^",
                    linewidth=1.4, markersize=6, linestyle=":",
                    color=colors.get(name, None), alpha=0.7,
                    label=f"Буква {name} (точно)")
        ax.plot(p, data["averages"], marker="s", linewidth=2.5,
                markersize=8, linestyle="--", color="#2c3e50",
                label="Среднее (узнан.)")

        ax.axhline(50, color="#888", linestyle=":", alpha=0.7)
        ax.set_title("Точность распознавания букв М, А, Р\n"
                     "сетью Хопфилда (35 нейронов) при зашумлении",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Процент зашумления, %", fontsize=11)
        ax.set_ylabel("Точность распознавания, %", fontsize=11)
        ax.set_xlim(0, 105)
        ax.set_ylim(-3, 108)
        ax.set_xticks(p)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, win).update()
        canvas.draw()

    def on_save_plot(self):
        if not self.last_analysis:
            self.log("Сначала выполните «Полный анализ 10–100%».")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG изображение", "*.png"),
                       ("PDF документ", "*.pdf"),
                       ("SVG вектор", "*.svg")],
            initialfile="hopfield_accuracy.png",
        )
        if not path:
            return

        fig = Figure(figsize=(9, 6), dpi=150)
        ax = fig.add_subplot(111)
        data = self.last_analysis
        p = data["percents"]
        colors = {"М": "#e74c3c", "А": "#3498db", "Р": "#27ae60"}

        for name in self.patterns.names:
            ax.plot(p, data["results"][name], marker="o",
                    linewidth=2, markersize=7,
                    color=colors.get(name, None),
                    label=f"Буква {name} (узнан.)")
            ax.plot(p, data["results_exact"][name], marker="^",
                    linewidth=1.4, markersize=6, linestyle=":",
                    color=colors.get(name, None), alpha=0.7,
                    label=f"Буква {name} (точно)")
        ax.plot(p, data["averages"], marker="s", linewidth=2.5,
                markersize=8, linestyle="--", color="#2c3e50",
                label="Среднее (узнан.)")

        ax.axhline(50, color="#888", linestyle=":", alpha=0.7)
        ax.set_title("Точность распознавания букв М, А, Р\n"
                     "сетью Хопфилда",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Процент зашумления, %", fontsize=11)
        ax.set_ylabel("Точность распознавания, %", fontsize=11)
        ax.set_xlim(0, 105)
        ax.set_ylim(-3, 108)
        ax.set_xticks(p)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
        fig.tight_layout()

        fig.savefig(path, dpi=150, bbox_inches="tight")
        self.log(f"График сохранён: {path}")