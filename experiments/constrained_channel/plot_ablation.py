"""Plot exact-channel benchmark timings and extrapolations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def duration(seconds):
    for size, unit in ((365 * 86400, "anos"), (86400, "dias"), (3600, "h"), (60, "min")):
        if seconds >= size:
            return f"{seconds / size:.2g} {unit}"
    return f"{seconds:.2g} s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text(encoding="utf-8"))
    measured, projected = data["measured"], data["projected"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([p["digits"] for p in measured], [p["seconds"] / 60 for p in measured],
            "o-", color="#136d68", label="Medido")
    extension = [measured[-1], *projected]
    ax.plot([p["digits"] for p in extension], [p["seconds"] / 60 for p in extension],
            "--", color="#a84e23", label="Projeção")
    ax.scatter([p["digits"] for p in projected], [p["seconds"] / 60 for p in projected],
               marker="o", facecolors="white", edgecolors="#a84e23")
    for point in measured + projected:
        ax.annotate(duration(point["seconds"]), (point["digits"], point["seconds"] / 60),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks([p["digits"] for p in measured + projected],
                  ["2 números", "2 + 1 dígito", "2 + 2 dígitos",
                   *[f"{p['digits'] // 3} números" for p in projected]])
    ax.set_yscale("log")
    ax.set_ylabel("Tempo de enumeração (min, escala log)")
    ax.set_xlabel("Observação do adversário")
    ax.set_title("Custo do canal exato · Qwen2.5-7B-Instruct")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.margins(x=0.10, y=0.15)
    fig.text(0.5, 0.015, "Projeção assume custo por prefixo igual ao medido no oitavo dígito.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(args.report.with_name(f"timing.{extension}"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
