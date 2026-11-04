"""R1 Fase 4: tabla de sensibilidad a la particion por paciente (R2.3, R3.2) para el manuscrito.

Lee las salidas de codigo/fase3/r1_fase3_partition_summary.py (no recalcula nada): AUROC del ensemble de 3 semillas
por familia, protocolo y particion (42, 101, 202, 303), con IC 95 % por paciente, y la asimetria
AUROC(ICBHI->SPRSound) - AUROC(SPRSound->ICBHI) por particion.

Uso: python codigo/fase4/r1_fase4_tabla_particiones.py   (desde la raiz del repositorio)
Salida: analisis/r1_20261101_fase4/tabla_particiones.tex (cuerpo de la tabla LaTeX)
"""
from pathlib import Path

import pandas as pd

SRC = Path("analisis/r1_20261029_fase3")
OUT = Path("analisis/r1_20261101_fase4")
SEEDS = [42, 101, 202, 303]
ROWS = [("sprsound_to_sprsound", "SPRSound $\\rightarrow$ SPRSound"), ("icbhi_to_icbhi", "ICBHI $\\rightarrow$ ICBHI"),
        ("icbhi_to_sprsound", "ICBHI $\\rightarrow$ SPRSound"), ("sprsound_to_icbhi", "SPRSound $\\rightarrow$ ICBHI")]
FAMILIES = [("CNN scratch", "CNN (scratch)"), ("YAMNet head (loop)", "YAMNet trainable head")]


def signed(x):
    return f"{x:+.3f}".replace("-", "$-$")


def main():
    metrics = pd.read_csv(SRC / "particiones_metricas.csv")
    asym = pd.read_csv(SRC / "particiones_asimetria.csv")
    lines = []
    for family, label in FAMILIES:
        lines.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for protocol, plabel in ROWS:
            sub = metrics[(metrics["family"] == family) & (metrics["protocol"] == protocol)].set_index("partition_seed")
            cells = [f"{sub.loc[s, 'auroc']:.3f}" for s in SEEDS]
            lines.append("    " + " & ".join([plabel] + cells) + " \\\\")
        a = asym[asym["family"] == family].set_index("partition_seed")
        cells = [f"{signed(a.loc[s, 'asymmetry'])} [{signed(a.loc[s, 'asymmetry_lo'])}, {signed(a.loc[s, 'asymmetry_hi'])}]"
                 for s in SEEDS]
        lines.append("    " + " & ".join(["Asymmetry$^{a}$"] + cells) + " \\\\")
        lines.append("    \\midrule")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tabla_particiones.tex").write_text("\n".join(lines[:-1]) + "\n")
    print((OUT / "tabla_particiones.tex").read_text())


if __name__ == "__main__":
    main()
