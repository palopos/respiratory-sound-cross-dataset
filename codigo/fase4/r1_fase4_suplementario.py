#!/usr/bin/env python3
"""R1 Fase 4: material suplementario (R4.CS6, R4.CS4, R3.5, R2.5/R3.6/R4.CS5).

Tabla S1  todas las configuraciones y protocolos: AUROC, AUPRC y F1 con IC 95 % por paciente, sensibilidad,
          especificidad y matriz de confusion (TN/FP/FN/TP) al umbral de validacion.
Tabla S2  calibracion frente a transferencia de umbral (ECE y Brier con IC por paciente, pendiente e intercepto,
          ECE tras Platt ajustado en validacion de origen, tasas de positivos predichos, F1 de referencia).
Figura S1 curvas de fiabilidad e histograma de probabilidades de las tres familias principales.

No recalcula nada: lee las fuentes unicas de la Fase 2 v2 y de la Fase 3.
  analisis/r1_20261101_fase2_v2/bootstrap_por_paciente.csv  (B = 2000, semilla 20261029)
  analisis/r1_20261101_fase2_v2/calibracion.csv             (IC de ECE y Brier: B = 1000, semilla 20261029)
  analisis/r1_20261029_fase3/etiquetas_comunes.csv          (entrenamiento sin roncus ni estridor)
La figura S1 vuelve a cargar las predicciones guardadas (mismo codigo que r1_fase2_calibration.py) para
dibujar con etiquetas limpias; reliability_bins y load_ensemble son los de la Fase 2.

Uso (desde la raiz del repositorio): python codigo/fase4/r1_fase4_suplementario.py
Salidas: analisis/r1_20261101_fase4/suplementario/{tabla_S1,tabla_S2}.{csv,tex}, figura S1 en
analisis/r1_20261101_fase4/figuras/ y manuscrito/r1/Figures/; los .tex de las tablas se copian en
manuscrito/r1/supplementary_tables/, que incluye manuscrito/r1/supplementary.tex (escrito a mano).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fase2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_fase2_calibration import reliability_bins  # noqa: E402
from r1_fase2_common import load_ensemble, model_registry  # noqa: E402
from r1_fase4_figuras import FAMILIES, FAMILY_COLORS, PROTOCOLS, WIDTH, setup  # noqa: E402

FASE2 = Path("analisis/r1_20261101_fase2_v2")
COMMON = Path("analisis/r1_20261029_fase3/etiquetas_comunes.csv")
OUT = Path("analisis/r1_20261101_fase4/suplementario")
FIG_OUT = Path("analisis/r1_20261101_fase4/figuras")
MANUSCRIPT = Path("manuscrito/r1")
ORDER = list(PROTOCOLS)
TEX_PROTOCOL = {k: v.replace("→", "$\\rightarrow$") for k, v in PROTOCOLS.items()}

# (bloque, etiqueta de la configuracion, modelo en las fuentes, protocolos admitidos o None = todos)
BLOCKS = {
    "A": "Principal configurations (Table 3)",
    "B": "Frozen-embedding baselines with logistic regression (zero-padded YAMNet input)",
    "C": "YAMNet trainable head: alternative input padding",
    "D": "Original wav2vec 2.0 fine-tuning configuration (replaced; see Methods)",
    "E": "Domain-adaptation screenings, ICBHI $\\rightarrow$ SPRSound (zero-padded YAMNet input)",
    "F": "Shared-label training: rhonchi and stridor removed from SPRSound training and validation",
}
CONFIGS = [
    ("A", "CNN (scratch)", "CNN scratch", None),
    ("A", "YAMNet trainable head, loop padding", "YAMNet trainable head (loop pad)", None),
    ("A", "wav2vec 2.0 fine-tuned", "wav2vec 2.0 fine-tuned (R1 fixed)", None),
    ("B", "YAMNet frozen + LR", "YAMNet frozen + LR", None),
    ("B", "wav2vec 2.0 frozen + LR", "wav2vec 2.0 frozen + LR", None),
    # Con origen ICBHI, las ejecuciones con ceros de tres semillas son las del envio (identicas bit a bit a la
    # replica R1, Fase 1); con origen SPRSound, el envio solo tenia una semilla y R1 anadio tres.
    ("C", "Zero padding$^{a}$", "YAMNet trainable head", ["icbhi_to_icbhi", "icbhi_to_sprsound"]),
    ("C", "Zero padding$^{a}$", "YAMNet trainable head (zero pad, R1 3 seeds)", None),
    ("C", "Zero padding, screening$^{a}$", "YAMNet trainable head", ["sprsound_to_sprsound", "sprsound_to_icbhi"]),
    ("C", "Low-level noise padding", "YAMNet trainable head (noise pad)", None),
    ("C", "Events $\\geq$0.975~s only$^{b}$", "YAMNet trainable head (events >=0.975 s)", None),
    ("D", "wav2vec 2.0 fine-tuned, original", "wav2vec 2.0 fine-tuned", None),
    ("E", "YAMNet head + Mixup", "YAMNet trainable head + Mixup", None),
    ("E", "YAMNet head + CORAL", "YAMNet trainable head + CORAL", None),
]
COMMON_LABELS = {"CNN scratch": "CNN (scratch)", "YAMNet head (loop)": "YAMNet trainable head, loop padding"}


def num(x: float, d: int) -> str:
    s = f"{x:.{d}f}"
    if float(s) == 0:  # sin "-0.00"
        s = s.lstrip("-")
    return f"$-${s[1:]}" if s.startswith("-") else s


def ci(point: float, lo: float, hi: float, d: int = 4) -> str:
    return f"{num(point, d)} [{num(lo, d)}, {num(hi, d)}]"


def select(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for i, (block, label, model, protocols) in enumerate(CONFIGS):
        sub = df[df["model"] == model]
        if protocols is not None:
            sub = sub[sub["protocol"].isin(protocols)]
        parts.append(sub.assign(block=block, config=label, config_order=i))
    out = pd.concat(parts, ignore_index=True)
    out["protocol_order"] = out["protocol"].map(ORDER.index)
    out["config_order"] = out.groupby(["block", "config"])["config_order"].transform("min")  # filas de una misma etiqueta juntas
    out = out.sort_values(["config_order", "protocol_order"]).reset_index(drop=True)
    if out.duplicated(["protocol", "model"]).any():
        raise ValueError("Fila repetida en la seleccion")
    return out


def body(df: pd.DataFrame, cells, ncols: int) -> str:
    lines, block, config = [], None, None
    for r in df.itertuples():
        if r.block != block:
            if block is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{r.block}. {BLOCKS[r.block]}}}}} \\\\")
            block, config = r.block, None
        row = cells(r)
        if r.config == config:  # la configuracion solo se escribe en su primera fila
            row[0] = ""
        config = r.config
        lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def table_s1(boot: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    df = select(boot)
    if len(df) != len(boot):
        raise ValueError(f"S1 deberia cubrir las {len(boot)} filas del bootstrap y tiene {len(df)}")
    shared = common[common["labels"] == "common"].copy()
    shared = shared.assign(block="F", config=shared["family"].map(COMMON_LABELS), protocol="sprsound_to_icbhi",
                           n_seeds=3, config_order=len(CONFIGS), protocol_order=ORDER.index("sprsound_to_icbhi"))
    df = pd.concat([df, shared], ignore_index=True)
    df.to_csv(OUT / "tabla_S1.csv", index=False)

    def cells(r):
        if r.block == "F":  # solo hay IC para AUROC
            return [r.config, TEX_PROTOCOL[r.protocol], f"{r.n_seeds:.0f}", num(r.val_threshold, 4),
                    ci(r.auroc, r.auroc_lo, r.auroc_hi), num(r.auprc, 4), num(r.f1, 4),
                    num(r.sensitivity, 3), num(r.specificity, 3), "--", "--", "--", "--"]
        return [r.config, TEX_PROTOCOL[r.protocol], f"{r.n_seeds:.0f}", num(r.val_threshold, 4),
                ci(r.auroc, r.auroc_patient_lo, r.auroc_patient_hi), ci(r.auprc, r.auprc_patient_lo, r.auprc_patient_hi),
                ci(r.f1, r.f1_patient_lo, r.f1_patient_hi), num(r.sensitivity, 3), num(r.specificity, 3),
                f"{r.tn:.0f}", f"{r.fp:.0f}", f"{r.fn:.0f}", f"{r.tp:.0f}"]

    (OUT / "tabla_S1.tex").write_text(body(df, cells, 13))
    return df


def table_s2(cal: pd.DataFrame) -> pd.DataFrame:
    df = select(cal)
    if len(df) != len(cal):
        raise ValueError(f"S2 deberia cubrir las {len(cal)} filas de calibracion y tiene {len(df)}")
    df.to_csv(OUT / "tabla_S2.csv", index=False)

    def cells(r):
        return [r.config, TEX_PROTOCOL[r.protocol], num(r.val_prevalence, 3), num(r.val_pred_positive_rate_at_val_thr, 3),
                num(r.test_prevalence, 3), num(r.test_pred_positive_rate_at_val_thr, 3),
                ci(r.test_ece, r.test_ece_patient_lo, r.test_ece_patient_hi, 3),
                ci(r.test_brier, r.test_brier_patient_lo, r.test_brier_patient_hi, 3),
                num(r.test_calib_slope, 2), num(r.test_calib_intercept, 2), num(r.test_platt_ece, 3),
                num(r.test_f1_at_val_thr, 3), num(r.oracle_test_f1, 3), num(r.test_f1_all_positive, 3)]

    (OUT / "tabla_S2.tex").write_text(body(df, cells, 14))
    return df


def figure_s1() -> None:
    setup()
    registry = model_registry().set_index(["protocol", "model"])
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.9), gridspec_kw={"height_ratios": [2.4, 1]})
    for j, (protocol, label) in enumerate(PROTOCOLS.items()):
        ax, axh = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
        for model, family in FAMILIES.items():
            val, test, thr = load_ensemble(registry.loc[(protocol, model), "run_dir"])
            yt, pt = test["y_true"].to_numpy(), test["prob"].to_numpy()
            b = pd.DataFrame(reliability_bins(yt, pt))
            b = b[b["n"] >= 10]
            ax.plot(b["mean_prob"], b["observed_rate"], "o-", color=FAMILY_COLORS[family], ms=2.5, lw=1.1, label=family)
            axh.hist(pt, bins=30, range=(0, 1), histtype="step", color=FAMILY_COLORS[family], lw=1.0)
            axh.axvline(thr, color=FAMILY_COLORS[family], ls=":", lw=1.1)
        ax.axhline(yt.mean(), color="0.3", lw=0.7, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"$\\bf{{{'ABCD'[j]}}}$  {label}", loc="left", fontsize=8.5, fontweight="normal")
        for a in (ax, axh):
            a.set_xticks([0, 0.5, 1])
            a.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        axh.set_xlabel("Predicted probability", fontsize=8)
        axh.set_xlim(0, 1)
        axh.set_yticks([])
        ax.tick_params(labelsize=7.5)
        axh.tick_params(labelsize=7.5)
        if j == 0:
            ax.set_ylabel("Observed adventitious rate", fontsize=8)
            axh.set_ylabel("Test events", fontsize=8)
        else:
            ax.set_yticklabels([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    stem = "r1_fig_S1_reliability"
    fig.savefig(FIG_OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if (MANUSCRIPT / "Figures").is_dir():
        shutil.copy2(FIG_OUT / f"{stem}.pdf", MANUSCRIPT / "Figures" / f"{stem}.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s1 = table_s1(pd.read_csv(FASE2 / "bootstrap_por_paciente.csv"), pd.read_csv(COMMON))
    s2 = table_s2(pd.read_csv(FASE2 / "calibracion.csv"))
    figure_s1()
    if MANUSCRIPT.is_dir():
        (MANUSCRIPT / "supplementary_tables").mkdir(exist_ok=True)
        for name in ("tabla_S1.tex", "tabla_S2.tex"):
            shutil.copy2(OUT / name, MANUSCRIPT / "supplementary_tables" / name)
    print(f"S1: {len(s1)} filas; S2: {len(s2)} filas -> {OUT}")


if __name__ == "__main__":
    main()
