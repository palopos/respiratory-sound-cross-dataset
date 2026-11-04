"""R1 Fase 4: tabla principal del manuscrito revisado (R2.2, R3.2, R3.3, R4.CS1, R4.CS2).

Familias principales, todas ensemble de 3 semillas, particion 42, umbral val_best_f1 de validacion:
CNN desde cero (ejecuciones del paper), YAMNet trainable head con relleno loop (Fase 1) y wav2vec 2.0 afinado
con el ajuste corregido (Fase 3).

No recalcula el bootstrap: lee analisis/r1_20261101_fase2_v2/bootstrap_por_paciente.csv (r1_fase2_patient_bootstrap.py
con R1_FASE2_OUT; IC 95 % por paciente, B = 2000, remuestras compartidas por destino), que es la fuente unica de los IC
del texto, de la tabla y de las comparaciones pareadas (comparaciones_pareadas.csv de la misma carpeta).

Uso: python codigo/fase4/r1_fase4_tabla_principal.py   (desde la raiz del repositorio)
Salidas: analisis/r1_20261101_fase4/tabla_principal.csv y tabla_principal.tex (cuerpo de la tabla LaTeX)
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fase2"))
from r1_fase2_common import PROTOCOL_LABEL  # noqa: E402

SRC = Path("analisis/r1_20261101_fase2_v2/bootstrap_por_paciente.csv")
OUT = Path("analisis/r1_20261101_fase4")
FAMILIES = {"CNN scratch": "CNN (scratch)", "YAMNet trainable head (loop pad)": "YAMNet trainable head",
            "wav2vec 2.0 fine-tuned (R1 fixed)": "wav2vec 2.0 fine-tuned"}
ORDER = ["sprsound_to_sprsound", "icbhi_to_icbhi", "icbhi_to_sprsound", "sprsound_to_icbhi"]


def main():
    boot = pd.read_csv(SRC)
    df = boot[boot["model"].isin(FAMILIES)].copy()
    df["protocol"] = pd.Categorical(df["protocol"], ORDER, ordered=True)
    df["model_order"] = df["model"].map({m: i for i, m in enumerate(FAMILIES)})
    df = df.sort_values(["protocol", "model_order"]).drop(columns="model_order")
    if len(df) != 12 or (df["n_seeds"] != 3).any():
        raise ValueError("Se esperaban 12 filas (4 protocolos x 3 familias) con 3 semillas")
    df["model_label"] = df["model"].map(FAMILIES)
    df["auprc_lift"] = df["auprc"] / df["prevalence"]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tabla_principal.csv", index=False)

    lines = []
    for protocol in ORDER:
        sub = df[df["protocol"] == protocol]
        label = PROTOCOL_LABEL[protocol].replace("->", "$\\rightarrow$")
        for i, r in enumerate(sub.itertuples()):
            cells = [label if i == 0 else "", r.model_label, f"{r.val_threshold:.4f}"]
            cells += [f"{getattr(r, k):.4f} [{getattr(r, k + '_patient_lo'):.4f}, {getattr(r, k + '_patient_hi'):.4f}]"
                      for k in ("auroc", "auprc", "f1")]
            lines.append("    " + " & ".join(cells) + " \\\\")
        lines.append("    \\midrule")
    (OUT / "tabla_principal.tex").write_text("\n".join(lines[:-1]) + "\n")
    cols = ["protocol", "model_label", "auroc", "auroc_patient_lo", "auroc_patient_hi", "auprc", "auprc_patient_lo",
            "auprc_patient_hi", "f1", "f1_patient_lo", "f1_patient_hi", "auprc_lift", "sensitivity", "specificity"]
    with pd.option_context("display.width", 250):
        print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
