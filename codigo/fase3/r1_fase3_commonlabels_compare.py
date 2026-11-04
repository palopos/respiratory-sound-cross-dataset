"""R1 Fase 3 (R2.4, R3.4): SPRSound->ICBHI entrenando SPRSound sin Rhonchi/Stridor (etiquetas comunes).

Compara, sobre el mismo test de ICBHI (particion 42), cada familia con etiquetas comunes frente a su linea base:
CNN (original del paper, 1414 eventos) y YAMNet head con relleno loop (Fase 1). Ensemble de 3 semillas, umbral
val_best_f1 de validacion, IC 95 % por paciente con remuestras compartidas (diferencia pareada, B = 2000).

Uso: python codigo/fase3/r1_fase3_commonlabels_compare.py   (desde la raiz del repositorio)
Salida: analisis/r1_20261029_fase3/etiquetas_comunes.csv
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_fase3_partition_summary import CNN_SETUP, YAMNET_SETUP, patients_for, run_dir  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fase2"))
from r1_fase2_common import load_ensemble, point_metrics  # noqa: E402

OUT = Path("analisis/r1_20261029_fase3")
B, SEED = 2000, 20261029
PAIRS = {
    "CNN scratch": (run_dir("CNN scratch", "sprsound_to_icbhi", 42),
                    Path(f"modelos/r1-20261029-cnn-common-labels/sprsound-to-icbhi/{CNN_SETUP}/split_seed_42/"
                         "r1_20261029_commonlabels_crosseval")),
    "YAMNet head (loop)": (run_dir("YAMNet head (loop)", "sprsound_to_icbhi", 42),
                           Path(f"modelos/r1-20261029-yamnet-head-padding-loopcommon/sprsound-to-icbhi/{YAMNET_SETUP}/"
                                "split_seed_42/r1_20261029_loopcommon")),
}


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for family, (base_dir, common_dir) in PAIRS.items():
        loaded = {}
        for name, d in (("base", base_dir), ("common", common_dir)):
            _, test, thr = load_ensemble(d)
            loaded[name] = (test.set_index("stem").sort_index(), thr)
        tb, tc = loaded["base"][0], loaded["common"][0]
        if not tb.index.equals(tc.index) or not (tb["y_true"] == tc["y_true"]).all():
            raise ValueError(f"{family}: los tests de linea base y etiquetas comunes no coinciden")
        y = tb["y_true"].to_numpy()
        pid = patients_for(tb.reset_index(), "sprsound_to_icbhi", 42)
        groups = list(pd.Series(range(len(y))).groupby(pid).indices.values())
        idxs = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]
        reps = {}
        for name, (t, thr) in loaded.items():
            p = t["prob"].to_numpy()
            reps[name] = np.array([roc_auc_score(y[i], p[i]) for i in idxs])
            m = point_metrics(y, p, thr)
            rows.append({"family": family, "labels": name, "n_events": len(y), "n_patients": len(groups),
                         "val_threshold": thr, "auroc": m["auroc"], "auroc_lo": np.percentile(reps[name], 2.5),
                         "auroc_hi": np.percentile(reps[name], 97.5), "auprc": m["auprc"], "f1": m["f1"],
                         "sensitivity": m["sensitivity"], "specificity": m["specificity"], "delta_auroc": np.nan,
                         "delta_lo": np.nan, "delta_hi": np.nan,
                         "run_dir": str(base_dir if name == "base" else common_dir)})
        diff = reps["common"] - reps["base"]
        rows[-1].update(delta_auroc=rows[-1]["auroc"] - rows[-2]["auroc"],
                        delta_lo=np.percentile(diff, 2.5), delta_hi=np.percentile(diff, 97.5))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "etiquetas_comunes.csv", index=False)
    with pd.option_context("display.width", 220):
        print(df.drop(columns="run_dir").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
