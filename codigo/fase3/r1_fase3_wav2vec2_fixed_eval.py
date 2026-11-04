"""R1 Fase 3 (R3.3, R4.CS2): wav2vec 2.0 afinado con el ajuste corregido, 3 semillas, origen SPRSound e ICBHI (A1).

1) Criterios de no colapso por semilla (validacion y test): DE y rango de las probabilidades, F1 de validacion.
2) Ensemble (media de semillas) con umbral val_best_f1 de validacion; AUROC/AUPRC/F1 e IC 95 % por paciente.
3) Diferencias pareadas frente a YAMNet head (loop) y CNN sobre el mismo test (particion 42), con remuestras
   por paciente compartidas (B = 2000): test ICBHI (SPRSound->ICBHI) y test oficial de SPRSound (mismos
   checkpoints, r1_Wav2Vec2_Fixed_Evaluate_SPRSound_Test.py); con origen ICBHI, test SPRSound (ICBHI->SPRSound)
   y test ICBHI (mismos checkpoints, r1_Wav2Vec2_Fixed_Evaluate_ICBHI_Test.py).

Uso: python codigo/fase3/r1_fase3_wav2vec2_fixed_eval.py   (desde la raiz del repositorio)
Salidas: analisis/r1_20261029_fase3/wav2vec2_fixed_{semillas,ensemble,ensemble_sprsound_test}.csv (origen SPRSound)
         y wav2vec2_fixed_{semillas_icbhi,ensemble_icbhi_to_sprsound,ensemble_icbhi_to_icbhi}.csv (origen ICBHI)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_fase3_partition_summary import patients_for, run_dir  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fase2"))
from r1_fase2_common import load_ensemble, point_metrics, val_best_f1_threshold  # noqa: E402

OUT = Path("analisis/r1_20261029_fase3")
B, SEED = 2000, 20261029
W2V_DIR = Path("modelos/r1-20261029-wav2vec2-finetune-fixed/sprsound-to-icbhi/wav2vec2/"
               "wav2vec2_facebook-wav2vec2-base_finetuned_sequence-classification-head_full/split_seed_42/"
               "r1_20261029_fixed")
W2V_SPR_DIR = Path(str(W2V_DIR).replace("sprsound-to-icbhi", "sprsound-to-sprsound") + "_sprsoundtest")
W2V_ICBHI_DIR = Path(str(W2V_DIR).replace("r1-20261029", "r1-20261030").replace("sprsound-to-icbhi", "icbhi-to-sprsound")
                     .replace("r1_20261029_fixed", "r1_20261030_fixed"))
W2V_ICBHI_INTRA_DIR = Path(str(W2V_ICBHI_DIR).replace("icbhi-to-sprsound", "icbhi-to-icbhi") + "_icbhitest")
SEED_TABLES = {"wav2vec2_fixed_semillas.csv": W2V_DIR, "wav2vec2_fixed_semillas_icbhi.csv": W2V_ICBHI_DIR}
PROTOCOLS = {  # protocolo -> (fichero de salida, modelos)
    "sprsound_to_icbhi": ("wav2vec2_fixed_ensemble.csv", {
        "wav2vec2 fine-tuned (R1 fixed)": W2V_DIR,
        "YAMNet head (loop)": run_dir("YAMNet head (loop)", "sprsound_to_icbhi", 42),
        "CNN scratch": run_dir("CNN scratch", "sprsound_to_icbhi", 42)}),
    "sprsound_to_sprsound": ("wav2vec2_fixed_ensemble_sprsound_test.csv", {
        "wav2vec2 fine-tuned (R1 fixed)": W2V_SPR_DIR,
        "YAMNet head (loop)": run_dir("YAMNet head (loop)", "sprsound_to_sprsound", 42),
        "CNN scratch": run_dir("CNN scratch", "sprsound_to_sprsound", 42)}),
    "icbhi_to_sprsound": ("wav2vec2_fixed_ensemble_icbhi_to_sprsound.csv", {
        "wav2vec2 fine-tuned (R1 fixed)": W2V_ICBHI_DIR,
        "YAMNet head (loop)": run_dir("YAMNet head (loop)", "icbhi_to_sprsound", 42),
        "CNN scratch": run_dir("CNN scratch", "icbhi_to_sprsound", 42)}),
    "icbhi_to_icbhi": ("wav2vec2_fixed_ensemble_icbhi_to_icbhi.csv", {
        "wav2vec2 fine-tuned (R1 fixed)": W2V_ICBHI_INTRA_DIR,
        "YAMNet head (loop)": run_dir("YAMNet head (loop)", "icbhi_to_icbhi", 42),
        "CNN scratch": run_dir("CNN scratch", "icbhi_to_icbhi", 42)}),
}


def seed_rows(w2v_dir):
    rows = []
    for seed_dir in sorted(w2v_dir.glob("seed_*")):
        val = pd.read_csv(seed_dir / "predictions" / "predicciones_validation.csv")
        test = pd.read_csv(seed_dir / "predictions" / "predicciones_test.csv")
        prob_col = next(c for c in ("prob_adventitious", "prob", "y_prob", "probability") if c in val.columns)
        yv, pv = val["y_true"].to_numpy(), val[prob_col].to_numpy()
        yt, pt = test["y_true"].to_numpy(), test[prob_col].to_numpy()
        thr = val_best_f1_threshold(yv, pv)
        rows.append({"seed": seed_dir.name, "val_auroc": roc_auc_score(yv, pv),
                     "val_f1_at_thr": f1_score(yv, pv >= thr), "val_prob_sd": pv.std(),
                     "val_prob_range": pv.max() - pv.min(), "val_threshold": thr,
                     "test_auroc": roc_auc_score(yt, pt), "test_f1": f1_score(yt, pt >= thr),
                     "test_prob_sd": pt.std(), "test_prob_range": pt.max() - pt.min(),
                     "test_frac_positive": (pt >= thr).mean(), "n_val": len(yv), "n_test": len(yt)})
    return pd.DataFrame(rows)


def compare(protocol, out_name, models):
    loaded = {}
    for name, d in models.items():
        _, test, thr = load_ensemble(d)
        loaded[name] = (test.set_index("stem").sort_index(), thr)
    ref = loaded["wav2vec2 fine-tuned (R1 fixed)"][0]
    for name, (t, _) in loaded.items():
        if not t.index.equals(ref.index) or not (t["y_true"] == ref["y_true"]).all():
            raise ValueError(f"{name}: el test no coincide con el de wav2vec2")
    y = ref["y_true"].to_numpy()
    pid = patients_for(ref.reset_index(), protocol, 42)
    groups = list(pd.Series(range(len(y))).groupby(pid).indices.values())
    rng = np.random.default_rng(SEED)
    idxs = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]

    reps, rows = {}, []
    for name, (t, thr) in loaded.items():
        p = t["prob"].to_numpy()
        reps[name] = np.array([roc_auc_score(y[i], p[i]) for i in idxs])
        m = point_metrics(y, p, thr)
        rows.append({"protocol": protocol, "model": name, "n_events": len(y), "n_patients": len(groups), "prevalence": y.mean(),
                     "val_threshold": thr, "auroc": m["auroc"], "auroc_lo": np.percentile(reps[name], 2.5),
                     "auroc_hi": np.percentile(reps[name], 97.5), "auprc": m["auprc"], "f1": m["f1"],
                     "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                     "f1_all_adventitious": 2 * y.mean() / (1 + y.mean())})
    df = pd.DataFrame(rows)
    w2v = "wav2vec2 fine-tuned (R1 fixed)"
    for i, name in enumerate(df["model"]):
        diff = reps[w2v] - reps[name]
        df.loc[i, "delta_vs_wav2vec2"] = df.loc[0, "auroc"] - df.loc[i, "auroc"] if name != w2v else np.nan
        df.loc[i, "delta_lo"] = np.percentile(diff, 2.5) if name != w2v else np.nan
        df.loc[i, "delta_hi"] = np.percentile(diff, 97.5) if name != w2v else np.nan
    df["run_dir"] = [str(models[n]) for n in df["model"]]
    df.to_csv(OUT / out_name, index=False)
    with pd.option_context("display.width", 240):
        print(df.drop(columns="run_dir").round(4).to_string(index=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for out_name, w2v_dir in SEED_TABLES.items():
        seeds = seed_rows(w2v_dir)
        seeds.to_csv(OUT / out_name, index=False)
        with pd.option_context("display.width", 240):
            print(seeds.round(4).to_string(index=False))
    for protocol, (out_name, models) in PROTOCOLS.items():
        compare(protocol, out_name, models)


if __name__ == "__main__":
    main()
