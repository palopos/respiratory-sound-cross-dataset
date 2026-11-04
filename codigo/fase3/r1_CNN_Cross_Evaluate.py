"""R1 Fase 3: evalua en el test del otro dataset los checkpoints CNN de una ejecucion intra-dataset.

No reentrena ni reselecciona: reutiliza el checkpoint elegido en validacion de origen por semilla y copia
las predicciones de validacion de la ejecucion intra (mismo checkpoint, mismo umbral de validacion).
Uso: python codigo/fase3/r1_CNN_Cross_Evaluate.py <icbhi|sprsound> <semilla_particion>   (desde la raiz del repositorio)
Variables opcionales R1_INTRA_DIR, R1_CROSS_DIR, R1_TARGET_TEST para otras ejecuciones (p. ej. etiquetas comunes).
Cuerpo dentro de main(): los workers del DataLoader (spawn en macOS) reimportan este modulo.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

DATASETS = {"icbhi": "0 ICBHI", "sprsound": "0 SPRSound"}
EXPECTED_TEST = {"icbhi": None, "sprsound": 2991}


def main():
    source, split_seed = sys.argv[1], int(sys.argv[2])
    target = "sprsound" if source == "icbhi" else "icbhi"
    run_tag = os.environ.get("R1_RUN_TAG", "r1_20261029")
    setup = "cnn/cnn_biocas-resnext-attention_scratch_trainable-mlp"
    intra_dir = Path(os.environ.get("R1_INTRA_DIR") or f"modelos/{run_tag.replace('_', '-')}-cnn-partitions/{source}-to-{source}/{setup}/split_seed_{split_seed}/{run_tag}_split{split_seed}")
    cross_dir = Path(os.environ.get("R1_CROSS_DIR") or f"modelos/{run_tag.replace('_', '-')}-cnn-partitions/{source}-to-{target}/{setup}/split_seed_{split_seed}/{run_tag}_split{split_seed}_crosseval")
    target_test = os.environ.get("R1_TARGET_TEST") or f"./ficheros/2 Espectrogramas/r1_particiones/{DATASETS[target]}_seed{split_seed}_resize_dinamico_MEL64_SIZE64/npy/segmentado_test"
    if not intra_dir.exists():
        raise SystemExit(f"No existe la ejecucion intra {intra_dir}")
    if cross_dir.exists():
        raise SystemExit(f"Ya existe {cross_dir}; no se sobrescribe")

    os.environ["DATASET_NAME"] = DATASETS[source]
    os.environ["DATASET_LABEL"] = source
    os.environ["TRAIN_DATASET_LABEL"] = source
    os.environ["VALIDATION_DATASET_LABEL"] = source
    os.environ["TEST_DATASET_LABEL"] = target
    os.environ["DATASET_PROTOCOL"] = f"{source}_to_{target}"
    os.environ["SPLIT_SEED"] = str(split_seed)
    os.environ["EXPERIMENT_BLOCK"] = f"{run_tag}_cnn_partitions"
    os.environ["RUN_ID"] = f"{run_tag}_split{split_seed}_crosseval"
    os.environ["TEST_NPY_PATH"] = target_test
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "cnn"))
    if source == "icbhi":
        import Train_CNN_Scratch_ICBHI as P
    else:
        import Train_CNN_Scratch_SPRSound as P

    test_ds = P.BioCASNumpyDataset(target_test)
    test_loader = P.make_dataloader(test_ds, batch_size=P.BATCH_SIZE, shuffle=False)
    manifest = {"intra_run": str(intra_dir), "target_test": target_test, "n_test": len(test_ds), "seeds": {}}
    for seed_dir in sorted(intra_dir.glob("seed_*")):
        ckpt = next((seed_dir / "checkpoints").glob("best_validation_*.pth"))
        out = cross_dir / seed_dir.name
        for sub in ("predictions", "reports", "figures"):
            (out / sub).mkdir(parents=True)
        shutil.copy2(seed_dir / "predictions" / "predicciones_validation.csv", out / "predictions" / "predicciones_validation.csv")
        P.metricas_informes(modelo=str(ckpt), loader=test_loader, split_name="test",
                            excel_file=str(out / "reports" / "metricas.xlsx"),
                            predictions_file=str(out / "predictions" / "predicciones_test.csv"),
                            figure_file=str(out / "figures" / "analisis_test.png"), seed=int(seed_dir.name.split("_")[1]))
        manifest["seeds"][seed_dir.name] = {"checkpoint": str(ckpt), "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
    with open(cross_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f">>> Evaluacion cruzada guardada en {cross_dir} ({len(test_ds)} eventos de test)")


if __name__ == "__main__":
    main()
