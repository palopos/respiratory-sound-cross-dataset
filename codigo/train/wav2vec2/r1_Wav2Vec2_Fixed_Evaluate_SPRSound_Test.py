"""R1 Fase 3: evalua en el test oficial de SPRSound los checkpoints de wav2vec 2.0 corregido (SPRSound -> SPRSound).

No reentrena ni reselecciona: carga checkpoints/best_model de cada semilla de r1_20261029_fixed (elegido con
validacion de SPRSound) y copia sus predicciones de validacion (mismo checkpoint, mismo umbral val_best_f1).
Uso: python codigo/train/wav2vec2/r1_Wav2Vec2_Fixed_Evaluate_SPRSound_Test.py   (desde la raiz del repositorio)
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

SETUP = "wav2vec2/wav2vec2_facebook-wav2vec2-base_finetuned_sequence-classification-head_full/split_seed_42"
SOURCE_RUN = Path(f"modelos/r1-20261029-wav2vec2-finetune-fixed/sprsound-to-icbhi/{SETUP}/r1_20261029_fixed")
OUT_RUN = Path(f"modelos/r1-20261029-wav2vec2-finetune-fixed/sprsound-to-sprsound/{SETUP}/r1_20261029_fixed_sprsoundtest")
SPR_TEST = "./ficheros/1 Segmentados_WAV/0 SPRSound/seed_42/segmentado_test"
os.environ.update({
    "DATASET_NAME": "0 SPRSound", "DATASET_LABEL": "sprsound", "TRAIN_DATASET_LABEL": "sprsound",
    "VALIDATION_DATASET_LABEL": "sprsound", "TEST_DATASET_LABEL": "sprsound", "DATASET_PROTOCOL": "sprsound_to_sprsound",
    "EXPERIMENT_BLOCK": "r1_20261029_wav2vec2_finetune_fixed", "RUN_ID": "r1_20261029_fixed_sprsoundtest",
    "TEST_SEGMENTED_WAV_DIR": SPR_TEST, "HF_LOCAL_FILES_ONLY": "1", "WAV2VEC2_DEVICE": "mps", "DISABLE_TQDM": "1",
})

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification  # noqa: E402

import Train_Wav2Vec2_Finetune as P  # noqa: E402


def main():
    if OUT_RUN.exists():
        raise SystemExit(f"Ya existe {OUT_RUN}; no se sobrescribe")
    test_dataset = P.AudioSegmentDataset("test", SPR_TEST)
    manifest = {"source_run": str(SOURCE_RUN), "test_dir": SPR_TEST, "n_test": len(test_dataset), "seeds": {}}
    for seed_dir in sorted(SOURCE_RUN.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        ckpt = seed_dir / "checkpoints" / "best_model"
        P.set_seed(seed)
        feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt, local_files_only=True)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(ckpt, local_files_only=True)
        model.to(torch.device(P.DEVICE_NAME))
        collator = P.AudioBatchCollator(feature_extractor, P.compute_min_input_samples(model))
        loader = P.build_dataloader(test_dataset, collator, P.EVAL_BATCH_SIZE, shuffle=False)
        metrics = P.evaluate_model(model, loader, "test", None, P.use_attention_mask_for_model(model), seed)
        out = OUT_RUN / seed_dir.name / "predictions"
        out.mkdir(parents=True)
        shutil.copy2(seed_dir / "predictions" / "predicciones_validation.csv", out / "predicciones_validation.csv")
        preds = metrics["metadata"].copy()
        preds["seed"] = seed
        preds["y_true"] = metrics["y_true"]
        preds["prob_adventitious"] = metrics["probabilities"]
        preds.to_csv(out / "predicciones_test.csv", index=False)
        weights = next(ckpt.glob("*.safetensors"))
        manifest["seeds"][seed_dir.name] = {"checkpoint": str(ckpt),
                                            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        print(f">>> {seed_dir.name}: {len(preds)} eventos de test SPRSound evaluados")
        del model
        if P.DEVICE_NAME == "mps":
            torch.mps.empty_cache()
    with open(OUT_RUN / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f">>> Guardado en {OUT_RUN}")


if __name__ == "__main__":
    main()
