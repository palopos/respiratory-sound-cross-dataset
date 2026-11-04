# Patient-independent cross-dataset evaluation of respiratory sound classifiers

Code and patient-level partitions for:

> P. López-Osorio and D. Sánchez-Morillo. *Patient-Independent Evaluation Reveals Direction- and
> Architecture-Dependent Cross-Dataset Transfer in Respiratory Sound Classification.* Applied Sciences (under review).

The study compares three model families for event-level normal-versus-adventitious classification on ICBHI 2017 and
SPRSound, within each dataset and in both transfer directions, under strict patient-independent partitions: a CNN
trained from scratch on log-Mel spectrograms, a trainable head on frozen YAMNet embeddings, and fine-tuned wav2vec 2.0.
Checkpoints and decision thresholds are selected on validation data only; test sets are used only for the final
evaluation.

- Code (this repository), archived on Zenodo: https://doi.org/10.5281/zenodo.23961520
- Event-level predictions of every reported run, run configurations and manifests, and analysis outputs:
  https://doi.org/10.5281/zenodo.23961572

## Contents

| Path | Content |
|---|---|
| `codigo/preprocess/` | Event segmentation, spectrograms (CNN), YAMNet and wav2vec 2.0 embeddings |
| `codigo/train/` | Training and evaluation of every reported configuration |
| `codigo/fase2/` | Patient-level cluster bootstrap, calibration, duration/prevalence/metadata/subtype analyses |
| `codigo/fase3/` | Alternative patient partitions, shared-label training, cross-dataset re-evaluation |
| `codigo/fase4/` | Tables, figures and supplementary material of the manuscript; embedding shift and domain distance |
| `codigo/analysis/` | Figures 1–2 and Table 1; frozen-embedding and domain-distance analyses of the original submission |
| `codigo/r1_padding_*.py` | Proportion of padded events and YAMNet padding comparison (Section 3.5) |
| `ficheros/splits/` | Patient and recording lists of every partition (seeds 42, 101, 202, 303) |
| `environment.yml` | Conda environment |

Code comments and log messages are in Spanish.

## Data

The audio is not redistributed. Download the datasets from their providers:

- ICBHI 2017 Respiratory Sound Database: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- SPRSound: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound

and place them as follows (paths are relative to the repository root):

```
ficheros/0 ICBHI/<recording>.wav, <recording>.txt
ficheros/0 SPRSound/train_wav/, train_json/, test_wav/, test_json/
```

`ficheros/splits/0 <dataset>/splits_patient_independent_seed<S>.csv` lists, for every recording used, its partition,
patient identifier, metadata and expected audio path. Recordings without a recoverable patient identifier and
SPRSound recordings annotated as Poor Quality are excluded (see the manuscript, Section 2.2).

To check the analyses without retraining, download the data archive (https://doi.org/10.5281/zenodo.23961572) and
extract it at the repository root; it uses the same relative paths (`modelos/`, `ficheros/predicciones_originales/`,
`ficheros/figuras_originales/`, `analisis/`).

## Environment

```
conda env create -f environment.yml
conda activate fix-paper1
```

All scripts are run from the repository root. The shell scripts in `codigo/fase3/` use the interpreter in the
`PYTHON` environment variable (default `python`) and write logs to `logs/` (`mkdir -p logs` first).

## Workflow

1. **Segmentation, seed-42 partition and spectrograms**:
   `codigo/preprocess/cnn/Preprocess_CNN_Scratch_ICBHI.py`, `..._SPRSound.py` (and `..._Fusion.py` for Appendix A).
2. **Embeddings**
   - YAMNet with each padding scheme: `codigo/preprocess/yamnet/r1_Extract_YAMNet_Padding.py <icbhi|sprsound> <loop|zero|noise>`;
     events of at least 0.975 s only: `r1_Build_YAMNet_Min0975.py`.
   - Original zero-padded YAMNet embeddings (frozen baselines, Mixup, CORAL): `Preprocess_YAMNet_Frozen_{ICBHI,SPRSound,Fusion}.py`;
     CORAL inputs: `Prepare_YAMNet_CORAL_*.py`.
   - Frozen wav2vec 2.0: `codigo/preprocess/wav2vec2/Preprocess_Wav2Vec2_Frozen_*.py`.
3. **Alternative partitions and shared labels**: `codigo/fase3/r1_make_partitions.py` (seeds 101, 202, 303; seed 42
   reproduces the primary partition), `codigo/fase3/r1_make_common_labels.py` (SPRSound without rhonchi and stridor).
4. **Training** (training seeds 42, 43, 44 unless stated otherwise)
   - CNN: `codigo/train/cnn/Train_CNN_Scratch_{ICBHI,SPRSound}.py`; ICBHI → SPRSound: `Train_CNN_Scratch_ICBHI_to_SPRSound.py`;
     SPRSound → ICBHI on the full ICBHI test set: `codigo/fase3/Reevaluate_CNN_Scratch_SPRSound_to_ICBHI.py`;
     alternative partitions: `codigo/fase3/r1_run_cnn_partitions.sh`; shared labels: `codigo/fase3/r1_run_cnn_commonlabels.sh`.
   - YAMNet trainable head: `codigo/train/yamnet/r1_Train_YAMNet_Head_Padding.py <protocol> <loop|zero|noise|min0975|loopcommon>`
     (protocol: `icbhi_to_icbhi`, `icbhi_to_sprsound`, `sprsound_to_sprsound`, `sprsound_to_icbhi`);
     alternative partitions: `r1_Train_YAMNet_Head_Partition.py <protocol> <split_seed>`;
     zero-padded runs of the original submission: `Train_YAMNet_TrainableHead_*.py`;
     Mixup: `Train_YAMNet_TrainableHead_ICBHI_to_SPRSound_Mixup_Confirm.py`; CORAL: `Train_YAMNet_TrainableHead_CORAL_ICBHI_to_SPRSound.py`.
   - Fine-tuned wav2vec 2.0: `codigo/train/wav2vec2/r1_Train_Wav2Vec2_Fixed_{ICBHI_to_SPRSound,SPRSound_to_ICBHI}.py`,
     evaluation on the other test set: `r1_Wav2Vec2_Fixed_Evaluate_{ICBHI,SPRSound}_Test.py`, pilot run:
     `r1_Train_Wav2Vec2_Smoke.py`; original configuration (Supplementary Table S1): `Train_Wav2Vec2_Finetune_*.py`.
   - Frozen embeddings with logistic regression: `codigo/train/yamnet/Train_YAMNet_Frozen_*.py`, `codigo/train/wav2vec2/Train_Wav2Vec2_Frozen_*.py`.
   - Pooled-data (Fusion) runs: the `*_Fusion*.py` scripts of each family.
5. **Analysis**: `codigo/fase2/r1_fase2_patient_bootstrap.py`, then `r1_fase2_calibration.py` and `r1_fase2_subgroups.py`;
   `codigo/fase3/r1_fase3_*.py`; `codigo/fase4/r1_fase4_embedding_shift.py {zero|loop}` and
   `r1_fase4_domain_distance.py {zero|loop}`; finally the table, figure and supplementary scripts below.

Each script documents its inputs, outputs and usage in its header.

## Manuscript items

| Item | Script | Output |
|---|---|---|
| Figures 1–2, Table 1 | `codigo/analysis/Build_Paper_Methods_Assets.py` | figures and dataset summary |
| Table 2 | architecture defined in `codigo/train/cnn/Train_CNN_Scratch_*.py` | — |
| Table 3, Figure 3 | `codigo/fase4/r1_fase4_tabla_principal.py`, `r1_fase4_figuras.py` | `analisis/r1_20261101_fase4/` |
| Table 4 | `codigo/fase3/r1_fase3_partition_summary.py`, `codigo/fase4/r1_fase4_tabla_particiones.py` | `analisis/r1_20261029_fase3/`, `analisis/r1_20261101_fase4/` |
| Patient-level intervals and paired differences | `codigo/fase2/r1_fase2_patient_bootstrap.py` | `analisis/r1_20261101_fase2_v2/` |
| Section 3.5 (padding) | `codigo/r1_padding_label_table.py`, `codigo/r1_padding_ensemble_metrics.py` | `analisis/r1_20261029_padding/` |
| Section 3.6 (shared labels) | `codigo/fase3/r1_fase3_commonlabels_compare.py` | `analisis/r1_20261029_fase3/` |
| Section 3.7 (calibration) | `codigo/fase2/r1_fase2_calibration.py` | `analisis/r1_20261101_fase2_v2/` |
| Sections 3.8–3.10 (duration, prevalence, metadata strata, sound type) | `codigo/fase2/r1_fase2_subgroups.py` | `analisis/r1_20261101_fase2_v2/` |
| Figures 4–5 | `codigo/fase4/r1_fase4_embedding_shift.py`, `r1_fase4_domain_distance.py`, `r1_fase4_figuras.py` | `analisis/r1_20261101_fase4/` |
| Table A1 | Fusion training scripts | predictions in the Zenodo archive |
| Supplementary Tables S1–S2, Figure S1 | `codigo/fase4/r1_fase4_suplementario.py` | `analisis/r1_20261101_fase4/suplementario/` |

## License

The code is released under the MIT License (see `LICENSE`). The datasets are subject to the terms of their providers.

## Citation

If you use this code, please cite the article above (citation details will be added on publication) and the archived
code (https://doi.org/10.5281/zenodo.23961520); see `CITATION.cff`.
