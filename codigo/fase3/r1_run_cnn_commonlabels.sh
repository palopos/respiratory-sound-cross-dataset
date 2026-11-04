#!/bin/zsh
# R1 Fase 3: CNN SPRSound sin Rhonchi/Stridor + evaluacion en el test ICBHI original. Uso desde la raiz del repositorio.
PY="${PYTHON:-python}"
SETUP=cnn/cnn_biocas-resnext-attention_scratch_trainable-mlp/split_seed_42
t=$(date +%s)
DISABLE_TQDM=1 $PY codigo/train/cnn/r1_Train_CNN_CommonLabels.py > logs/r1_20261029_cnn_commonlabels_train.log 2>&1
rc=$?; echo "$(date '+%F %T') train commonlabels rc=$rc $(( $(date +%s)-t ))s"
[ $rc -ne 0 ] && exit $rc
t=$(date +%s)
R1_INTRA_DIR="modelos/r1-20261029-cnn-common-labels/sprsound-to-sprsound/$SETUP/r1_20261029_commonlabels" \
R1_CROSS_DIR="modelos/r1-20261029-cnn-common-labels/sprsound-to-icbhi/$SETUP/r1_20261029_commonlabels_crosseval" \
R1_TARGET_TEST="./ficheros/2 Espectrogramas/r1_comunes/0 ICBHI_seed42_resize_dinamico_MEL64_SIZE64/npy/segmentado_test" \
DISABLE_TQDM=1 $PY codigo/fase3/r1_CNN_Cross_Evaluate.py sprsound 42 > logs/r1_20261029_cnn_commonlabels_crosseval.log 2>&1
echo "$(date '+%F %T') crosseval commonlabels rc=$? $(( $(date +%s)-t ))s"
