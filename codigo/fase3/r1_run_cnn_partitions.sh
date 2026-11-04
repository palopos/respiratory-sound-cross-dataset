#!/bin/zsh
# R1 Fase 3: CNN intra + evaluacion cruzada para las particiones 101, 202, 303. Secuencial (MPS).
# Uso: zsh codigo/fase3/r1_run_cnn_partitions.sh   (desde la raiz del repositorio)
PY="${PYTHON:-python}"
LOG=logs
for seed in 101 202 303; do
  for src in sprsound icbhi; do
    t=$(date +%s)
    DISABLE_TQDM=1 $PY codigo/train/cnn/r1_Train_CNN_Partition.py $src $seed > $LOG/r1_20261029_cnn_partition_${src}_split${seed}.log 2>&1
    rc=$?
    echo "$(date '+%F %T') train $src split$seed rc=$rc $(( $(date +%s)-t ))s"
    [ $rc -ne 0 ] && continue
    t=$(date +%s)
    DISABLE_TQDM=1 $PY codigo/fase3/r1_CNN_Cross_Evaluate.py $src $seed > $LOG/r1_20261029_cnn_crosseval_${src}_split${seed}.log 2>&1
    echo "$(date '+%F %T') crosseval $src split$seed rc=$? $(( $(date +%s)-t ))s"
  done
done
