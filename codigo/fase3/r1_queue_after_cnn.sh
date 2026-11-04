#!/bin/zsh
# R1 Fase 3: cola en MPS. Espera a que termine la CNN por particiones, luego CNN con etiquetas comunes
# y despues wav2vec 2.0 con el ajuste corregido (SPRSound -> ICBHI, 3 semillas). Uso desde la raiz del repositorio.
LOG=logs/r1_20261029_queue.log
while pgrep -f r1_run_cnn_partitions.sh > /dev/null; do sleep 60; done
echo "$(date '+%F %T') CNN particiones terminada" >> $LOG
zsh codigo/fase3/r1_run_cnn_commonlabels.sh >> $LOG 2>&1
t=$(date +%s)
"${PYTHON:-python}" -u codigo/train/wav2vec2/r1_Train_Wav2Vec2_Fixed_SPRSound_to_ICBHI.py > logs/r1_20261029_wav2vec2_fixed_sprsound_to_icbhi.log 2>&1
echo "$(date '+%F %T') wav2vec2 fixed rc=$? $(( $(date +%s)-t ))s" >> $LOG
