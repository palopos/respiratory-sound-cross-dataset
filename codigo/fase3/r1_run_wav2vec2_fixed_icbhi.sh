#!/bin/zsh
# R1 Fase 3 (opcion A1): wav2vec 2.0 corregido con origen ICBHI (3 semillas; test SPRSound) y reevaluacion de los
# mismos checkpoints en el test de ICBHI. Secuencial en MPS. Uso: zsh codigo/fase3/r1_run_wav2vec2_fixed_icbhi.sh (desde la raiz del repositorio)
PY="${PYTHON:-python}"
LOG=logs
t=$(date +%s)
$PY -u codigo/train/wav2vec2/r1_Train_Wav2Vec2_Fixed_ICBHI_to_SPRSound.py > $LOG/r1_20261030_wav2vec2_fixed_icbhi_to_sprsound.log 2>&1
rc=$?
echo "$(date '+%F %T') train wav2vec2 icbhi rc=$rc $(( $(date +%s)-t ))s" >> $LOG/r1_20261030_wav2vec2_icbhi_batch.log
[ $rc -ne 0 ] && exit $rc
t=$(date +%s)
$PY -u codigo/train/wav2vec2/r1_Wav2Vec2_Fixed_Evaluate_ICBHI_Test.py > $LOG/r1_20261030_wav2vec2_fixed_icbhitest.log 2>&1
rc=$?
echo "$(date '+%F %T') eval icbhi test rc=$rc $(( $(date +%s)-t ))s" >> $LOG/r1_20261030_wav2vec2_icbhi_batch.log
exit $rc
