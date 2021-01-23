python train_attack.py \
    --attack feature_inversion \
    --experiment feature_inversion_complex \
    --gpus 1 \
    --precision 16 \
    --benchmark \
    --encoder_weights <PATH_TO_PRETAINED_WEIGHTS> \
    --max_epochs 15 \
    --complex
