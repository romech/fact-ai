python train_attack.py \
    --attack feature_inversion_angle \
    --gpus 1 \
    --precision 16 \
    --benchmark \
    --encoder_weights <PATH_TO_COMPLEX_NETWORK_WEIGHTS> \
    --angle_dis_weights <PATH_TO_ANGLE_DIS_WEIGHTS> \
    --max_epochs 15 \
