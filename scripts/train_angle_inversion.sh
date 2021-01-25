python train_attack.py \
    --attack angle_inversion \
    --gpus 1 \ 
    --precision 16 \ 
    --benchmark \
    --encoder_weights <PATH_TO_PRETRAINED_WEIGHTS> \
    --max_epochs 30 \
    --lr 0.0001 \
    --schedule step \
    --steps 15 25
