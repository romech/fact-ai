python train_attack.py \
    --experiment inv2_complex \
    --gpus 1 \
    --precision 16 \
    --benchmark \
    --weights output/resnet20-complex.ckpt  \
    --max_epochs 15 \
    --complex
