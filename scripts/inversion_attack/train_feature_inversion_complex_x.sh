python train_inversion_attack.py \
    --attack feature_inversion \
    --complex \
    --gpus 1 \
    --precision 16 \
    --encoder_weights output/complex.ckpt  \
    --max_epochs 15 \
    --batch_size 128 \
    --lr 0.0001 \
    --dataset cifar10 \
