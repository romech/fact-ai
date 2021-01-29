python train_complex.py \
    --gpus 1 \
    --precision 16 \
    --arch resnet20 \
    --resnet_variant alpha \
    --max_epochs 200 \
    --schedule step \
    --steps 100 150 \
    --step_factor 0.1 \
    --dataset cifar10 \
    --batch_size 128 \
    --optimizer adam \
    --lr 0.001 \

