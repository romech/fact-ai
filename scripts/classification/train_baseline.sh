python train_baseline.py \
    --gpus 1 \
    --precision 16 \
    --arch resnet20 \
    --max_epochs 200 \
    --schedule step \
    --steps 100 150 \
    --step_factor 0.1 \
    --dataset cifar10 \
    --batch_size 128 \
    --optimizer sgd \
    --momentum 0.9 \
    --lr 0.1 \

