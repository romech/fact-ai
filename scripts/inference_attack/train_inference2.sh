python train_inference_attack.py 
    --attack inference2 \
    --complex \
    --gpus 1 \
    --precision 16 \
    --angle_dis_weights <PATH_TO_ANGLE_DIS_CHECKPOINT> \
    --encoder_weights <PATH_TO_COMPLEX_NET_CHECKPOINT>  \
    --arch resnet56 \
    --dataset cifar10 \
    --batch_size 128 \
    --optimizer adam \
    --lr 0.001 \
    --schedule step \
    --steps 100 150 \
    --step_factor 0.1 \
