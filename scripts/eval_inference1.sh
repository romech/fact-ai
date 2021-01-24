python train_inference_attack.py 
    --attack inference1 \
    --gpus 1 \
    --classifier_weights <PATH_TO_BASELINE_NET_WEIGHTS> \ 
    --inversion_net_weights <PATH_TO_INVERSION_NET_WEIGHTS> \
    --angle_dis_weights <PATH_TO_ANGLE_DIS_WEIGHTS> \
    --encoder_weights <PATH_TO_COMPLEX_NET_WEIGHTS> \
    --test 
