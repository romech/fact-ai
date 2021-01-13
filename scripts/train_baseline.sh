python train_baseline.py \
    --gpus 1 \
    --precision 16 \
    --experiment_name test \
    --benchmark \
    --max_epochs 200 \
    --schedule step \
    --steps 100 150 \
