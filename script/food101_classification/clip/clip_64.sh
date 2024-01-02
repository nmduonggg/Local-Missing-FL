python generate_fedtask.py \
    --benchmark food101_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing \

python main.py \
    --task food101_classification_cnum20_dist0_skew0_seed0_clip_local_missing \
    --model clip \
    --algorithm local_missing.food101_classification.clip \
    --sample uniform \
    --aggregate other \
    --num_rounds 300 \
    --proportion 0.1 \
    --num_epochs 3 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 32 \
    --test_batch_size 1 \
    --gpu 0 \
    --seed 1234 \
    --pm 0.0 \
    --ps 0.0 \
    # --wandb