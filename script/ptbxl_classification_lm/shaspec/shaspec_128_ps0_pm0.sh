python generate_fedtask.py \
    --benchmark ptbxl_classification_lm \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing \


python main.py \
    --task ptbxl_classification_lm_cnum20_dist0_skew0_seed0_missing_mifl_local_missing \
    --model shaspec \
    --algorithm multimodal.ptbxl_classification_lm.shaspec \
    --sample full \
    --aggregate other \
    --num_rounds 300 \
    --proportion 1.0 \
    --num_epochs 3 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 0 \
    --seed 1234 \
    --local_cls_lambda 0.1 \
    --local_aln_lambda 0.1 \
    --global_aln_lambda 0.1 \
    --ps 0.0 \
    --pm 0.0 \
    # --wandb