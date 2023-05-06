#!/bin/bash

cp task_config.gpt task_config.py

python3 ga.py \
    --mlm_model "roberta-large" \
    --eval_model "gpt2-large" \
    --petri_size 64 \
    --petri_iter_num 50 \
    --max_seq_length 512 \
    --mutate_prob 0.75 \
    --crossover_prob 0.5 \
    --seed $1 \
    --k_shot 16 \
    --task_name $2 \
    --batch_size 32
