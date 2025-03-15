import os
import json

num_seeds = 5
dataset = 'Bongard-HOI'

for seed in range(1, 1+num_seeds):
    os.path.join('ma_splits', f'{dataset}_train_seed{seed}')