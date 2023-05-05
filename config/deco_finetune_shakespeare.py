import time

out_dir = 'out-shakespeare-deco'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'deco-ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 5
gradient_accumulation_steps = 32
max_iters = 2000

# finetune at constant LR
#learning_rate = 3e-5
#decay_lr = False

# pruning settings
prune_interval = 10 # how often to prune the model
prune_initial_threshold = 1.0
prune_final_threshold = 0.05
prune_beta1 = 0.85
prune_beta2 = 0.95
prune_deltaT = 10
pruner_warmup_steps = 100
pruner_initial_warmup = 1
pruner_final_warmup = 5

# inspired on config/train_gpt2.py
max_iters = 50000
lr_decay_iters = 50000