export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=0
export VLLM_VERSION=0.9.1

# examples/run_async_ppo.sh
python3 training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=2 \
    allocation_mode=vllm.d1p1m1+d1p1m1 \
    cluster.fileroot=/sfs_turbo/lzs/save \
    actor.type._class=qwen3 \
    actor.path=/sfs_turbo/models/qwen3_1_7b \
    ref.type._class=qwen3 \
    ref.path=/sfs_turbo/models/qwen3_1_7b \
    dataset.path=/sfs_turbo/lzs/datasets/AReaL-boba-Data/AReaL-boba-106k.jsonl \
    dataset.train_bs_n_seqs=4 \
    group_size=8 \
    ppo.gen.max_new_tokens=4096 \
    ppo.ppo_n_minibatches=1 \
    actor_train.mb_spec.max_tokens_per_mb=32768 \
    actor_inf.mb_spec.max_tokens_per_mb=32768 \
    max_concurrent_rollouts=1 \
    max_head_offpolicyness=2