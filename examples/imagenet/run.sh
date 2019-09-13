python -m torch.distributed.launch --nproc_per_node=4 main_amp.py --opt-level O0 /scratch/dataset/imagenet/ --deterministic
