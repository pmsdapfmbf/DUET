# DUET
This repository contains DUET, a data-mixing method that exploits feedback from an unseen task.
```
CUDA_VISIBLE_DEVICES=1 python3 -u BO_runs_LLM_specific.py --contaminate=0 --iterations=10 --num_data=5000 --epochs=1 --trials=10 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=in_dist --output_dir=results
```
