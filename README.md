# DUET
This repository contains DUET, a data-mixing method that exploits feedback from an unseen task.

## Requirements

```
pip3 install -r requirements.txt
```

## Running DUET
Running DUET is straightforward. Simply run the following command (ensure you have at least one free GPU):
```
CUDA_VISIBLE_DEVICES=0 python3 -u BO_runs_LLM_specific.py --contaminate=0 --iterations=10 --num_data=5000 --epochs=1 --trials=10 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=results
```
The python script automatically fetches data from 9 training domains with the following evaluation performance, where the performance is used as feedback elicited in DUET's problem setting:
```task_metrics = {
  "commonsense_qa": "acc,none",
  "gsm8k": "exact_match,strict-match",
  "headqa_en": "acc,none",
  "hellaswag": "acc,none",
  "pubmedqa": "acc,none",
  "sciq": "acc_norm,none",
  "triviaqa": "exact_match,remove_whitespace",
  "truthfulqa_gen": "bleu_acc,none",
  "wikitext": "word_perplexity,none",
}
```

There are a few important arguments.
- The first important argument here is `--eval_tasks=gsm8k`, which specifies the unseen evaluation task. You can also specify something like `--eval_tasks=gsm8k,headqa_en`, which means both tasks will be set as the evaluation task (the performance is averaged). You can specify as many domains as you want.
- The second important argument is `--experiments_setting=ood`, which implies we are removing the eval_task(s) from our training domains. Alternatively, you can use `--experiments_setting=in_dist` to keep the eval_task in the training domain (this makes the training data mixture easier to optimize, since the eval data is found in the training data).
