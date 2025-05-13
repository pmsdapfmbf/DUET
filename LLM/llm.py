import torch_influence
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from influence import *
torch.set_warn_always(False)
from copy import deepcopy
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import lm_eval
from lm_eval.models.huggingface import HFLM

import datasets
import os
import sys

import transformers
from datasets import load_dataset, concatenate_datasets
import gc

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)

from transformers import LlamaForCausalLM, LlamaTokenizer

from LLM.tokenize_util import *

def get_tokenizer_and_model(model_id = "LLM/llama_8b_instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # adjust tokenizer (from alpaca repo, MIGHT NOT BE NEEDED IDK)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    model  = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto'
    )

    return tokenizer, model

batch_size = 8
num_epochs = 100
learning_rate= 3e-4
cutoff_len = 256
val_set_size = 2000
# lora hyperparams
lora_r = 8
lora_alpha = 16
lora_dropout= 0.05
lora_target_modules = [
    "q_proj",
    "v_proj",
]
# llm hyperparams
train_on_inputs = False  # if False, masks out inputs in loss
add_eos_token = False
group_by_length = False  # faster, but produces an odd training loss curve
gradient_accumulation_steps = 1
use_wandb = True

import numpy as np

from sklearn.utils.extmath import fast_logdet
from sklearn.metrics.pairwise import rbf_kernel

def get_log_det(kernel_values):
    return fast_logdet(kernel_values)

def get_kernel(list_of_embeddings):
    return rbf_kernel(list_of_embeddings, list_of_embeddings)

def get_next_best_datapoint(current_list_datapoint, current_log_det, dataset):
    
    best_idx = -1
    best_gain = -10000000000
    for idx, embedded_datapoint in enumerate(dataset): #datapoint is already an embedding
        new_list = current_list_datapoint + [embedded_datapoint]
        new_kernel = rbf_kernel(new_list, new_list)
        new_log_det = get_log_det(new_kernel)
        change_in_log_det = new_log_det - current_log_det
        
        if change_in_log_det > best_gain:
            best_idx = idx
            best_gain = change_in_log_det
    return best_idx, best_gain

def get_best_N_set(initial_dataset, N):
    current_chosen_dataset = []
    current_chosen_dataset.append(initial_dataset[0]) # take 1 datapoint at the start
    all_selected_idx = []
    all_selected_idx.append(0)
    for i in range(N):
        current_log_det = fast_logdet(rbf_kernel(current_chosen_dataset, current_chosen_dataset))
        best_idx, best_gain = get_next_best_datapoint(current_chosen_dataset, current_log_det, initial_dataset)
        current_chosen_dataset.append(initial_dataset[best_idx])
        all_selected_idx.append(best_idx)
        
    assert len(current_chosen_dataset) == N + 1
    return current_chosen_dataset, all_selected_idx

def sample(dataset, num_datapoints, additional_info, method, data_domain):
    if method == "remove_harmful" and additional_info == None:
        assert False, "bad combination!"
    print("method to use: ", method)
    # if random sample, just set all weight to 1
    if method == "random":
        additional_info = [1] * len(dataset)
    elif method == "IF_random":
        if sum(additional_info.tolist()) == 0:
            print("IF values are 0, go back to normal random sampling")
            additional_info = [1] * len(dataset)
    elif method == "IF_remove_harmful":
        if sum(additional_info.tolist()) == 0:
            print("IF values are 0, go back to normal random sampling")
            additional_info = [1] * len(dataset)
            method = "random"
    elif method == "log_det":
        pass
    else:
        assert False, "unknown method of sampling"

    if method == "log_det" and data_domain == None:
        method = "random"
    
    if method == "IF_remove_harmful":
        print("method is remove harmful, we will remove bottom 10% IF value datapoints")
        normalized_influences = deepcopy(additional_info)
        # remove lowest 10% data
        normalized_influences += abs(min(normalized_influences)) # make everything more than 0
        percentile_value = torch.quantile(additional_info, 0.2).item()
        # Set values below the 10th percentile to zero
        normalized_influences = normalized_influences.numpy()
        normalized_influences[normalized_influences < percentile_value] = 0
        
        num_harmful_points = sum(i == 0 for i in normalized_influences)
        
        # random sample from the rest of useful data uniformly
        normalized_influences[normalized_influences!=0] = 1/(len(normalized_influences) - num_harmful_points)
        indices = np.random.choice(len(normalized_influences), size=num_datapoints, p=normalized_influences) # sample from new list
    elif method == "log_det":
        # read the embedding from 
        embeddings = np.load("LLM/domain_training_embeddings/"+data_domain+".npy") # size N x embed_dim
        embeddings = np.squeeze(embeddings)
        _, indices = get_best_N_set(embeddings, num_datapoints)
    else:
        print("method is to randomly sample")
        # influence sample; maybe use other methods
        normalized_influences = additional_info # normalized
        normalized_influences = np.asarray(normalized_influences).astype('float64')
        min_inf = abs(min(normalized_influences))
        normalized_influences += min_inf # normalized
        sum_inf = sum(normalized_influences)
        normalized_influences /= sum_inf # normalized
        normalized_influences[0] += (1 - sum(normalized_influences)) # bypass any smallprecision errors
        indices = np.random.choice(len(normalized_influences), size=num_datapoints, p=normalized_influences)
        # Use the `select` method to extract those samples
    sampled_dataset = dataset.select(indices)
    return sampled_dataset

def extract_data_mixture_and_train(model, random_dir, tokenizer, train_datasets, val_datasets, data_domains, mixing_ratio, additional_info, total_number_datapoints, run_name, method="random", train_epochs=1, batch_size=8, max_step=-1, eval_steps=100, lora_config=None, callback=[]):
    torch.manual_seed(42)
#     '''
#     model: llama base model
#     tokenizer: llama tokenizer
#     train_datasets: List of datasets
#     data_domains: List of data domain names, should be same size as train_datasets
#     mixing_ratio: List of mixing ratio, should sum to 1, list size same as train_datasets
#     additional_information: List of List of IF values for each dataset, for us to do sampling
    output_dir = "LLM/BO/"+random_dir+"/"+run_name # store this model here every BO runs, to be evaluated
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # LORA config
    # if lora_config == None:
    #     config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=lora_target_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    # else:
    config = lora_config
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # apply tokenization to all data
    print("tokenizing all data into correct format...")
    tokenizing_method = {
        "wikitext":generate_and_tokenize_prompt_wikiQA,
        "triviaqa":generate_and_tokenize_prompt_trivialQA,
        "pubmedqa":generate_and_tokenize_prompt_pubmedQA,
        "truthfulqa_gen":generate_and_tokenize_prompt_truthfulQA,
        "commonsense_qa":generate_and_tokenize_prompt_commonsenseQA,
        "hellaswag":generate_and_tokenize_prompt_hellaswag,
        "sciq":generate_and_tokenize_prompt_sciq,
        "gsm8k":generate_and_tokenize_prompt_gsm8k,
        "squadv2":generate_and_tokenize_prompt_squad,
        "headqa_en":generate_and_tokenize_prompt_headqa
    }
    
    # sample the correct amount of data from each domain
    print("iterating through each data domain and sampling the sufficient datapoints")
    print("mixing ratio: ", mixing_ratio)
    print("ALL DATA DOMAINS: ", data_domains)
    all_sampled_train_data = []
    all_sampled_val_data = []
    for train_dataset, val_dataset, data_domain, ratio, IF_values in zip(train_datasets, val_datasets, data_domains, mixing_ratio, additional_info):
        
        print("doing sampling for domain: ", data_domain)
        print("ratio: ", ratio)
        total_datapt = int(total_number_datapoints * ratio)
        print("number of datapoints needed (ratio * total): ", total_datapt)
        if total_datapt == 0:
            continue # skip if no data needed
        if ratio == 1.0:
            print("ratio is 1.0, don't have to sample")
            sampled_train_data = train_dataset
            sampled_val_data = val_dataset
        else:
            print("sampling...")
            sampled_train_data = sample(train_dataset, total_datapt, additional_info=IF_values, method=method, data_domain=data_domain)
            print("done sampling training")
            sampled_val_data = sample(train_dataset, total_datapt, additional_info=None, method="random", data_domain=None)
            print("done sampling validation")
        
        sampled_train_data = sampled_train_data.shuffle().map(tokenizing_method[data_domain], fn_kwargs={"tokenizer": tokenizer,
                                                                                   "add_eos_token": add_eos_token,
                                                                                   "train_on_inputs": train_on_inputs,
                                                                                   })
        sampled_val_data = sampled_val_data.shuffle().map(tokenizing_method[data_domain], fn_kwargs={"tokenizer": tokenizer,
                                                                                   "add_eos_token": add_eos_token,
                                                                                   "train_on_inputs": train_on_inputs,
                                                                                   })
        print("done mapping!")
        
        # drop columns
        
        sampled_train_data = sampled_train_data.select_columns(['input_ids', 'attention_mask', 'labels'])
        sampled_val_data = sampled_val_data.select_columns(['input_ids', 'attention_mask', 'labels'])
        
        all_sampled_train_data.append(sampled_train_data)
        all_sampled_val_data.append(sampled_val_data)
    
    print(all_sampled_train_data)
    combined_train_dataset = concatenate_datasets(all_sampled_train_data)
    combined_val_dataset = concatenate_datasets(all_sampled_val_data)
    print("first datapoint of training data: ", combined_train_dataset[0])
    print("length of training data: ", len(combined_train_dataset))
    output_model_dir = train(model, tokenizer, combined_train_dataset, combined_val_dataset, output_dir, run_name, train_epochs, batch_size, max_step, eval_steps, lora_config=config, callback=callback)
    return output_model_dir
    
def train(model, tokenizer, train_dataset, val_dataset, output_dir, run_name, train_epochs=1, batch_size=8, max_step=-1, eval_steps=100, lora_config=None, callback=[]):
    # if torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = False
    model.model_parallel = False
    model.train()
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=10,
            num_train_epochs=train_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=20,
            optim="adamw_torch",
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_total_limit=1,
            save_steps=eval_steps,
            max_steps=max_step,
            output_dir=output_dir,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=True,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=run_name,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callback
    )

    print("training model...")
    model.print_trainable_parameters()
    trainer.train()

    print("saving final model LoRA weights at: ", output_dir + "/" + "final_model_after_training")
    model.save_pretrained(output_dir + "/" + "final_model_after_training")
    model.to("cpu")
    del trainer
    with torch.no_grad():
        torch.cuda.empty_cache()
    del model
    gc.collect()
    return output_dir + "/" + "final_model_after_training"
    
def evaluate_tasks(tasks : List[str], model, tokenizer, batch=1, few_shot=0):

    print("creating HFLM wrapper for model_path")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, dtype=torch.bfloat16, max_length=tokenizer.model_max_length,
                batch_size=batch, trust_remote_code=True)

    print("evaluating on tasks: ", tasks)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        task_manager=lm_eval.tasks.TaskManager(),batch_size=batch,max_batch_size=batch, num_fewshot=few_shot)
    return results

def load_data(data_domain):
        # Load the dataset
    print(data_domain)
    if data_domain == "headqa_en":
        data_domain = "headqa"
    if data_domain == "wikitext":
        dataset = datasets.load_dataset(data_domain, "wikitext-2-v1", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "triviaqa":
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "pubmedqa":
        dataset = datasets.load_dataset("bigbio/pubmed_qa", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "truthfulqa_gen":
        dataset = datasets.load_dataset("truthfulqa/truthful_qa", "generation", cache_dir = "./datasets")
        train_dataset = dataset["validation"]
        val_dataset = dataset["validation"]
    elif data_domain == "commonsense_qa":
        dataset = datasets.load_dataset("tau/commonsense_qa", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "hellaswag":
        dataset = datasets.load_dataset("DatologyAI/hellaswag", cache_dir = "./datasets", trust_remote_code=True)
        train_dataset = dataset["eval"]
        val_dataset = dataset["eval"]
    elif data_domain == "sciq":
        dataset = datasets.load_dataset("allenai/sciq", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "gsm8k":
        dataset = datasets.load_dataset("openai/gsm8k", "main", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    elif data_domain == "squadv2":
        dataset = datasets.load_dataset("rajpurkar/squad_v2" , cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "headqa":
        dataset = datasets.load_dataset("dvilares/head_qa", "en", cache_dir = "./datasets", trust_remote_code=True)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "hellaswag":
        dataset = datasets.load_dataset("DatologyAI/hellaswag", cache_dir = "./datasets", trust_remote_code=True)
        train_dataset = dataset["eval"]
        val_dataset = dataset["eval"]
    else:
        assert False, "data_domain not valid, pls check"
    return train_dataset, val_dataset