from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
import torch

from helper import get_data_from_mixing_ratio
from image_training import train
from typing import List
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)

lora_alpha = 16
lora_dropout= 0.05
lora_r=16
lora_target_modules = [
    "q_proj",
    "v_proj",
]
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
    
def iterative_loop(data_sources : List[DataLoader], validation_data : DataLoader, method : str, additional_info : List[List[float]], seed, layers_freeze : int, cuda : str, num_epochs=10, iterations=10, data="images", printout=True):
    
    input_X = torch.Tensor((len(data_sources))*[float(1/len(data_sources))]) # initial X
    GP_input = []
    observed_output = []
    for i in range(iterations):
        print("iteration: ", i)
        
        if printout:
            print("mixing data with method: ", method)

        mixed_data = get_data_from_mixing_ratio(data_sources, mixing_ratio=input_X,
                                                additional_info=additional_info,
                                                method=method,
                                                seed=seed,
                                                base_number_of_batches=20) # each agent do some influence function process to get data
        
        if data=="images":
            acc_all, observed_performance, _ = train(mixed_data, validation_data, seed=seed, lr=5e-5, cuda=cuda, num_epochs=num_epochs, num_layer_to_unfreeze=layers_freeze, printout=printout) # observe the performance of this dataset from finetuning
        if printout:
            print("performance after training: ", observed_performance)
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X)
        #current_gp_input.append(current_mixing_parameter)
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=1)
        bounds = torch.stack([torch.zeros(len(current_gp_input)), torch.ones(len(current_gp_input))]) # need to change the bounds for parameters
        A = [1.0] * len(data_sources)
        x = list(range(len(data_sources)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=50,
            #equality_constraints = [(torch.tensor(list(range(len(data_sources)))), torch.tensor([1.0] * len(data_sources)), 1)]
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        input_X = [x if x >= 0.05 else 0 for x in candidate[0]]
        if printout:
            print("proposed parameters for next round by BO:", input_X)
    return GP_input, observed_output, gp

def get_BO_plots(observations):
    BO_to_plot = []
    for x in range(0,len(observations)):
        BO_to_plot.append((max(observations[:x+1])))
    return BO_to_plot

def run_BO(all_loaders, validaton_dataloader, method, additional_info, seed, iterations, num_epochs, cuda, layers_freeze, printout=False):
    print("running BO...")
    X, observations, gp = iterative_loop(all_loaders, validaton_dataloader, cuda=cuda, method=method, additional_info=additional_info, layers_freeze=layers_freeze, seed=seed, num_epochs=num_epochs, iterations=iterations, printout=printout)
    BO_to_plot = get_BO_plots(observations) # BO results
    naive_combine = BO_to_plot[0] # naive mixing result is the first iteration result of BO

    # plot model performance as BO progresses...
    # plt.plot(range(len(BO_to_plot)), BO_to_plot, c="blue", alpha=0.3, label="BO on mixing ratio")
    # plt.axhline(naive_combine, linestyle="--", c="red", label="sample from each data source equally")
    # plt.xlabel("BO iterations")
    # plt.ylabel("accuracy on evaluation task")
    # plt.title("method: "+method)
    # plt.legend()
    # plt.savefig(method+".png")

    # plot posterior
    # posterior_acc = []
    # for x in np.linspace(0,1,100):
    #     posterior_acc.append(gp.posterior(torch.Tensor([[x,1-x]])).mean.item())
        
    # plt.plot(np.linspace(0,1,100), posterior_acc)
    # plt.xlabel("mixing ratio (percentage on cats and dogs)")
    # plt.ylabel("accuracy")
    # plt.title("evaluation ratio : 1.0 cats and dogs")
    # plt.show()

    def get_optimal_mixture_from_GP_posterior():
        UCB = UpperConfidenceBound(gp, beta=0.0)
        bounds = torch.stack([torch.zeros(len(all_loaders)), torch.ones(len(all_loaders))]) # need to change the bounds for parameters
        A = [1.0] * len(all_loaders)
        x = list(range(len(all_loaders)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=30,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        return candidate

    def get_best_observation_mixture():
        
        # Find the index in list B that has the highest value
        highest_index = observations.index(max(observations))
        
        # Return the corresponding item in list A
        return X[highest_index]

    
    print("best mixture found in BO iterations is: ", get_best_observation_mixture())
    return BO_to_plot

from LLM.llm import extract_data_mixture_and_train, evaluate_tasks, load_data, get_tokenizer_and_model
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from tqdm import tqdm

def run_BO_for_LLM(data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100):
    
    model_id = "LLM/llama_8b_instruct"
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    #model = prepare_model_for_kbit_training(model)

    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        
        if printout:
            print("mixing data with method: ", sampling_method)

        # sample from each domain and train a model
        path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                        train_datasets=train_datasets, 
                                                        val_datasets=val_datasets, 
                                                        data_domains=data_domains, 
                                                        mixing_ratio=input_X, 
                                                        additional_info=all_influences, # add IF value
                                                        total_number_datapoints=total_data, 
                                                        run_name="BO_run_" +str(i),
                                                        method=sampling_method,
                                                        train_epochs=train_epochs, 
                                                        batch_size=training_batch,
                                                        max_step=max_steps,
                                                        lora_config=lora_config,
                                                        eval_steps=eval_steps)
        # free gpu memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        print("evaluating...")
        lora_path = path_to_final_model #final_model_after_training
        config = PeftConfig.from_pretrained(lora_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
        lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
        
        observed_performance = 0
        tasks = list(evaluation_task.keys())
        results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch)
        print("results: ", results["results"])
        for task in evaluation_task:
            task_weight, metric = evaluation_task[task]
            print(task_weight)
            print(metric)
            print(results["results"][task][metric])
            perf = results["results"][task][metric]
            if task == "wikitext":
                perf = - perf # we want to maximize the score, so for perplexity we maximize instead
            observed_performance += (perf * task_weight)
        print("current iteration performance: ", observed_performance)
        lora_model.to("cpu")
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X)
        
        #current_gp_input.append(current_mixing_parameter)
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=1)
        bounds = torch.stack([torch.zeros(len(current_gp_input)), torch.ones(len(current_gp_input))]) # need to change the bounds for parameters
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=50,
            #equality_constraints = [(torch.tensor(list(range(len(data_sources)))), torch.tensor([1.0] * len(data_sources)), 1)]
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        input_X = [x if x >= 0.05 else 0 for x in candidate[0]]
        if printout:
            print("proposed parameters for next round by BO:", input_X)
    return GP_input, observed_output, gp
