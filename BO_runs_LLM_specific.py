import json
from BO import run_BO_for_LLM

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--contaminate", help="to contaminate training data?", type=int, default=0)
parser.add_argument("--iterations", help="iterations BO?", type=int, default=10)
parser.add_argument("--num_data", help="total_data?", type=int, default=10000)
parser.add_argument("--epochs", help="epochs", default=1)
parser.add_argument("--trials", help="trials", default=1)
parser.add_argument("--evaluation_cuda", help="evaluation_cuda", default=0)
parser.add_argument("--sample_method", help="sample_method", default="random") # random, IF_random, IF_remove_harmful
parser.add_argument("--eval_tasks", help="eval_tasks")
parser.add_argument("--experiments_setting", help="either ood or in_dist")
parser.add_argument("--output_dir", help="output_dir")

args = vars(parser.parse_args())
print("command-line args: ", args)

# WIP
to_contaminate= bool(args["contaminate"])
if not to_contaminate:
    influence_path="influence/"
    print("getting influence from: ", influence_path)
else:
    influence_path="influence/contaminated/"
    print("getting influence from: ", influence_path)

setting=(args["experiments_setting"])
output_dir=(args["output_dir"])

epochs=int(args["epochs"])
trials=int(args["trials"])
cuda=int(args["evaluation_cuda"])
cuda="cuda:"+str(cuda)
BO_run = int(args["iterations"])
total_data = int(args["num_data"])
sample_methods = str(args["sample_method"]).split(",")
tasks = str(args["eval_tasks"]).split(",") # list of eval task
evaluation_weights = [1/len(tasks)] * len(tasks)
import random
import string
seed = random.randint(0,1000)
# Generate a random string of size 5 using uppercase and lowercase letters
random_string = ''.join(random.choices(string.ascii_letters, k=5))
print("random sentence created:", random_string)

task_metrics = {
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

data_domains_initial = list(task_metrics.keys())
print("current eval task: ", tasks)
if setting == "ood":
    data_domains =  [x for x in data_domains_initial if x not in tasks] # remove training domain that is in task
else:
    data_domains = [x for x in data_domains_initial]

'''
{'commonsense_qa': {'alias': 'commonsense_qa', 'acc,none': 0.7624897624897625, 'acc_stderr,none': 0.012183673723473462}, 'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.025018953752843062, 'exact_match_stderr,strict-match': 0.004302045046564279, 'exact_match,flexible-extract': 0.332827899924185, 'exact_match_stderr,flexible-extract': 0.012979892496598268}, 'headqa_en': {'alias': 'headqa_en', 'acc,none': 0.4310722100656455, 'acc_stderr,none': 0.00945908371802805, 'acc_norm,none': 0.4777534646243618, 'acc_norm_stderr,none': 0.009540808712468811}, 'headqa_es': {'alias': 'headqa_es', 'acc,none': 0.3636032093362509, 'acc_stderr,none': 0.00918804950673874, 'acc_norm,none': 0.40809628008752735, 'acc_norm_stderr,none': 0.009387551551893495}, 'hellaswag': {'alias': 'hellaswag', 'acc,none': 0.5769766978689504, 'acc_stderr,none': 0.004930293787545628, 'acc_norm,none': 0.758514240191197, 'acc_norm_stderr,none': 0.004271094187098002}, 'pubmedqa': {'alias': 'pubmedqa', 'acc,none': 0.746, 'acc_stderr,none': 0.019486596801643427}, 'sciq': {'alias': 'sciq', 'acc,none': 0.963, 'acc_stderr,none': 0.005972157622389641, 'acc_norm,none': 0.933, 'acc_norm_stderr,none': 0.007910345983177549}, 'triviaqa': {'alias': 'triviaqa', 'exact_match,remove_whitespace': 0.509808292465448, 'exact_match_stderr,remove_whitespace': 0.0037319764897777853}, 'truthfulqa_gen': {'alias': 'truthfulqa_gen', 'bleu_max,none': 20.059961039741832, 'bleu_max_stderr,none': 0.7225497258741972, 'bleu_acc,none': 0.46511627906976744, 'bleu_acc_stderr,none': 0.017460849975873962, 'bleu_diff,none': -0.5175936691929918, 'bleu_diff_stderr,none': 0.6303941458560193, 'rouge1_max,none': 43.30567524349279, 'rouge1_max_stderr,none': 0.8669771974211276, 'rouge1_acc,none': 0.4969400244798042, 'rouge1_acc_stderr,none': 0.017503173260960635, 'rouge1_diff,none': -0.4731320522373688, 'rouge1_diff_stderr,none': 0.8553571011524773, 'rouge2_max,none': 27.14173439603888, 'rouge2_max_stderr,none': 0.95047247986369, 'rouge2_acc,none': 0.3598531211750306, 'rouge2_acc_stderr,none': 0.01680186046667716, 'rouge2_diff,none': -1.9186008288656529, 'rouge2_diff_stderr,none': 0.9106913132355565, 'rougeL_max,none': 40.423027758258726, 'rougeL_max_stderr,none': 0.8648234890867836, 'rougeL_acc,none': 0.4810281517747858, 'rougeL_acc_stderr,none': 0.017490896405762364, 'rougeL_diff,none': -0.9313105921815803, 'rougeL_diff_stderr,none': 0.8623845067257}, 'wikitext': {'alias': 'wikitext', 'word_perplexity,none': 16.536551378163434, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6898777792350577, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.7569189070590712, 'bits_per_byte_stderr,none': 'N/A'}}
'''

evaluation_task = {}
for task, weight in zip(tasks, evaluation_weights):
    evaluation_task[task] = (float(weight), task_metrics[task])

print("evaluation tasks and weights: ", evaluation_task)

train_epochs = 1
training_batch = 8
evaluation_batch = 4
evaluation_steps=50000
    
final_info_stored = {"command line args": args,
                    "hash string": random_string,
                    "training domain": data_domains,
                    "evaluation domain": tasks,
                    "weight": evaluation_weights} # weight in str

for sample_method in sample_methods:
    results = []
    for x in range(trials):
        model_id="Qwen/Qwen2.5-7B-Instruct" # pass this into next function if necessary
        GP_input, observed_output, gp = run_BO_for_LLM(data_domains = data_domains,
                                                    random_dir = random_string, 
                                                        BO_run = BO_run,
                                                        total_data = total_data, 
                                                        evaluation_cuda = cuda, 
                                                        evaluation_task = evaluation_task,
                                                        sampling_method = sample_method, 
                                                        train_epochs=train_epochs, 
                                                        training_batch=training_batch, 
                                                        evaluation_batch=evaluation_batch,
                                                        eval_steps=evaluation_steps,
                                                        printout=True)
        # except Exception as e:
        #     print("exception encountered at the following evaluation and training domain: ")
        #     print("training: ", data_domains)
        #     print("evaluation: ", tasks)
        #     print("exception: ", e)
        #     print("training batch: ", training_batch)
        #     print("evaluation batch: ", evaluation_batch)

        current_max = float('-inf')  # Start with negative infinity
        max_until_now = []           # List to store max values at each step

        # Iterate through the list
        for num in observed_output:
            current_max = max(current_max, num)  # Update the current maximum
            max_until_now.append(current_max)    # Store the max up to this step

        # Output the result
        print("Best at every step:", max_until_now)
        results.append(max_until_now)
    final_info_stored[sample_method] = results
print("final results: ", final_info_stored)
# store results
try:
    with open("LLM/BO/" + output_dir + "/" + "_".join(tasks)+".json", 'w') as f:
        json.dump(final_info_stored, f)
except:
    print("error with storing json, it's ok. moving to next...")





