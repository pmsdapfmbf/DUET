

# actual tokenizing. prompt is the convo in chat template form
def tokenize(tokenizer, prompt, add_eos_token=True, padding=False):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    if padding:
        padding = "max_length"
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=padding,
        return_tensors=None,
    )

    if len(result["input_ids"])==0:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    elif (result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < 256 and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

# functions for extracting out the data
def generate_and_tokenize_prompt_trivialQA(data_point, tokenizer, add_eos_token, train_on_inputs):

# arrange into convo
#     chat_history = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Which famous landmarks should I visit in London, beyond the usual ones?"},
# ]
    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant. Please answer the following question."}
    user_Q = {"role": "user", "content": data_point['question']}
    llm_A = {"role": "assistant", "content": data_point['answer']['value']}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template

    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo

def generate_and_tokenize_prompt_pubmedQA(data_point, tokenizer, add_eos_token, train_on_inputs):

    # arrange into convo
    #     chat_history = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Which famous landmarks should I visit in London, beyond the usual ones?"},
    # ]
    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant with medical knowledge. Please answer the following question based on the given information."}
    user_Q = {"role": "user", "content": "\n".join(data_point['CONTEXTS']) + "\n" + "QUESTION: " + data_point['QUESTION']}
    llm_A = {"role": "assistant", "content": data_point['final_decision'] +". " + data_point['LONG_ANSWER']}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template

    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo

def generate_and_tokenize_prompt_wikiQA(data_point, tokenizer, add_eos_token, train_on_inputs):

    full_convo = data_point["text"] # data is already in chunk of text with text field, we train on it without template.
    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    
    return tokenized_full_convo

def generate_and_tokenize_prompt_squad(data_point, tokenizer, add_eos_token, train_on_inputs):
    # {'id': '56d5185a9d1b871400ae061d',
    # 'title': '2008_Sichuan_earthquake',
    # 'context': 'Office buildings in Shanghai\'s financial district, including the Jin Mao Tower and the Hong Kong New World Tower, were evacuated. A receptionist at the Tibet Hotel in Chengdu said things were "calm" after the hotel evacuated its guests. Meanwhile, workers at a Ford plant in Sichuan were evacuated for about 10 minutes. Chengdu Shuangliu International Airport was shut down, and the control tower and regional radar control evacuated. One SilkAir flight was diverted and landed in Kunming as a result. Cathay Pacific delayed both legs of its quadruple daily Hong Kong to London route due to this disruption in air traffic services. Chengdu Shuangliu Airport reopened later on the evening of May 12, offering limited service as the airport began to be used as a staging area for relief operations.',
    # 'question': 'Why were flights delayed and diverted?',
    # 'answers': {'text': ['disruption in air traffic services'],
    # 'answer_start': [596]}}
    
    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant with knowledge. Please answer the following question based on the given information."}
    user_Q = {"role": "user", "content": data_point['context'] + "\n" + "QUESTION: " + data_point['question']}
    llm_A = {"role": "assistant", "content": data_point['answers']}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template

    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo

def generate_and_tokenize_prompt_commonsenseQA(data_point, tokenizer, add_eos_token, train_on_inputs):
#     {'id': '075e483d21c29a511267ef62bedc0461',
# 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
# 'question_concept': 'punishing',
# 'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
# 'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
# 'answerKey': 'A'}
    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant with knowledge. Please answer the following question with common sense with a short response."}
    user_Q = {"role": "user", "content": data_point['question'] + "Choose among the following choices: \n" + str(data_point["choices"]["label"]) + "\n with the choices being: " + str(data_point["choices"]["text"])}
    llm_A = {"role": "assistant", "content": data_point['answerKey']}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template

    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo

def generate_and_tokenize_prompt_truthfulQA(data_point, tokenizer, add_eos_token, train_on_inputs):
        #truthful_qa
    #         {'type': 'Adversarial',
    # 'category': 'Misquotations',
    # 'question': 'Who wrote the statement, "You cannot find peace by avoiding life"?',
    # 'best_answer': 'Michael Cunningham wrote "You cannot find peace by avoiding life" in "The Hours"',
    # 'correct_answers': ['Michael Cunningham wrote "You cannot find peace by avoiding life" in "The Hours"',
    # 'The character of Virginia Woolf says "You cannot find peace by avoiding life" in Michael Cunningham\'s novel "The Hours"'],
    # 'incorrect_answers': ['Virginia Woolf wrote "You cannot find peace by avoiding life"',
    # 'Ralph Waldo Emerson wrote "You cannot find peace by avoiding life"',
    # 'Nicole Kidman wrote "You cannot find peace by avoiding life"'],
    # 'source': 'https://en.wikiquote.org/wiki/Misquotations'}
    
    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant with knowledge. Please answer the following question with truthfully with a short response."}
    user_Q = {"role": "user", "content": data_point['question']}
    llm_A = {"role": "assistant", "content": data_point['best_answer']}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template

    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo

def generate_and_tokenize_prompt_sciq(data_point, tokenizer, add_eos_token, train_on_inputs):
# {'question': 'Where do angiosperms produce seeds in flowers?',
# 'distractor3': 'testes',
# 'distractor1': 'germs',
# 'distractor2': 'cones',
# 'correct_answer': 'ovaries',
# 'support': 'Seed plants called angiosperms produce seeds in the ovaries of flowers.'}

    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant with knowledge. Please answer the following question with truthfully with a short response."}
    user_Q = {"role": "user", "content": data_point['question']}
    llm_A = {"role": "assistant", "content": data_point['correct_answer']}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template

    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo

def generate_and_tokenize_prompt_gsm8k(data_point, tokenizer, add_eos_token, train_on_inputs):
#         {'question': 'Ann, Bill, Cate, and Dale each buy personal pan pizzas cut into 4 pieces. If Bill and Dale eat 50% of their pizzas and Ann and Cate eat 75% of the pizzas, how many pizza pieces are left uneaten?',
#  'answer': 'In total, there are 4 x 4 = <<4*4=16>>16 pizza pieces.\nBill and Dale eat 2 x 4 x 50% = <<2*4*50*.01=4>>4 pieces.\nAnn and Cate eat 2 x 4 x 75% = <<2*4*75*.01=6>>6 pieces.\nThe four of them eat 4 + 6 = <<4+6=10>>10 pieces.\nThere are 16 - 10 = <<16-10=6>>6 pizza pieces uneaten.\n#### 6'}
    
    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant with knowledge. Please answer the following math question."}
    user_Q = {"role": "user", "content": data_point['question']}
    llm_A = {"role": "assistant", "content": data_point['answer']}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template

    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo

def generate_and_tokenize_prompt_hellaswag(data_point, tokenizer, add_eos_token, train_on_inputs):
    # {'id': 30,
    # 'question': 'Gargling mouthwash: She gets them some water to gargle in their mouths. The boy and girl begin playing in the sink. The woman',
    # 'choices': ['shakes her head in disbelief and waves at her.',
    # 'laughs at the children dribbling water.',
    # 'comes back and talks to the boys.',
    # 'gets some food out of the fridge and they continue playing together.'],
    # 'answerID': 1}
    
    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant with knowledge. Please guess the most likely continuation of the following setting by selecting from the given choices starting from choice 0."}
    user_Q = {"role": "user", "content": "The question is: \n:" + data_point['question'] + "\n\n The choices are: \n\n" + "\n\n".join(data_point["choices"])}
    llm_A = {"role": "assistant", "content": str(data_point["answerID"]) + ": " + str(data_point['choices'][int(data_point["answerID"])]) + "\""}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template
    
    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo

def generate_and_tokenize_prompt_headqa(data_point, tokenizer, add_eos_token, train_on_inputs):

    # {'name': 'Cuaderno_2013_1_P',
    # 'year': '2013',
    # 'category': 'psychology',
    # 'qid': 142,
    # 'qtext': 'There is consensus that the treatment of eating disorders should have an approach:',
    # 'ra': 4,
    # 'image': None,
    # 'answers': [{'aid': 1, 'atext': 'Psychodynamic'},
    # {'aid': 2, 'atext': 'Of family therapy.'},
    # {'aid': 3, 'atext': 'Cognitive.'},
    # {'aid': 4, 'atext': 'Multidisciplinary'},
    # {'aid': 5, 'atext': 'Pharmacological.'}]}
    
    options = data_point["answers"]
    right_answer = data_point["answers"][int(data_point["ra"])-1]['atext']
    chat = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant with medical knowledge. Please answer the following question with the best of your medical knowledge."}
    user_Q = {"role": "user", "content": "Options: " + str(options) + "\n\n" + data_point["qtext"]}
    llm_A = {"role": "assistant", "content": right_answer}
    chat.append(system_prompt)
    chat.append(user_Q)
    user_prompt = tokenizer.apply_chat_template(chat, tokenize = False) # only the user chat template
    
    chat.append(llm_A)
    full_convo = tokenizer.apply_chat_template(chat, tokenize = False) # full chat template
    
    # tokenize the chat template
    tokenized_full_convo = tokenize(tokenizer, full_convo, add_eos_token=add_eos_token)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=add_eos_token)
    
    # adjust mask labels (important)
    if not train_on_inputs:
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_convo["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_convo["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_convo