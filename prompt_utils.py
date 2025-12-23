import random
from collections import defaultdict

ID_PROMPT_TEMPLATES = [
    # A. Direct I/O phrasing
    "input: {x}\noutput: {y}",
    "input -> {x}\noutput -> {y}",
    "Input:\n{x}\nOutput:\n{y}",
    "given input: {x}\nreturn output: {y}",
    "input = {x}\noutput = {y}",

    # D. Formal / technical tone
    "input sequence:\n{x}\noutput sequence:\n{y}",
    "given:\n{x}\nexpected:\n{y}",
    "input data:\n{x}\noutput data:\n{y}",
    "source:\n{x}\ntarget:\n{y}",
    "x:\n{x}\ny:\n{y}",
]

OOD_PROMPT_TEMPLATES = [
    # B. Question-like (weak semantics)
    "what is the answer of {x}?\n{y}",
    "answer for the following:\n{x}\n{y}",
    "the answer to this input is:\n{x}\n{y}",
    "solve the following:\n{x}\n{y}",
    "response to:\n{x}\n{y}",

    # C. Minimal instruction verbs
    "process:\n{x}\nresult:\n{y}",
    "convert:\n{x}\n{y}",
    "map the following input to output:\n{x}\n{y}",
    "produce output for:\n{x}\n{y}",
    "complete:\n{x}\n{y}",
]

def shuffle_label_func(demon_pool):
    all_outputs = [item["output"] for item in demon_pool]
    random.shuffle(all_outputs)
    for item, new_output in zip(demon_pool, all_outputs):
        item["output"]=new_output
    return demon_pool

def random_selection(demon_pool, num_shots=0, shuffle_label=False):
    selection=random.sample(demon_pool, k=num_shots)
    if shuffle_label:
        selection = shuffle_label_func(selection)
    return selection

def stratify_selection(demon_pool, num_shots=0, label_list=None, shuffle_label=False):
    buckets = defaultdict(list)
    n_shots_by_class = num_shots//len(label_list)
    for item in demon_pool:
        if item["output"] in label_list:
            buckets[item["output"]].append(item)
    selection = []
    for l in label_list:
        if len(buckets[l])<n_shots_by_class:
            raise ValueError(f"Label {l} has only {len(buckets[l])} samples, but {n_shots_by_class} requested")
        selection.extend(random.sample(buckets[l], n_shots_by_class))
    random.shuffle(selection)
    if shuffle_label:
        selection = shuffle_label_func(selection)
    return selection



def create_prompt(demon_pool, query=None, num_shots_by_class=0, option="random", label_list=None, shuffle_label = False, prefixes = None, separators = None):
    
    prompt=""
    
    if prefixes is None:
        prefixes = {"input":"Input:", "output":"Output:"}
    
    if separators is None:
        separators = {"input":"\n", "output":"\n\n"}
    
    num_shots = len(label_list)*num_shots_by_class
    
    if num_shots==0:
        selection = []
    elif option == "stratify":
        selection = stratify_selection(demon_pool=demon_pool, num_shots=num_shots, label_list=label_list, shuffle_label=shuffle_label)
    elif option == "random":
        selection = random_selection(demon_pool=demon_pool, num_shots=num_shots, shuffle_label=shuffle_label)
    elif option == "all":
        selection = demon_pool
        random.shuffle(selection)
    else:
        selection = []
    
    for d in selection:
        prompt+=prefixes["input"]+" "+d['input']+separators["input"]+prefixes["output"]+" "+d['output']+separators["output"]
    
    if query is not None:
        prompt+=prefixes["input"]+" "+query['input']+separators["input"]+prefixes["output"]
        return prompt, query['output']
    
    return prompt

def create_prompt_generation(demon_pool, query=None, num_shots=0):
    prompt=""
    prefixes = {"input":"Input:", "output":"Output:"}
    separators = {"input":"\n", "output":"\n\n"}
    selection = random.sample(demon_pool, k=num_shots) if num_shots!=0 else []
    for d in selection:
        prompt+=prefixes["input"]+" "+d['input']+separators["input"]+prefixes["output"]+" "+d['output']+separators["output"]
    prompt+=prefixes["input"]+" "+query['input']+separators["input"]+prefixes["output"]

    return prompt, query['output']

def create_prompt_template_shuffle(demon_pool, query=None, num_shots_by_class=0, option="random", label_list=None, shuffle_label = False, separators = None):
    
    prompt=""
    
    prefixes = random.choice(ID_PROMPT_TEMPLATES)
    
    if separators is None:
        separators = {"input":"\n", "output":"\n\n"}
    
    num_shots = len(label_list)*num_shots_by_class
    
    if num_shots==0:
        selection = []
    elif option == "stratify":
        selection = stratify_selection(demon_pool=demon_pool, num_shots=num_shots, label_list=label_list, shuffle_label=shuffle_label)
    elif option == "random":
        selection = random_selection(demon_pool=demon_pool, num_shots=num_shots, shuffle_label=shuffle_label)
    elif option == "all":
        selection = demon_pool
        random.shuffle(selection)
    else:
        selection = []
    
    for d in selection:
        prompt+=prefixes["input"]+" "+d['input']+separators["input"]+prefixes["output"]+" "+d['output']+separators["output"]
    
    if query is not None:
        prompt+=prefixes["input"]+" "+query['input']+separators["input"]+prefixes["output"]
        return prompt, query['output']
    
    return prompt