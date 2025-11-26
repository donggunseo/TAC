import random
from collections import defaultdict

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



def create_prompt(demon_pool, query=None, num_shots_by_class=0, option="random", label_list=None, shuffle_label = False):
    prompt=""
    prefixes = {"input":"Input:", "output":"Output:"}
    separators = {"input":"\n", "output":"\n\n"}
    num_shots = len(label_list)*num_shots_by_class
    if num_shots==0:
        selection = []
    elif option == "stratify":
        selection = stratify_selection(demon_pool=demon_pool, num_shots=num_shots, label_list=label_list, shuffle_label=shuffle_label)
    elif option == "random":
        selection = random_selection(demon_pool=demon_pool, num_shots=num_shots, shuffle_label=shuffle_label)
    else:
        selection = []
    for d in selection:
        prompt+=prefixes["input"]+" "+d['input']+separators["input"]+prefixes["output"]+" "+d['output']+separators["output"]
    prompt+=prefixes["input"]+" "+query['input']+separators["input"]+prefixes["output"]

    return prompt, query['output']

def create_prompt_generation(demon_pool, query=None, num_shots=0):
    prompt=""
    prefixes = {"input":"Input:", "output":"Output:"}
    separators = {"input":"\n", "output":"\n\n"}
    selection = random.sample(demon_pool, k=num_shots) if num_shots!=0 else []
    for d in selection:
        prompt+=prefixes["input"]+" "+d['input']+separators["input"]+prefixes["output"]+" "+d['output']+separators["output"]
    prompt+=prefixes["input"]+" "+query['input']+separators["input"]+prefixes["output"]

    return prompt, query['output']
