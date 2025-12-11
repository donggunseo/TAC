import argparse
import os
import json
import torch
from tqdm import tqdm
from model_utils import *
from data_utils import *
from prompt_utils import *
torch.set_grad_enabled(False)

MODEL_CARD={
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen2.5-7b" : "Qwen/Qwen2.5-7B",
    "gemma3-12b" : "google/gemma-3-12b-pt",
    "llama3.2-1b" : "meta-llama/Llama-3.2-1B",
    "qwen2.5-1.5b" : "Qwen/Qwen2.5-1.5B",
    "gemma3-1b" : "google/gemma-3-1b-pt"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama3.1-8b')
    parser.add_argument("--num_examples_by_class", type=int, default=2)
    parser.add_argument("--num_shots_by_class", type=int, default=2)
    parser.add_argument("--result_folder", type=str, default='./final_result')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='banking77')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demon_selection", type=str, default = "stratify")

    args = parser.parse_args()

    set_seed(args.seed)
    model_card = MODEL_CARD[args.model_name]

    res_dir = os.path.join(args.result_folder, args.dataset_name, args.model_name, str(args.seed))

    os.makedirs(res_dir, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, label_list = load_data(args.dataset_name, args.data_dir)
    
    device = 'cuda'
    model, tokenizer, model_config = load_model_and_tokenizer(model_card)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    train_buckets = defaultdict(list)
    for item in train_dataset:
        if item["output"] in label_list:
            train_buckets[item["output"]].append(item)
    for k,v in train_buckets.items():
        train_buckets[k] = random.sample(train_buckets[k], args.num_examples_by_class)
    train_dataset = [item for v in train_buckets.values() for item in v]

    correct_cnt = 0
    res = []
    with torch.no_grad():
        for test_item in tqdm(test_dataset):
            test_prompt, test_target = create_prompt(demon_pool=train_dataset, query = test_item, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
            test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
            outputs = model.generate(**test_tokenized_input, max_new_tokens = 10, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, do_sample=False, temperature = None, top_p = None)
            output_str = tokenizer.decode(outputs.squeeze()[len(test_tokenized_input.input_ids.squeeze()):], skip_speical_tokens=True)
            output_str = output_str.strip()
            output_str = output_str.replace(tokenizer.eos_token, "")
            output_str = output_str.split("\n\n")[0]
            if output_str == test_target:
                correct_cnt+=1
            res.append({"prompt": test_prompt, "query": test_item["input"], "gt": test_target, "pred": output_str})
    acc = correct_cnt/len(test_dataset)
    print("acc: ", acc)
    fs_result = {"acc": acc, "res": res}
    with open(os.path.join(res_dir, "few_shot_result.json"), "w") as f:
        json.dump(fs_result, f)
