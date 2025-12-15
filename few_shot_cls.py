import argparse
import os
import json
import torch
from tqdm import tqdm
from model_utils import *
from data_utils import *
from prompt_utils import *
torch.set_grad_enabled(False)
import copy

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
    parser.add_argument("--num_examples_by_class", type=int, default=3)
    parser.add_argument("--num_shots_by_class", type=int, default=3)
    parser.add_argument("--result_folder", type=str, default='./final_result')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='banking77')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demon_selection", type=str, default = "stratify")
    parser.add_argument("--max_generate_length", type=int, default=10)

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


    demon_prompt = create_prompt(demon_pool=train_dataset, query = None, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
    demon_tokenized_input = tokenizer(demon_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        prefix_out = model(**demon_tokenized_input, use_cache=True)
    kv_cache = prefix_out.past_key_values
    prefix_len = demon_tokenized_input.input_ids.shape[-1]
    prefix_ids = demon_tokenized_input.input_ids

    correct_cnt = 0
    res = []
    with torch.no_grad():
        for test_item in tqdm(test_dataset):
            test_prompt, test_target = create_prompt(demon_pool=None, query = test_item, num_shots_by_class=0, option=args.demon_selection, label_list=label_list, shuffle_label=False)
            print(test_prompt)
            query_input_ids = tokenizer(test_prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
            query_len = query_input_ids.shape[-1]
            prefix_kv_cache = copy.deepcopy(kv_cache)
            prefill_out = model(input_ids = query_input_ids, past_key_values = prefix_kv_cache, use_cache=True)
            past = prefill_out.past_key_values
            logits = prefill_out.logits             
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated = [next_token.item()]
            for _ in range(args.max_generate_length-1):
                input_ids_step = next_token.unsqueeze(-1)  # [1, 1]
                with torch.no_grad():
                    out = model(
                        input_ids=input_ids_step,
                        past_key_values=past,
                        use_cache=True,
                    )
                past = out.past_key_values
                logits = out.logits  # [1, 1, vocab]
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                token_id = next_token.item()
                if token_id == tokenizer.eos_token_id or token_id == tokenizer.encode("\n\n", add_special_tokens=False)[0]:
                    break
                generated.append(token_id)
            output_str = tokenizer.decode(generated, skip_speical_tokens=True)
            output_str = output_str.strip()
            output_str = output_str.replace(tokenizer.eos_token, "")
            output_str = output_str.split("\n\n")[0]
            print(output_str)
            print(test_target)
            print("_______________")
            if output_str == test_target:
                correct_cnt+=1
            res.append({"prompt": test_prompt, "query": test_item["input"], "gt": test_target, "pred": output_str})
    acc = correct_cnt/len(test_dataset)
    print("acc: ", acc)
    fs_result = {"acc": acc, "res": res}
    with open(os.path.join(res_dir, "few_shot_result.json"), "w") as f:
        json.dump(fs_result, f)
