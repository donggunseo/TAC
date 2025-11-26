import argparse
import os
import json
import torch
from tqdm import tqdm
from model_utils import *
from data_utils import *
from prompt_utils import *
torch.set_grad_enabled(False)
from baukit import TraceDict, get_module
import torch.nn as nn

MODEL_CARD={
    "meta-llama/Llama-3.1-8B": "llama3.1-8b",
    "Qwen/Qwen2.5-7B" : "qwen2.5-7b",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument("--num_shots_by_class", type=int, default=2)
    parser.add_argument("--result_folder", type=str, default='./result_extended')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='banking77')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demon_selection", type=str, default = "stratify")

    args = parser.parse_args()

    set_seed(args.seed)
    model_card = MODEL_CARD[args.model_name]

    res_dir = os.path.join(args.result_folder, args.dataset_name, model_card)

    os.makedirs(res_dir, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, label_list = load_data(args.dataset_name, args.data_dir)
    
    device = 'cuda'
    model, tokenizer, model_config = load_model_and_tokenizer(args.model_name)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

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
            res.append({"prompt": test_prompt, "gt": test_target, "pred": output_str})
    acc = correct_cnt/len(test_dataset)
    print("acc: ", acc)
    fs_result = {"acc": acc, "res": res}
    with open(os.path.join(res_dir, "few_shot_result.json"), "w") as f:
        json.dump(fs_result, f)
