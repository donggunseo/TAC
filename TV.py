import argparse
import os
import json
import torch
from tqdm import tqdm
from model_utils import *
from data_utils import *
from prompt_utils import *
from baukit import TraceDict, get_module
import torch.nn as nn
from collections import defaultdict
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
    parser.add_argument("--num_shots_by_class", type=int, default=2)
    parser.add_argument("--n_trials_by_class", type=int, default=3)
    parser.add_argument("--result_folder", type=str, default='./exp1')
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

    if os.path.exists(os.path.join(res_dir, "hidden_act.pt")):
        activation_storage = torch.load(os.path.join(res_dir, "hidden_act.pt"))
    else:
        train_buckets = defaultdict(list)
        for item in train_dataset:
            if item["output"] in label_list:
                train_buckets[item["output"]].append(item)
        activation_storage = []
        num_act_by_class = args.n_trials_by_class
        for l in tqdm(label_list):
            cur_act_cnt = 0
            while cur_act_cnt<num_act_by_class:
                dummy_query = random.choice(train_buckets[l])
                demon_pool = [item for item in train_dataset if item!=dummy_query]
                dummy_prompt, dummy_query_target, _= create_prompt(demon_pool=demon_pool, query = dummy_query, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
                dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
                dummy_output = model.generate(**dummy_tokenized_input, max_new_tokens = 10, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, do_sample=False, temperature = None, top_p = None, stop_strings = ["\n\n"])
                dummy_pred_str = tokenizer.decode(dummy_output.squeeze()[len(dummy_tokenized_input.input_ids.squeeze()):], skip_speical_tokens=True)
                dummy_pred_str = dummy_pred_str.strip()
                dummy_pred_str = dummy_pred_str.replace(tokenizer.eos_token, "")
                dummy_pred_str = dummy_pred_str.split("\n\n")[0]
                if dummy_pred_str==dummy_query_target:
                    torch.cuda.empty_cache()
                    with torch.no_grad() and TraceDict(model, layers=model_config['layer_hook_names'], retain_input=False, retain_output=True) as td:
                        outputs = model.forward(**dummy_tokenized_input)
                    stack_initial = torch.vstack([td[layer].output for layer in model_config['layer_hook_names']])
                    cur_activation = stack_initial[:, -1, :].cpu().detach()
                    activation_storage.append(cur_activation)
                    cur_act_cnt+=1
                else:
                    continue
        activation_storage = torch.stack(activation_storage)
        activation_storage = torch.mean(activation_storage, dim=0)
        torch.save(activation_storage, os.path.join(res_dir, "hidden_act.pt"))
    print(activation_storage.shape)


    def add_task_vector(edit_layer, task_vector, device, idx=-1):
        def add_act(output, layer_name):
            current_layer = int(layer_name.split(".")[2])
            if current_layer == edit_layer:
                if isinstance(output, tuple):
                    output[0][:, idx] = task_vector.to(device)
                    return output
                else:
                    output[:, idx] = task_vector.unsqueeze(0).to(device)
                    return output
            else:
                return output
        return add_act
    
    softmax = nn.Softmax(dim=-1)
    correct_cnt = 0
    first_token_match=0
    new_line_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    edit_layer = 15
    tv_res = []
    with torch.no_grad():
        tv = activation_storage[edit_layer]
        for test_item in tqdm(test_dataset[1500:2000]):
            test_prompt, test_target, _ = create_prompt(demon_pool=None, query = test_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
            test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
            kv_cache = None
            pred_seq = []
            prob_seq= []
            target_seq = tokenizer.encode(": "+test_target, add_special_tokens=False)[1:]
            for t in range(args.max_generate_length):
                if t==0:
                    intervention_fn = add_task_vector(edit_layer = edit_layer, task_vector = tv, device=device, idx=-1)
                    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):              
                        output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                        kv_cache = output.past_key_values
                else:
                    output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                output_logits = output.logits[0,-1]
                output_prob = softmax(output_logits)
                topk_probs, topk_indices = torch.topk(output_prob, k=10)
                probs_dict = {tokenizer.decode(int(token_id)): float(prob) for token_id, prob in zip(topk_indices, topk_probs)}
                pred_token_id = torch.argmax(output_logits, dim=-1)
                pred_seq.append(pred_token_id.item())
                prob_seq.append(probs_dict)
                if pred_token_id.item()==new_line_id or pred_token_id.item()==tokenizer.eos_token_id:
                    break
                test_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                test_tokenized_input['attention_mask'] = None
            pred_str = tokenizer.decode(pred_seq).strip()
            pred_str = pred_str.replace(tokenizer.eos_token, "")
            pred_str = pred_str.split("\n\n")[0]
            print(pred_str)
            print(test_target)
            print("____________")
            if pred_str == test_target:
                correct_cnt+=1
            if pred_seq[0]==target_seq[0]:
                first_token_match+=1
            tv_res.append({"prompt": test_prompt, "gt": test_target, "pred": pred_str, "probs": prob_seq})
        test_interv_acc = correct_cnt/len(test_dataset)
        print(test_interv_acc)
        print(first_token_match)
        res = {"acc": test_interv_acc, "res": tv_res}
    with open(os.path.join(res_dir, "tv_result.json"), "w") as f:
        json.dump(res, f)