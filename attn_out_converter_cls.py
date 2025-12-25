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
from transformers import get_cosine_schedule_with_warmup
torch.set_grad_enabled(False)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
    parser.add_argument("--num_shots_by_class", type=int, default=5)
    parser.add_argument("--n_trials_by_class", type=int, default=5)
    parser.add_argument("--result_folder", type=str, default='./final_result3')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='banking77')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demon_selection", type=str, default = "stratify")
    parser.add_argument("--lr", type=float, default=7e-3)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
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

    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations
    
    if os.path.exists(os.path.join(res_dir, "head_act.pt")):
        activation_storage = torch.load(os.path.join(res_dir, "head_act.pt"))
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
                dummy_prompt, dummy_query_target = create_prompt(demon_pool=demon_pool, query = dummy_query, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
                dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
                dummy_output = model.generate(**dummy_tokenized_input, max_new_tokens = 10, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, do_sample=False, temperature = None, top_p = None, stop_strings = ["\n\n"])
                dummy_pred_str = tokenizer.decode(dummy_output.squeeze()[len(dummy_tokenized_input.input_ids.squeeze()):], skip_speical_tokens=True)
                dummy_pred_str = dummy_pred_str.strip()
                dummy_pred_str = dummy_pred_str.replace(tokenizer.eos_token, "")
                dummy_pred_str = dummy_pred_str.split("\n\n")[0]
                if dummy_pred_str==dummy_query_target:
                    torch.cuda.empty_cache()
                    with torch.no_grad() and TraceDict(model, layers=model_config['attn_hook_names'], retain_input=False, retain_output=True) as td:
                        outputs = model.forward(**dummy_tokenized_input)
                    cur_activation = torch.vstack([td[layer].output for layer in model_config['attn_hook_names']])[:,-1,:].cpu().detach()
                    activation_storage.append(cur_activation)
                    cur_act_cnt+=1
                else:
                    continue
        activation_storage = torch.stack(activation_storage)
        activation_storage = torch.mean(activation_storage, dim=0)
        torch.save(activation_storage, os.path.join(res_dir, "head_act.pt"))
    print("activation_shape:", activation_storage.shape)
    
    # def intervention_fn_train(head_act, model_config, batch_size, act_converter, device, batched_convert_idx):
    #     def mix_head_act(output, layer_name, inputs):
    #         current_layer = int(layer_name.split(".")[2])           
    #         intervention_embedding = head_act[current_layer,:].to(device)  #shape: (hidden_dim)
    #         if isinstance(output, tuple):
    #             output = output[0]
    #         modified_output = output.clone()
            
    #         completion_activations = []
    #         completion_lengths = []
    #         for batch_idx in range(batch_size):
    #             single_output = output[batch_idx] # (tokens (n), hidden_dim)
    #             convert_idx = batched_convert_idx[batch_idx]
    #             single_output = single_output[convert_idx:] # (generated_tokens, hidden_dim)
    #             completion_activations.append(single_output)
    #             completion_lengths.append(single_output.shape[0])
    #         longest = max(completion_lengths)
    #         for batch_idx in range(batch_size):
    #             if completion_activations[batch_idx].shape[0]<longest:
    #                 completion_activations[batch_idx] = torch.cat([completion_activations[batch_idx], torch.zeros((longest-completion_activations[batch_idx].shape[0], model_config['resid_dim']), dtype = torch.bfloat16).to(device)], dim=0)
    #         completion_activations = torch.stack(completion_activations, dim=0) ## (batch_size, padded_generated_tokens, hidden_dim)
    #         cell_state = intervention_embedding.repeat(batch_size,1).unsqueeze(0)
    #         out, _ = act_converter(completion_activations, (cell_state, cell_state)) # out shape : (batch_size, padded_generated_tokens, head_dim)
    #         for batch_idx in range(batch_size):
    #             modified_output[batch_idx, batched_convert_idx[batch_idx]:] += out[batch_idx, :completion_lengths[batch_idx]]
    #         modified_output = modified_output.contiguous()
    #         return modified_output
    #     return mix_head_act

    def intervention_fn_train(head_act, model_config, batch_size, act_converter, device, batched_convert_idx):
        """
        head_act: (n_layers, resid_dim)  or something indexable by [layer]
        batched_convert_idx: List[int] length = batch_size
        """

        D = model_config['resid_dim']

        def mix_head_act(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])

            # hook output가 tuple일 수 있음
            if isinstance(output, tuple):
                output = output[0]  # (B, T, D)

            # 실제 batch_size는 padding 때문에 다를 수 있으니 output 기준으로 잡는게 안전
            B = output.size(0)
            assert B == batch_size, f"batch_size mismatch: output has {B}, but batch_size arg is {batch_size}"

            # (1) 개입 임베딩 준비
            # head_act를 미리 device에 올려두면 여기서 .to(device) 안 해도 됨
            intervention_embedding = head_act[current_layer, :].to(device)  # (D,)

            # (2) completion 길이 계산: 각 샘플마다 convert_idx 이후 길이
            # output의 seq_len은 padding 포함 길이이므로,
            # convert_idx가 실제 prompt 끝 토큰 위치라는 전제가 유지되어야 함
            T = output.size(1)
            lengths = []
            for i in range(B):
                ci = batched_convert_idx[i]
                li = max(T - ci, 0)
                lengths.append(li)

            # 만약 전부 0이면 (이론상 거의 없겠지만) 그냥 원본 반환
            if max(lengths) == 0:
                return output

            Tmax = max(lengths)

            # (3) padded completion tensor 생성: (B, Tmax, D)
            # dtype/device는 output과 동일하게 맞춰서 안전하게
            completion = output.new_zeros((B, Tmax, D))  # (B, Tmax, D)

            # completion[b, :lengths[b]] = output[b, convert_idx[b] : convert_idx[b] + lengths[b]]
            for b in range(B):
                lb = lengths[b]
                if lb > 0:
                    ci = batched_convert_idx[b]
                    completion[b, :lb, :] = output[b, ci:ci+lb, :]

            # (4) pack해서 LSTM forward (padding 구간 계산 스킵)
            # lengths는 보통 CPU list/tensor가 제일 호환성 좋음
            lengths_cpu = torch.tensor(lengths, dtype=torch.long, device="cpu")

            packed = pack_padded_sequence(
                completion,
                lengths=lengths_cpu,
                batch_first=True,
                enforce_sorted=False
            )

            # 초기 hidden/cell: (num_layers * num_directions, B, hidden_size)
            # 너 원래 코드 의도대로 h0=c0=intervention_embedding 반복
            h0 = intervention_embedding.view(1, 1, D).expand(1, B, D).contiguous()
            c0 = h0

            packed_out, _ = act_converter(packed, (h0, c0))

            # 다시 padded로 복원: out -> (B, Tmax, D)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=Tmax)

            # (5) 원본 output에 out을 add (convert_idx 이후에만)
            modified_output = output.clone()
            for b in range(B):
                lb = lengths[b]
                if lb > 0:
                    ci = batched_convert_idx[b]
                    modified_output[b, ci:ci+lb, :] += out[b, :lb, :]

            return modified_output.contiguous()

        return mix_head_act

    h_status = []
    def intervention_fn_inference(head_act, model_config, act_converter, device, step=0):
        def mix_head_act_inference(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])
            if step==0:
                intervention_embedding = head_act[current_layer:current_layer+1,:].unsqueeze(0).to(device) #shape: (1,hidden_dim)
                # intervention_embedding = (torch.zeros(*intervention_embedding.shape, dtype=torch.bfloat16).to(device), intervention_embedding)
                intervention_embedding = (intervention_embedding, intervention_embedding)
            else:
                intervention_embedding = head_act[current_layer]
            if isinstance(output, tuple):
                output = output[0]
            modified_output = output.clone()
            out, status = act_converter(output[:,-1:,:], intervention_embedding) # out shape : (1, , head_dim)
            global h_status
            h_status.append(status)
            modified_output[:, -1:] += out # (batch_size, n_heads, head_dim)
            modified_output = modified_output.contiguous()
            return modified_output
        return mix_head_act_inference
    

    act_converter = nn.LSTM(input_size=model_config['resid_dim'], hidden_size=model_config['resid_dim'], batch_first=True,  dtype = torch.bfloat16)
    act_converter.to(device)

    best_val_acc = 0
    best_converter_state = None

    if os.path.exists(os.path.join(res_dir, "best_converter_state2.pt")):
        best_converter_state = torch.load(os.path.join(res_dir, "best_converter_state2.pt"))
    else:
        total_steps = ((len(train_dataset)+args.batch_size-1)//args.batch_size)*args.epoch
        warmup_steps = int(0.2*total_steps)
        optimizer = torch.optim.AdamW(act_converter.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps=total_steps)
        early_stop_cnt = 0
        early_stop_ths = 2
        for epoch in tqdm(range(args.epoch)):
            with torch.set_grad_enabled(True):
                random.shuffle(train_dataset)
                for idx, batch_start in tqdm(enumerate(range(0, len(train_dataset), args.batch_size))):
                    optimizer.zero_grad()
                    batch_end = min(batch_start + args.batch_size, len(train_dataset))
                    cur_batch_size = batch_end - batch_start
                    train_items = [train_dataset[i] for i in range(batch_start, batch_end)]
                    batched_train_prompt = []
                    batched_train_target = []
                    batched_convert_idx = []
                    for train_item in train_items:
                        train_prompt, train_target = create_prompt(demon_pool=None, query = train_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
                        batched_train_prompt.append(train_prompt+" "+train_target + "\n\n")
                        target_seq = " "+ train_target+ "\n\n" + tokenizer.eos_token
                        prompt_length = len(tokenizer.encode(train_prompt))
                        target_length = len(tokenizer.encode(target_seq, add_special_tokens = False))
                        batched_train_target.append([-100]*(prompt_length-1) + tokenizer.encode(target_seq, add_special_tokens=False))
                        batched_convert_idx.append(prompt_length-1)
                    train_tokenized_input = tokenizer(batched_train_prompt, return_tensors='pt', padding='longest')
                    train_tokenized_input.to(device)
                    batch_len = len(train_tokenized_input.input_ids[0])
                    for i, b in enumerate(batched_train_target):
                        if len(b)<batch_len:
                            batched_train_target[i]+=[-100]*(batch_len-len(b))
                    batched_train_target = torch.tensor(batched_train_target)
                    intervention_fn = intervention_fn_train(head_act = activation_storage, model_config = model_config, batch_size=cur_batch_size, act_converter=act_converter, device=device, batched_convert_idx=batched_convert_idx)
                    with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn, retain_input=False, retain_output=False):              
                        output = model.forward(**train_tokenized_input)
                    out_logit = output.logits[torch.arange(cur_batch_size)]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index = -100)
                    task_loss = loss_fct(out_logit.reshape(-1,out_logit.shape[-1]), batched_train_target.reshape(-1).to(device)).mean()
                    task_loss.backward()
                    torch.nn.utils.clip_grad_norm_(act_converter.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    if idx%10==0:
                        print("train loss : ", task_loss.item())
            torch.cuda.empty_cache()
            with torch.no_grad():
                correct_cnt=0
                for val_item in tqdm(valid_dataset):
                    val_prompt, val_target = create_prompt(demon_pool=None, query = val_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
                    val_tokenized_input = tokenizer(val_prompt, return_tensors='pt').to(device)
                    kv_cache = None
                    pred_seq = []
                    act_input = activation_storage
                    for t in range(args.max_generate_length):
                        intervention_fn = intervention_fn_inference(head_act = act_input, model_config = model_config, act_converter=act_converter, device = device, step=t)
                        with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn, retain_input=False, retain_output=False):              
                            output = model.forward(**val_tokenized_input, use_cache=True, past_key_values = kv_cache)
                            kv_cache = output.past_key_values
                        output_logits = output.logits[0,-1]
                        pred_token_id = torch.argmax(output_logits, dim=-1)
                        pred_seq.append(pred_token_id.item())
                        if pred_token_id==tokenizer.eos_token_id:
                            break
                        val_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                        val_tokenized_input['attention_mask'] = None
                        act_input = h_status.copy()
                        h_status = []
                    pred_str = tokenizer.decode(pred_seq).strip()
                    pred_str = pred_str.replace(tokenizer.eos_token, "")
                    pred_str = pred_str.split("\n\n")[0]
                    print(pred_str)
                    print(val_target)
                    print("_____________")
                    if pred_str == val_target:
                        correct_cnt+=1
                    h_status = []
                val_acc = correct_cnt/len(valid_dataset)
                print(f"Epoch {epoch+1} validation acc : {val_acc}")
                if val_acc>best_val_acc:
                    best_val_acc = val_acc
                    best_converter_state = act_converter.state_dict()
                else:
                    early_stop_cnt+=1
                if early_stop_cnt==early_stop_ths:
                    break
        torch.save(best_converter_state, os.path.join(res_dir, "best_converter_state2.pt"))
    act_converter.load_state_dict(best_converter_state)
    act_converter.eval()
    
    softmax = nn.Softmax(dim=-1)
    correct_cnt = 0
    act_convert_res = []
    new_line_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    with torch.no_grad():
        for test_item in tqdm(test_dataset):
            test_prompt, test_target = create_prompt(demon_pool=None, query = test_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
            test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
            kv_cache = None
            pred_seq = []
            act_input = activation_storage
            for t in range(args.max_generate_length):
                intervention_fn = intervention_fn_inference(head_act = act_input, model_config = model_config, act_converter=act_converter, device = device, step=t)
                with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn, retain_input=False, retain_output=False):              
                    output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                    kv_cache = output.past_key_values
                output_logits = output.logits[0,-1]
                pred_token_id = torch.argmax(output_logits, dim=-1)
                pred_seq.append(pred_token_id.item())
                if pred_token_id.item()==new_line_id:
                    break
                test_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                test_tokenized_input['attention_mask'] = None
                act_input = h_status.copy()
                h_status = []
            pred_str = tokenizer.decode(pred_seq).strip()
            pred_str = pred_str.replace(tokenizer.eos_token, "")
            pred_str = pred_str.split("\n\n")[0]
            print(pred_str)
            print(test_target)
            print("____________")
            if pred_str == test_target:
                correct_cnt+=1
            act_convert_res.append({"prompt": test_prompt, "query": test_item["input"], "gt": test_target, "pred": pred_str})
            h_status = []
        interv_acc = correct_cnt/len(test_dataset)
        print(interv_acc)
        site = {"acc": interv_acc, "res": act_convert_res}
    with open(os.path.join(res_dir, "extended_act_converter_result2.json"), "w") as f:
        json.dump(site, f)