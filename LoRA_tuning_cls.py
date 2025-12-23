import argparse
import os
import json
import torch
from tqdm import tqdm
from model_utils import *
from data_utils import *
from prompt_utils import *
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

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
    parser.add_argument("--result_folder", type=str, default='./final_result2')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='banking77')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_generate_length", type=int, default=10)

    args = parser.parse_args()

    set_seed(args.seed)
    model_card = MODEL_CARD[args.model_name]

    res_dir = os.path.join(args.result_folder, args.dataset_name, args.model_name, str(args.seed))

    os.makedirs(res_dir, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, label_list = load_data(args.dataset_name, args.data_dir)

    device = 'cuda'
    model, tokenizer, model_config = load_model_and_tokenizer(model_card)
    model.to(device)

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.2
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    lora_dir = os.path.join(res_dir, "LoRA")
    os.makedirs(lora_dir, exist_ok=True)


    trainable_params = [p for p in model.parameters() if p.requires_grad]
    tokenizer.padding_size="right"

    def label_text_match(pred, label_list):
        for l in label_list:
            label_len = len(l)
            if pred[:label_len]==l:
                return pred[:label_len]
        return pred

    if not os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors")):
        best_val_acc = 0
        total_steps = ((len(train_dataset)+args.batch_size-1)//args.batch_size)*args.epoch
        warmup_steps = int(0.2*total_steps)
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.001)
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
                    for train_item in train_items:
                        train_prompt, train_target = create_prompt(demon_pool=None, query = train_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
                        batched_train_prompt.append(train_prompt+" "+train_target)
                        prompt_length = len(tokenizer.encode(train_prompt))
                        target_length = len(tokenizer.encode(" "+ train_target+ tokenizer.eos_token, add_special_tokens=False))
                        batched_train_target.append([-100]*(prompt_length-1) + tokenizer.encode(" "+ train_target+ tokenizer.eos_token, add_special_tokens=False))
                    train_tokenized_input = tokenizer(batched_train_prompt, return_tensors='pt', padding='longest')
                    batch_len = len(train_tokenized_input.input_ids[0])
                    for i, b in enumerate(batched_train_target):
                        if len(b)<batch_len:
                            batched_train_target[i]+=[-100]*(batch_len-len(b))
                    batched_train_target = torch.tensor(batched_train_target)
                    train_tokenized_input.to(device)            
                    output = model.forward(**train_tokenized_input)
                    out_logit = output.logits[torch.arange(cur_batch_size),:,:]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index = -100)
                    task_loss = loss_fct(out_logit.reshape(-1,out_logit.shape[-1]), batched_train_target.reshape(-1).to(device)).mean()
                    task_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    if idx%10==0:
                        print("train loss : ", task_loss.item())
            torch.cuda.empty_cache()
            correct_cnt=0
            with torch.no_grad():
                for val_item in tqdm(valid_dataset):
                    val_prompt, val_target = create_prompt(demon_pool=None, query = val_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
                    val_tokenized_input = tokenizer(val_prompt, return_tensors='pt').to(device)
                    kv_cache = None
                    pred_seq = []
                    for t in range(args.max_generate_length):
                        output = model.forward(**val_tokenized_input, use_cache=True, past_key_values = kv_cache)
                        kv_cache = output.past_key_values
                        output_logits = output.logits[0,-1]
                        pred_token_id = torch.argmax(output_logits, dim=-1)
                        pred_seq.append(pred_token_id.item())
                        if pred_token_id.item()==tokenizer.eos_token_id:
                            break
                        val_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                        val_tokenized_input['attention_mask'] = None
                    pred_str = tokenizer.decode(pred_seq).strip()
                    pred_str = pred_str.split(tokenizer.eos_token)[0]
                    pred_str = pred_str.replace(tokenizer.eos_token, "")
                    print(pred_str)
                    print(val_target)
                    print("_____________")
                    if pred_str == val_target:
                        correct_cnt+=1
                val_acc = correct_cnt/len(valid_dataset)
                print(f"Epoch {epoch+1} validation acc : {val_acc}")
                if val_acc>=best_val_acc:
                    best_val_acc = val_acc
                    model.save_pretrained(lora_dir)
                else:
                    early_stop_cnt+=1
                if early_stop_cnt==early_stop_ths:
                    break
    del model
    model, tokenizer, model_config = load_model_and_tokenizer(model_card)
    model.to(device)
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    res = []
    correct_cnt=0
    with torch.no_grad():
        for test_item in tqdm(test_dataset):
            test_prompt, test_target = create_prompt(demon_pool=None, query = test_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False, prefixes = {"input":"Sentence:", "output":"Label:"})
            test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
            kv_cache = None
            pred_seq = []
            for t in range(args.max_generate_length):
                output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                kv_cache = output.past_key_values
                output_logits = output.logits[0,-1]
                pred_token_id = torch.argmax(output_logits, dim=-1)
                pred_seq.append(pred_token_id.item())
                if pred_token_id.item()==tokenizer.eos_token_id:
                    break
                test_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                test_tokenized_input['attention_mask'] = None
            pred_str = tokenizer.decode(pred_seq).strip()
            pred_str = pred_str.split(tokenizer.eos_token)[0]
            pred_str = pred_str.replace(tokenizer.eos_token, "")
            print(pred_str)
            print(test_target)
            print("_____________")
            if pred_str == test_target:
                correct_cnt+=1
            res.append({"prompt": test_prompt, "query": test_item["input"], "gt": test_target, "pred": pred_str})
        test_acc = correct_cnt/len(test_dataset)
        print("acc : ", test_acc)
    # lora = {"acc": test_acc, "res": res}
    # with open(os.path.join(res_dir, "lora_result.json"), "w") as f:
    #     json.dump(lora, f)
            


    