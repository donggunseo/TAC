from comet import load_from_checkpoint, download_model
import json

task = "wmt19_en-de"
model = "qwen2.5-7b"

with open(f"./result_extended/{task}/{model}/extended_act_converter_result.json", "r") as f:
    tac_res = json.load(f)

# with open(f"./result_extended/{task}/{model}/few_shot_result.json", "r") as f:
#     tac_res = json.load(f)

hyp = [item['pred'] for item in tac_res['res']]
refs = [item['gt'] for item in tac_res['res']]

with open(f"./dataset/{task}/test.json", "r") as f:
    test_dataset = json.load(f)

src = [item['input'] for item in test_dataset]

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

data = [{"src" : s, "mt": p, "ref" : r} for s,p,r in zip(src, hyp, refs)]

scores = comet_model.predict(data, batch_size=16, gpus=1)
print(sum(scores["scores"])/len(scores["scores"]))