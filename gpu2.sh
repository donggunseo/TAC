CUDA_VISIBLE_DEVICES=1 python3 few_shot_cls.py --dataset_name banking77 --seed 42 --num_shots_by_class 35 --model_name llama3.2-1b
CUDA_VISIBLE_DEVICES=1 python3 few_shot_cls.py  --dataset_name trec_fine --seed 42 --num_shots_by_class 4 --model_name llama3.2-1b
CUDA_VISIBLE_DEVICES=1 python3 few_shot_cls.py  --dataset_name clinc150 --seed 42 --num_shots_by_class 30 --model_name llama3.2-1b
