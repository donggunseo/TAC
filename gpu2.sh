# CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name clinc150 --num_shots_by_class 3
# CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name banking77
# CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name trec_fine

CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name clinc150 --model_name qwen2.5-7b --num_shots_by_class 3
CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name banking77 --model_name qwen2.5-7b
CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name trec_fine --model_name qwen2.5-7b

CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name clinc150 --model_name llama3.2-1b --num_shots_by_class 3
CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name banking77 --model_name llama3.2-1b
CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name trec_fine --model_name llama3.2-1b

CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name clinc150 --model_name qwen2.5-1.5b --num_shots_by_class 3
CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name banking77 --model_name qwen2.5-1.5b
CUDA_VISIBLE_DEVICES=1 python3 act_converter_cls.py --dataset_name trec_fine --model_name qwen2.5-1.5b