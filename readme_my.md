#评测
分为两种模式，评测下游任务


#mylora-params

CUDA_VISIBLE_DEVICES=0,5  python run_crispedit.py  --model llama3-8b --data_type zsre  --energy_threshold 0.5    --batch_size 32  --projection_method_lora v2_grad --alg_name mylora --lr 


#eval_vllm
CUDA_VISIBLE_DEVICES=3 python run_edited_benchmarks_gpt.py --edited_model_dir llama3-8b_crispedit_zsre_0_5_10000 --data_type zsre --context_type no_context --alg_name ft_edit --model_name llama3-8b --evaluation_criteria llm_judge --judge_batch_size 16 --no_wandb

#eval
python run_edited_benchmarks.py --edited_model_dir llama3-8b_crispedit_zsre_0_5_10000 --model_name llama3-8b --max_length 40 --no_wandb --context_type no_context --alg_name Base --data_type zsre  --eval_num 3000  --evaluation_criteria llm_judge --no_wandb
