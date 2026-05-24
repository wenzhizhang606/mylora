#评测
分为两种模式，评测下游任务


#mylora-params

CUDA_VISIBLE_DEVICES=0,5  python run_crispedit.py  --model llama3-8b --data_type zsre  --energy_threshold 0.5    --batch_size 32  --projection_method_lora v2_grad --alg_name mylora


#eval
python run_edited_benchmarks_gpt.py --edited_model_dir xxx --data_type zsre 
  --context_type qa_inst \ 
  --alg_name ft_edit \
  --model_name xxx \
  --evaluation_criteria llm_judge \
  --judge_batch_size 16