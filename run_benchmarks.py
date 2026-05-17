#所有评测的总入口，参考run_crispedit.py,所有的评测都需要从这里分发下去
import argparse
import os
import re





def get_arguments():
    parser = argparse.ArgumentParser()
    # 根据测试类型分发
    parser.add_argument("--test_mode",required=True, type=str, default="base", choices=['base','edited'] ,help='Select Test Mode')

    parser.add_argument('--edited_model_dir', required=True, type=str, default=None, help='Path to edited model for evaluation.')
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'multi_counterfact', 'wiki', 'safeedit_train', 'safeedit_test'])
    parser.add_argument('--eval_num', required=False, type=int, default=3000, help='Number of evaluation instances to use. Default uses all.')
    parser.add_argument('--max_length', required=False, type=int, default=40, help='Maximum length of the generated sequences.')
    parser.add_argument('--context_type', required=True, type=str, default='qa_inst', choices=['qa_inst', 'chat_temp', 'no_context'], help='Type of context to use for evaluation.')
    
    parser.add_argument('--alg_name', required=True, type=str, default='ft_edit', help='Name of the editing algorithm used.')
    #parser.add_argument('--model_name', required=True, type=str, default='gpt2-xl', help='Name of the base model used.')
    parser.add_argument('--evaluation_criteria', required=True, type=str, default='exact_match', choices=['exact_match', 'llm_judge'], help='Evaluation criteria to use.')  
    parser.add_argument('--wandb_project', type=str, default='CrispEdit_EVAL', help='WandB project name.')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='WandB run ID for resuming runs.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging.')

    # 新增参数，控制使用哪一张显卡 
    parser.add_argument('--cuda_visible_devices', required=True, type=str, default="0", help='Select Available GPU')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    # 提取命令行参数
    args = get_arguments()
    #分发评测
    