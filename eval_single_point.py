from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import os
from tqdm import tqdm
from collections import Counter
import re



INST = "请回答以下航天知识点有关的提问，要求回答一个简短的词汇。你的结果不应该包含任何对该回答的详细描述以及特殊符号。"
LLAMA_INST = "请回答以下航天知识点有关的提问，要求回答一个简短的词汇。你的结果不应该包含任何对该回答的详细描述以及特殊符号。请用中文回答。"


def normalize_answer(answer):
    return ' '.join(answer.split())

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

def get_metric_dict(preds, goldens):
    metric_dict = {
        "em": get_em_score(preds, goldens),
        "f1": get_f1_score(preds, goldens)
    }
    return metric_dict
    
def get_em_score(preds, goldens_list):
    em = 0
    for pred, full_goldens in zip(preds, goldens_list):
        em += max([compute_exact(pred, gold) for gold in full_goldens])
    return em / len(preds)

def compute_exact(a_pred, a_gold):
    return int(a_pred == a_gold)

def get_f1_score(preds, goldens_list):
    f1 = 0
    for pred, full_goldens in zip(preds, goldens_list):
        f1 += max([compute_f1(pred, gold) for gold in full_goldens])
    return f1 / len(preds)

def compute_f1(pred, golden):
    f1_score = 0 
    c_pred = Counter(pred)
    c_gold = Counter(golden)
    nums_tp = sum((c_pred & c_gold).values())
    precision = nums_tp / sum(c_pred.values())
    recall = nums_tp / sum(c_gold.values())
    f1_score += 2 * (precision * recall) / (precision + recall) if precision != 0 and recall != 0 else 0
    return f1_score

def main(args):
    
    model_path = args.model_path
    model_name = model_path.split('/')[-1]

    input_path = args.input_path
    output_path = "output/{0}/single_point.jsonl".format(model_name)
    device = args.device

    
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    objs_list = []
    goldens_list = []
    preds_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
        file.seek(0)
        for line in tqdm(file, desc="generating answer:", ascii=True, dynamic_ncols=True, mininterval=0.5, total=total_lines):
            bench_obj = json.loads(line.strip())
            prompt = INST + bench_obj['input']
            messages = [
                {"role": "system", "content": "你是一个航天科普助手，可以回答用户提出的航天相关问题。"},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            pred = normalize_answer(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
            
            res_obj = {
                "task": "single_point",
                "input": bench_obj['input'],
                "golden": bench_obj['output'],
                "pred": pred
            }

            preds_list.append(pred)
            goldens_list.append(bench_obj['output'])
            objs_list.append(res_obj)

            
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in objs_list:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    metric = get_metric_dict(preds_list, goldens_list)
    print(metric)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default="../../models/HT-Model-Full-0906")
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--input_path', type=str, default='./datas/single_point.jsonl')

    args = parser.parse_args()

    main(args)

    