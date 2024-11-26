import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import random
from tqdm import tqdm
import os

random.seed(42)

TEMPLATE = {
    "运载火箭": "请介绍一下运载火箭：{title}。",
    "综合任务": "请介绍一下航天综合任务：{title}。",
    "机构": "请介绍一下航天机构：{title}。",
    "卫星": "请介绍一下卫星：{title}。",
    "发射场": "请介绍一下航天发射场：{title}。",
    "载人飞船": "请介绍一下载人飞船：{title}。",
    "行星车": "请介绍一下行星车：{title}。",
    "空间站": "请介绍一下空间站：{title}。",
    "支持设施": "请介绍一下航天支持设施：{title}。",
    "航天人": "请介绍一下航天人：{title}。",
}



def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def main(args):
    output_path = f"./output/{args.model.split('/')[-1]}/factual_long.jsonl"
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    output = open(output_path, "w", encoding="utf-8")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ratio_list = []
    data = read_jsonl_file(args.input_file)
    random.shuffle(data)

    for item in tqdm(data):
        info_type = item['info_type']
        title = item['title']
        instruction = TEMPLATE[info_type].format(title=title)
        messages = [
            {"role": "user", "content": instruction}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(args.device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        item['response'] = response
        output.write(json.dumps(item, ensure_ascii=False) + "\n")
        output.flush()

        pairs = item['pairs']
        values = [i[1] for i in pairs]
        count = 0
        for value in values:
            if value in response:
                count += 1
        ratio = count / len(values)
        ratio_list.append(ratio)

    average_ratio = sum(ratio_list) / len(ratio_list)
    print(average_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='../../models/HT-Model-Full-0906')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_file', type=str, default='./datas/factual_long.jsonl')
    args = parser.parse_args()
    main(args)
    
