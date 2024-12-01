from transformers import AutoTokenizer
from utils import read_jsonl

import argparse
import json
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

SEED = 42

os.environ["HF_TOKEN"] = "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"

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_examples_prompt_content(examples):
    examples_prompt_content = ""
    for example in examples:
        examples_prompt_content += example["input"]
        examples_prompt_content += f"""
```python
{example["python_code"]}
```
Output
```
{example["answer"]}
```
"""
        examples_prompt_content += "\n"
    
    return examples_prompt_content

def get_prompt(tokenizer, system_message, user_message, assistant_message = None, examples = None):

    messages = []

    system_message_supported = True
    if "gemma" in tokenizer.name_or_path.lower():
        system_message_supported = False

    if system_message_supported and system_message:
        messages.append({
            "role": "system",
            "content": system_message,
        })
    
    if user_message:

        content = ""

        if not system_message_supported and system_message:
            content += f"{system_message}\n"

        if examples:
            examples_prompt_content = get_examples_prompt_content(examples)
            content += examples_prompt_content

        content += user_message

        messages.append({
            "role": "user",
            "content": content,
        })

    add_generation_prompt = True
    if assistant_message:
        messages.append({
            "role": "assistant",
            "content": assistant_message,
        })
        add_generation_prompt = False

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )

    return prompt

def verbalizer_finetune(tokenizer, data_point):

    system_message = data_point["instruction"]
    user_message = data_point["input"]
    assistant_message = f"""
```python
{data_point["python_code"]}
```
Output
```
{data_point["answer"]}
```
"""

    prompt = get_prompt(tokenizer, system_message, user_message, assistant_message=assistant_message)

    return prompt

def verbalizer_inference(tokenizer, data_point, examples=None):

    system_message = data_point["instruction"]
    user_message = data_point["input"]

    prompt = get_prompt(tokenizer, system_message, user_message, examples=examples)

    return prompt

def get_finetune_prompts(tokenizer, dataset):

    prompts = []
    for data_point in dataset:
        prompt = verbalizer_finetune(tokenizer, data_point)
        prompts.append(prompt)

    return prompts

def get_inference_prompts(tokenizer, dataset, examples=None):

    prompts = []
    for data_point in dataset:
        prompt = verbalizer_inference(tokenizer, data_point, examples=examples)
        prompts.append(prompt)

    return prompts

def save_prompts(prompts, file_path):
    with open(file_path, 'w') as f:
        for prompt in prompts:
            item = {
                "prompt": prompt
            }
            f.write(json.dumps(item) + '\n')

def sample_n_data_points(dataset, n):
    random.seed(SEED)
    return random.sample(dataset, n)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run script with specified model_id.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The model ID to use (e.g., meta-llama/Llama-3.1-8B-Instruct)."
    )
    args = parser.parse_args()

    # model_id = "meta-llama/Llama-3.2-3B-Instruct"
    # model_id = "google/gemma-2-2b-it"
    # model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    model_id = args.model_id

    tokenizer = get_tokenizer(model_id)
    # print(tokenizer.special_tokens_map)
    
    dataset_id = "dtruong46me/mathqa-python"
    data_dir = os.path.join(BASE_DIR, "data", dataset_id)

    dataset_train = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    dataset_test = read_jsonl(os.path.join(data_dir, "test.jsonl"))
    dataset_challenge_test = read_jsonl(os.path.join(data_dir, "challenge_test.jsonl"))

    prompts_train = get_finetune_prompts(tokenizer, dataset_train)
    prompts_test = get_inference_prompts(tokenizer, dataset_test)
    prompts_challenge_test = get_inference_prompts(tokenizer, dataset_challenge_test)


    prompts_dir = os.path.join(BASE_DIR, "prompts", model_id)
    os.makedirs(prompts_dir, exist_ok=True)

    finetune_prompts_file_path = os.path.join(prompts_dir, "finetune_prompts.jsonl")
    save_prompts(prompts_train, finetune_prompts_file_path)
    inference_prompts_file_path = os.path.join(prompts_dir, "inference_prompts.jsonl")
    save_prompts(prompts_test, inference_prompts_file_path)
    inference_challenge_prompts_file_path = os.path.join(prompts_dir, "inference_challenge_prompts.jsonl")
    save_prompts(prompts_challenge_test, inference_challenge_prompts_file_path)


    # few_shot_ns = [3, 10]
    # for few_shot_n in few_shot_ns:
    #     examples = sample_n_data_points(dataset_train, few_shot_n)
    #     prompts_test_few_shot = get_inference_prompts(tokenizer, dataset_test, examples)
    #     prompts_challenge_test_few_shot = get_inference_prompts(tokenizer, dataset_challenge_test, examples)

    #     inference_prompts_few_shot_file_path = os.path.join(prompts_dir, f"inference_prompts_few_shot_{few_shot_n}.jsonl")
    #     save_prompts(prompts_test_few_shot, inference_prompts_few_shot_file_path)
    #     inference_challenge_prompts_few_shot_file_path = os.path.join(prompts_dir, f"inference_challenge_prompts_few_shot_{few_shot_n}.jsonl")
    #     save_prompts(prompts_challenge_test_few_shot, inference_challenge_prompts_few_shot_file_path)

    print("Prompts generated successfully!")