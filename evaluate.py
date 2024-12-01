from tqdm import tqdm
from utils import extract_code_block, get_python_code_output, read_jsonl, safe_parse_float

import argparse
import json
import math
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

SEED = 42

CODE_BLOCK_START = "```python"
CODE_BLOCK_END = "```"


def execute_generations(generation_dir, output_file_path):

    if not os.path.exists(generation_dir):
        print(f"\nSkipping... Generation directory not found: {generation_dir}")
        return

    outputs = dict()
    count = 0

    file_list = os.listdir(generation_dir)
    file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # # Method 2: Using sorted with lambda
    # file_list = sorted(file_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
    #print(file_list)
    for file_name in tqdm(file_list):
        if not file_name.endswith(".txt"):
            continue
        #print(file_name)
        with open(os.path.join(generation_dir, file_name), "r") as f:
            generation = f.read()
        #print(generation)
        code_block = extract_code_block(generation, CODE_BLOCK_START, CODE_BLOCK_END)
        #print(code_block)
        if code_block is None:
            continue

        output = get_python_code_output(code_block)
        if output is None:
            continue
        #file_index = int(file_name.split("_")[1][:2])
        file_index = count
        outputs[file_index] = output.strip()
        count += 1

    with open(output_file_path, "w") as f:
        json.dump(outputs, f, indent=4)

def compute_accuracy(dataset_jsonl, output_file_path):
    if not os.path.exists(output_file_path):
        print(f"Skipping... Output file not found: {output_file_path}")
        return
    with open(output_file_path, "r") as f:
        outputs = json.load(f)

    outputs = {int(k): v for k, v in outputs.items()}

    correct = 0
    total = 0

    # read dataset
    dataset = read_jsonl(dataset_jsonl)

    for i, data in enumerate(dataset):
        if i not in outputs:
            continue

        total += 1

        expected_output = data["answer"]
        actual_output = safe_parse_float(outputs[i])

        if actual_output is not None and math.isclose(actual_output, expected_output, rel_tol=0, abs_tol=1e-6):
            correct += 1
        
    if total == 0:
        return None
    else:
        return correct / total

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run script with specified model_id.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        help="The model ID to use (e.g., meta-llama/Llama-3.1-8B-Instruct)."
    )
    #args = parser.parse_args()
    #model_id = args.model_id
    #model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    # model_id = "google/gemma-2-2b-it"
    model_id = "meta-llama/Llama-3.2-3B-Instruct_baseline"
    # model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

    dataset_id = "dtruong46me/mathqa-python"

    generation_dir = os.path.join(BASE_DIR, "generations_llama")
    evaluation_dir = os.path.join(BASE_DIR, "evaluation", model_id)
    data_dir = os.path.join(BASE_DIR, "data", dataset_id)
    
    dataset_test = read_jsonl(os.path.join(data_dir, "test.jsonl"))
    dataset_challenge_test = read_jsonl(os.path.join(data_dir, "challenge_test.jsonl"))


    default_test_output_dir = os.path.join(evaluation_dir, "default")
    os.makedirs(default_test_output_dir, exist_ok=True)

    accuracies = dict()

    #default_test_generation_dir = os.path.join(generation_dir, "default", "test")
    default_test_output = os.path.join(default_test_output_dir, "test.json")
    #execute_generations(default_test_generation_dir, default_test_output)
    execute_generations(generation_dir, default_test_output)
    default_test_accuracy = compute_accuracy(os.path.join(data_dir, "test.jsonl"), default_test_output)
    print(f"Default Test Accuracy: {default_test_accuracy}")
    accuracies["default_test_accuracy"] = default_test_accuracy

    # default_challenge_test_generation_dir = os.path.join(generation_dir, "default", "challenge_test")
    # default_challenge_test_output = os.path.join(default_test_output_dir, "challenge_test.json")
    # execute_generations(default_challenge_test_generation_dir, default_challenge_test_output)
    # default_challenge_test_accuracy = compute_accuracy(os.path.join(data_dir, "challenge_test.jsonl"), default_challenge_test_output)
    # print(f"Default Challenge Test Accuracy: {default_challenge_test_accuracy}")
    # accuracies["default_challenge_test_accuracy"] = default_challenge_test_accuracy


    # few_shot_ns = [3, 10]
    # for few_shot_n in few_shot_ns:

    #     few_shot_output_dir = os.path.join(evaluation_dir, f"few_shot_{few_shot_n}")
    #     os.makedirs(few_shot_output_dir, exist_ok=True)

    #     few_shot_generation_dir = os.path.join(generation_dir, f"few_shot_{few_shot_n}", "test")
    #     few_shot_output = os.path.join(few_shot_output_dir, "test.json")
    #     execute_generations(few_shot_generation_dir, few_shot_output)
    #     few_shot_accuracy = compute_accuracy(os.path.join(data_dir, "test.jsonl"), few_shot_output)
    #     print(f"Few Shot {few_shot_n} Test Accuracy: {few_shot_accuracy}")
    #     accuracies[f"few_shot_{few_shot_n}_test_accuracy"] = few_shot_accuracy

    #     few_shot_challenge_generation_dir = os.path.join(generation_dir, f"few_shot_{few_shot_n}", "challenge_test")
    #     few_shot_challenge_output = os.path.join(few_shot_output_dir, "challenge_test.json")
    #     execute_generations(few_shot_challenge_generation_dir, few_shot_challenge_output)
    #     few_shot_challenge_accuracy = compute_accuracy(os.path.join(data_dir, "challenge_test.jsonl"), few_shot_challenge_output)
    #     print(f"Few Shot {few_shot_n} Challenge Test Accuracy: {few_shot_challenge_accuracy}")
    #     accuracies[f"few_shot_{few_shot_n}_challenge_test_accuracy"] = few_shot_challenge_accuracy


    accuracies_output_file = os.path.join(evaluation_dir, "accuracies.json")
    with open(accuracies_output_file, "w") as f:
        json.dump(accuracies, f, indent=4)

    print("Outputs saved successfully.")