import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import json
import os
import tqdm

file_path = "/home/ubuntu/codegen/inference_prompts_baseline.jsonl"

output_folder = "generations_llama_baseline"
os.makedirs(output_folder, exist_ok=True)

#output_file = "generations.txt"

def load_qlora_model(base_model_name="meta-llama/Llama-3.2-3B-Instruct", adapter_path="math_solver_qlora"):
    """
    Load the QLoRA fine-tuned model and tokenizer.
    
    Args:
        base_model_name: Name of the base model used for fine-tuning
        adapter_path: Path to the saved adapter weights
    """
    # Configure 4-bit quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load the model with PEFT adapter
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     adapter_path,
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    #     quantization_config=bnb_config
    # )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # # Load the PEFT adapter
    # model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_solution(model, tokenizer, problem_text, max_length=1000):
    """
    Generate solution for a given math problem.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        problem_text: The math problem text
        max_length: Maximum length of generated text
    """
    # Format the input prompt
#     prompt = f"""<|system|>You are an expert in Python. Solve the following math problems using Python code. Return the final answer in square brackets.\n<|eot_id|>
# <|start_header_id|>user<|end_header_id|>Solve this math problem: {problem_text}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>Certainly! I'll solve the problem and provide a Python code solution."""
    #prompt = f"""You are an expert in Python. Solve the following math problems using Python code. Return the final answer in square brackets.\n Solve this math problem: {problem_text}"""
    
    # Tokenize input
    prompt = problem_text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate solution
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("Generated:",generated_text)
    return generated_text

def batch_generate_solutions(model, tokenizer, problems, max_length=512, batch_size=2):
    """
    Generate solutions for multiple problems in batches.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        problems: List of problem texts
        max_length: Maximum length of generated text
        batch_size: Number of problems to process at once
    """
    solutions = []
    for i in tqdm.tqdm(range(0, len(problems), batch_size)):
        batch_prompts = []
        batch = problems[i:i + batch_size]
        #print("HERE")
        #print(batch)
        for datapt in batch:
            prompt = datapt['prompt']
            batch_prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate solutions for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode outputs
        batch_solutions = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        solutions.extend(batch_solutions)
    
    return solutions

# Example usage
if __name__ == "__main__":
    # Load the model
    model, tokenizer = load_qlora_model()
    
    # Single problem inference
    # problem = "# if the price of gasoline increases by 25 % and a driver intends to spend only 20 % more on gasoline , by how much percent should the driver reduce the quantity of gasoline that he buys ? n0 = 25.0 n1 = 20.0"
    # solution = generate_solution(model, tokenizer, problem)
    # print(f"Problem: {problem}")
    # print(f"Generated Solution:\n{solution}\n")
    
    with open(file_path, "r") as file:
        problems = [json.loads(line) for line in file]
    #print(problems)
    # Batch inference example
    # problems = [
    #     "# the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ? n0 = 3.0 n1 = 10.0 n2 = 36.0"
    # ]
    problems = problems[:100]
    solutions = batch_generate_solutions(model, tokenizer, problems)
    #print("HERE")
    #print(solutions.shape)
    for count, data_point in enumerate(zip(problems, solutions)):
        q, solution = data_point
        file_index = count
        output_file = os.path.join(output_folder,f"generation_{file_index}.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            # print(f"Problem: {prob}")
            #print(f"Generated Solution:\n{sol}\n")
            f.write(f"\n{solution}\n")
            f.write("-" * 80 + "\n\n")
        print(f"Generated solution {file_index} saved to {output_file}")