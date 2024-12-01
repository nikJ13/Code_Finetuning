# import torch
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from transformers import (
#     AutoModelForCausalLM, 
#     AutoTokenizer, 
#     TrainingArguments,
#     BitsAndBytesConfig
# )
# from trl import SFTTrainer
# import json
# import os


# os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"  # Replace with your actual token

# file_path = "/home/ubuntu/codegen/prompts/finetune_prompts.jsonl"

# def prepare_data(problems_and_solutions):
#     """
#     Prepare data in the format needed for fine-tuning.
    
#     Args:
#         problems_and_solutions: List of dictionaries containing problem text, code, and answer
#     """
#     formatted_data = []
    
#     for item in problems_and_solutions:
# #         formatted_text = f"""<|system|>{item['instruction']}<|eot_id|>
# # <|start_header_id|>user<|end_header_id|>

# # Solve this math problem: {item['input']}<|eot_id|>

# # <|start_header_id|>assistant<|end_header_id|>Certainly! I'll solve the problem and provide a Python code solution.

# # Python Solution:

# # ```python
# # {item['python_code']}
# # ```

# # Answer: {item['answer']}<|eot_id|>"""

#         formatted_text = f"""{item['prompt']}"""
#         formatted_data.append({
#             "text": formatted_text
#         })
    
#     dataset = Dataset.from_list(formatted_data)
#     return dataset

# def setup_qlora_model(model_name="Qwen/Qwen2.5-Coder-7B-Instruct"):
#     """
#     Setup the model with QLoRA configuration using 4-bit quantization.
#     """
#     # Configure 4-bit quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",  # Normal Float 4
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True  # Double quantization
#     )
    
#     # Load model with quantization config
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map="auto",
#         torch_dtype=torch.float16
#     )
    
#     # Load tokenizer
#     # tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # tokenizer.pad_token = tokenizer.eos_token

#     # special_tokens = {
#     # "additional_special_tokens": [
#     #     "<|system|>", "<|user|>", "<|assistant|>", "<|ipython|>",
#     #     "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
#     #     "<|image|>", "<|begin_of_text|>"
#     # ]
#     # }

#     # tokenizer.add_special_tokens(special_tokens)
#     # model.resize_token_embeddings(len(tokenizer))
#     model_for_tokenizer = "Qwen/Qwen2.5-Coder-7B-Instruct"

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_for_tokenizer,
#         trust_remote_code=True,
#         padding_side="right"
#     )

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     model.gradient_checkpointing_enable()
#     # Prepare model for training
#     model = prepare_model_for_kbit_training(model)
    
#     # Define QLoRA Config
#     qlora_config = LoraConfig(
#         r=64,  # Higher rank for better performance
#         lora_alpha=16,
#         target_modules=[
#             "q_proj",
#             "k_proj",
#             "v_proj",
#             "o_proj",
#             "gate_proj",
#             "up_proj",
#             "down_proj",
#         ],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
    
#     # Get PEFT model
#     model = get_peft_model(model, qlora_config)
#     model.print_trainable_parameters()
#     return model, tokenizer

# def train_model(model, tokenizer, dataset, output_dir="math_solver_qlora_qwen"):
#     """
#     Train the model using the prepared dataset with QLoRA-specific optimizations.
#     """
#     # Define training arguments optimized for QLoRA
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=3,
#         per_device_train_batch_size=2,  # Smaller batch size due to 4-bit quantization
#         gradient_accumulation_steps=8,   # Increased for effective batch size
#         warmup_ratio=0.1,
#         learning_rate=5e-4,
#         fp16=True,
#         logging_steps=10,
#         save_strategy="epoch",
#         optim="paged_adamw_32bit",      # Memory-efficient optimizer
#         gradient_checkpointing=True,
#         max_grad_norm=0.3,              # Gradient clipping
#     )
    
#     # Initialize trainer with QLoRA optimizations
#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=dataset,
#         tokenizer=tokenizer,
#         args=training_args,
#         max_seq_length=512,
#         packing=False,  # Disable packing for more stable training
#     )
    
#     # Train the model
#     trainer.train()
    
#     # Save the trained adapter
#     trainer.save_model()

# # Example usage
# if __name__ == "__main__":
#     # Example data format
#     # sample_data = [
#     #     {
#     #         "problem": "If John has 5 apples and gives 2 to Mary, how many apples does he have left?",
#     #         "code": """def solve_apple_problem():
#     # initial_apples = 5
#     # given_away = 2
#     # remaining = initial_apples - given_away
#     # return remaining""",
#     #         "answer": "3 apples"
#     #     }
#     #     # Add more examples...
#     # ]
    
#     with open(file_path, "r") as file:
#         data = [json.loads(line) for line in file]

#     # Prepare dataset
#     dataset = prepare_data(data)
#     #print(dataset[0])
#     # Setup QLoRA model
#     model, tokenizer = setup_qlora_model()
    
#     # Train model
#     train_model(model, tokenizer, dataset)
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
import json
import os
import wandb  # Add wandb import
# import flash_attn

# Initialize wandb
wandb.init(
    project="qwen-qlora-training",  # Replace with your project name
    config={
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "learning_rate": 5e-4,
        "epochs": 3,
        "batch_size": 2
        # "flash_attention": True
    }
)

os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

file_path = "/home/ubuntu/codegen/prompts/Qwen/Qwen2.5-Coder-7B-Instruct/finetune_prompts.jsonl"

def prepare_data(problems_and_solutions):
    """
    Prepare data in the format needed for fine-tuning.
    """
    formatted_data = []
    
    for item in problems_and_solutions:
        formatted_text = f"""{item['prompt']}"""
        formatted_data.append({
            "text": formatted_text
        })
    
    dataset = Dataset.from_list(formatted_data)
    return dataset

def setup_qlora_model(model_name="Qwen/Qwen2.5-Coder-7B-Instruct"):
    """
    Setup the model with QLoRA configuration using 4-bit quantization.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        # use_flash_attention_2=True,  # Enable Flash Attention 2
        # attn_implementation="flash_attention_2"
    )
    
    model_for_tokenizer = "Qwen/Qwen2.5-Coder-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_for_tokenizer,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    qlora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, qlora_config)
    model.print_trainable_parameters()
    return model, tokenizer

class WandbCallback:
    """Custom callback to log metrics to wandb"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)

def train_model(model, tokenizer, dataset, output_dir="math_solver_qlora_qwen"):
    """
    Train the model using the prepared dataset with QLoRA-specific optimizations.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        learning_rate=5e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to=["wandb"],  # Enable wandb reporting
        # torch_compile=True,  # Enable torch.compile
        # max_sequence_length=2048,  # Adjust based on your needs
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
        packing=False,
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained adapter
    trainer.save_model()
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    # Load and prepare data
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]

    dataset = prepare_data(data)
    model, tokenizer = setup_qlora_model()
    train_model(model, tokenizer, dataset)