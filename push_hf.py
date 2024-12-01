from peft import AutoPeftModelForCausalLM

def push_peft_to_hub(
    adapter_path,
    repo_name,
    private=False
):
    """
    Push PEFT/LoRA model to Hub
    
    Args:
        adapter_path: Path to your adapter weights
        repo_name: Name for your model repository
        private: Whether to make the repository private
    """
    # Load the PEFT model
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto"
    )
    
    # Push the model
    model.push_to_hub(
        repo_id=repo_name,
        private=private
    )
    
    print(f"Model pushed successfully to: https://huggingface.co/{repo_name}")

# Example usage:
adapter_path = "/home/ubuntu/codegen/math_solver_qlora"
repo_name = "nikJ13/llama3.2_3B_code_finetuned"
push_peft_to_hub(adapter_path, repo_name, private=True)