# merging the model adapters to fl16 model (full precision so that awq quantization can be done)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration ---
base_model_path = "google/gemma-7b-it"
lora_adapter_path = "../CPM_EXP_13.1/CPM_EXP_13.1" 
merged_model_path = "CPM_EXP_13.1_merged_fp16"      
hf_token = "hf_Your_hf_token_here"            

os.makedirs(merged_model_path, exist_ok=True)

# --- Load Base Model in FP16 ---
print("Loading base model in float16...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token  
    
# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    token=hf_token  
    
# --- Load and Merge LoRA Adapters ---
print(f"Loading LoRA adapters from {lora_adapter_path}...")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
print("Merging LoRA adapters into the base model...")
model = model.merge_and_unload()

# --- Save the Merged FP16 Model ---
print(f"Saving merged float16 model to {merged_model_path}...")
model.save_pretrained(merged_model_path, safe_serialization=True)
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model successfully saved to: {merged_model_path}")
