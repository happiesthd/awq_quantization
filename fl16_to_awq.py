from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model_path = "CPM_EXP_13.1_merged_fp16" 
quant_path = "CPM_EXP_13.1_merged_awq"

quant_config = {
    "zero_point": True,      # Use asymmetric quantization
    "q_group_size": 128,     # Group size for quantization
    "w_bit": 4,              # 4-bit quantization
    "version": "GEMM",       # Optimized kernel for matrix multiplication
}

print(f"Loading model for quantization from: {model_path}")
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map="auto",           # Distribute across available GPUs
    safetensors=True,            # It's good practice to use safetensors
    low_cpu_mem_usage=True,
    use_cache=False              # Prevent memory leak during quantization
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Starting AWQ quantization...")
model.quantize(tokenizer, quant_config=quant_config)

print(f"Saving quantized model to: {quant_path}")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"✅ Model quantized and saved at: {quant_path}")
