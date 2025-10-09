import streamlit as st
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login 
import traceback
import time
import os

# === Force vLLM to use a compatible attention backend ===
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

HF_TOKEN = "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"

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("streamlit_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Load vLLM AWQ Model and Tokenizer ===
@st.cache_resource
def load_vllm_resources():
    try:
        logger.info("Loading vLLM model and tokenizer...")

        main_model_id = "CPM_EXP_13.1_merged_awq"
        draft_model_id = "google/gemma-2b-it"

        # --- Login to Hugging Face ---
        try:
            login(token=HF_TOKEN)  
            logger.info("Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            logger.error("Hugging Face login failed.")
            logger.error(traceback.format_exc())
            st.error("Hugging Face authentication failed. Please check your token.")
            raise e

        # --- Load the AWQ quantized model with vLLM ---
        llm = LLM(
            model=main_model_id,
            quantization="awq",              
            tensor_parallel_size=1,
            dtype='float16',
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enforce_eager=True,
            # --- Model-based speculative decoding ---
            speculative_config={
                "model": draft_model_id,
                "draft_tensor_parallel_size": 1,
            },
        )

        # --- Load tokenizer from the main model ---
        tokenizer = AutoTokenizer.from_pretrained(main_model_id, token=HF_TOKEN)

        logger.info("vLLM model and tokenizer loaded successfully.")
        return llm, tokenizer

    except Exception as e:
        logger.error("Error loading vLLM resources:")
        logger.error(traceback.format_exc())
        st.error(f"Failed to load the model. Please check the logs. Error: {e}")
        raise e


# === Initialize resources ===
llm, tokenizer = load_vllm_resources()


# === Response Generation ===
def get_vllm_response(prompt, llm, tokenizer):
    try:
        logger.info("Applying chat template to the prompt...")

        system_prompt = (
            "Extract list of requirements from given citation text. "
            "Give the numbered list of requirements with title of each requirement "
            "followed by requirement text. Avoid emojis, informal language, "
            "generic responses, repetitions or assumptions."
        )

        messages = [{"role": "user", "content": system_prompt + prompt}]

        prompt_formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,   # mild repetition control
            frequency_penalty=0.6,    # penalize frequent tokens
            presence_penalty=0.4,     # encourage some diversity
        )

        logger.info("Generating response with vLLM...")
        start_time = time.time()

        outputs = llm.generate(prompts=[prompt_formatted], sampling_params=sampling_params)

        elapsed = time.time() - start_time
        logger.info(f"Response generation completed in {elapsed:.2f} seconds.")

        response = outputs[0].outputs[0].text.strip()
        return response

    except Exception as e:
        logger.error("Error during vLLM response generation:")
        logger.error(traceback.format_exc())
        return "An error occurred while generating a response."


# === Streamlit UI ===
st.set_page_config(page_title="Requirement Extractor", layout="wide")
st.title("📋 Requirement Extractor (vLLM + AWQ + Speculative Decoding)")

st.markdown("""
This tool extracts structured requirements from technical citation text.
It uses an **AWQ-quantized fine-tuned Gemma model** with **speculative decoding (Gemma-2B-IT)** 
for faster, high-quality results.
""")

user_input = st.text_area("Enter citation text below:", height=200, key="user_input")

if st.button("Generate Requirements"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        logger.info(f"Received input: {user_input[:100]}...")
        with st.spinner("Extracting requirements... please wait ⏳"):
            response = get_vllm_response(user_input, llm, tokenizer)
            logger.info("Response successfully generated.")
            st.success("Extracted Requirements:")
            st.markdown(response)