import streamlit as st
import logging
from vllm import LLM
from vllm.beam_search import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
    create_sort_beams_key_function,
    get_beam_search_score
)
from transformers import AutoTokenizer
import traceback
import time
import os

# === Force vLLM to use a compatible attention backend ===
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

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
        
        model_id = "CPM_EXP_13.1_merged_awq"

        llm = LLM(
            model=model_id,
            quantization="awq",
            tensor_parallel_size=1,
            dtype='float16',
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enforce_eager=True    
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info("vLLM model and tokenizer loaded successfully.")
        return llm, tokenizer

    except Exception as e:
        logger.error("Error loading vLLM resources:")
        logger.error(traceback.format_exc())
        st.error(f"Failed to load the model. Please check the logs. Error: {e}")
        raise e

llm, tokenizer = load_vllm_resources()

# === Response Generation with Beam Search ===
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
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode prompt into tokens
        prompt_tokens = tokenizer(prompt_formatted)["input_ids"]

        # Initialize Beam Search Instance
        beam_instance = BeamSearchInstance(prompt_tokens=prompt_tokens)

        logger.info("Generating with beam search...")
        start_time = time.time()

        # Generate with vLLM 
        outputs = llm.generate(prompts=[prompt_formatted], sampling_params=None)
        generated_tokens = outputs[0].outputs[0].token_ids

        # Wrap into BeamSearchSequence
        score = get_beam_search_score(
            tokens=generated_tokens,
            cumulative_logprob=0.0,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=0.9
        )
        beam_seq = BeamSearchSequence(
            tokens=generated_tokens,
            logprobs=[],
            cum_logprob=score,
            text=tokenizer.decode(generated_tokens)
        )

        # Create output and rank beams
        beam_output = BeamSearchOutput(sequences=[beam_seq])
        key_fn = create_sort_beams_key_function(
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=0.9
        )
        sorted_beams = sorted(beam_output.sequences, key=key_fn, reverse=True)

        elapsed = time.time() - start_time
        logger.info(f"Beam search completed in {elapsed:.2f} seconds.")

        response = sorted_beams[0].text
        return response

    except Exception as e:
        logger.error("Error during vLLM response generation:")
        logger.error(traceback.format_exc())
        return "An error occurred while generating a response."

# === Streamlit UI ===
st.set_page_config(page_title="Requirement Extractor", layout="wide")
st.title("📋 Requirement Extractor (vLLM + AWQ + Beam Search)")

user_input = st.text_area("Enter citation text below:", height=200, key="user_input")

if st.button("Generate Requirements"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        logger.info(f"Received input: {user_input[:100]}...") 
        with st.spinner("Extracting requirements..."):
            response = get_vllm_response(user_input, llm, tokenizer)
            logger.info("Response successfully generated.")
            st.success("Extracted Requirements:")
            st.markdown(response)
