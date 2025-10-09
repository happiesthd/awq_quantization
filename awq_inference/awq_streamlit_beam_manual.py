#beam search instance, heuristic score-cumulative score(completeness-not defined well,  diversity = unique_tokens / total_tokens) to re-rank multiple outputs
import streamlit as st
import logging
from vllm import LLM, SamplingParams
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
def get_vllm_response(prompt, llm, tokenizer, beam_width=4, max_tokens=2048):
    try:
        logger.info("Applying chat template to the prompt...")
        
        system_prompt = (
            "Extract list of requirements from given citation text. "
            "Give the numbered list of requirements with title of each requirement "
            "followed by requirement text. Avoid emojis, informal language, "
            "generic responses, repetitions or assumptions. "
            "Each requirement should be stated only once."
        )

        messages = [{"role": "user", "content": system_prompt + prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode prompt
        prompt_tokens = tokenizer(prompt_formatted)["input_ids"]

        # Initialize Beam Search
        beam_instance = BeamSearchInstance(prompt_tokens=prompt_tokens)

        logger.info("Generating with beam search...")
        start_time = time.time()

        # IMPROVED: Better sampling params to reduce repetition
        sampling_params = SamplingParams(
            max_tokens=3072,  # Increased to allow longer outputs
            min_tokens=200,   # Increased minimum to ensure completeness
            temperature=0.6,  # Slightly lower for more focused generation
            top_p=0.9,        # Balanced for quality and coverage
            top_k=40,         # Focused vocabulary
            n=beam_width,
            presence_penalty=0.5,   # Moderate penalty to avoid repetition
            frequency_penalty=0.2,  # Light penalty for frequency
            stop=None,  # Remove all stop sequences to let model complete
            skip_special_tokens=True,
            best_of=max(beam_width * 2, 4),  # Generate more candidates
            # use_beam_search=False
        )

        outputs = llm.generate(prompts=[prompt_formatted], sampling_params=sampling_params)
        candidates = outputs[0].outputs

        # Log all candidate statistics
        for i, cand in enumerate(candidates):
            req_count = len([line for line in cand.text.split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('Requirement'))])
            logger.info(f"Candidate {i}: {len(cand.token_ids)} tokens, {req_count} requirements detected")

        # Filter and score candidates
        scored_candidates = []
        for cand in candidates:
            text = cand.text
            
            # Calculate metrics
            has_repetition = has_excessive_repetition(text, threshold=0.6)  # Very lenient
            completeness = calculate_completeness_score(text)
            diversity = calculate_diversity_score(cand.token_ids)
            
            # Composite score: prioritize completeness over everything
            composite_score = (completeness * 10.0) + (diversity * 2.0) - (5.0 if has_repetition else 0.0)
            
            scored_candidates.append({
                'candidate': cand,
                'score': composite_score,
                'completeness': completeness,
                'has_repetition': has_repetition
            })
            
            logger.info(f"Candidate score: {composite_score:.2f} (completeness: {completeness:.2f}, repetition: {has_repetition})")
        
        # Sort by composite score and take top candidates
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates_to_use = [sc['candidate'] for sc in scored_candidates]
        
        logger.info(f"Selected top {len(candidates_to_use)} candidates based on completeness")

        # Wrap candidates into BeamSearchSequences
        sequences = []
        for cand in candidates_to_use:
            tokens = cand.token_ids
            
            # Get cumulative log probability safely
            if hasattr(cand, 'cumulative_logprob') and cand.cumulative_logprob is not None:
                cum_logprob = cand.cumulative_logprob
            else:
                # Fallback: use sum of logprobs if available
                if hasattr(cand, 'logprobs') and cand.logprobs:
                    cum_logprob = sum(cand.logprobs)
                else:
                    # Default to 0.0 if no logprobs available
                    cum_logprob = 0.0
            
            # Calculate diversity bonus (penalize repetitive patterns)
            diversity_score = calculate_diversity_score(tokens)
            
            # Calculate completeness score (reward longer outputs with more requirements)
            completeness_score = calculate_completeness_score(cand.text)
            
            score = get_beam_search_score(
                tokens=tokens,
                cumulative_logprob=cum_logprob,
                eos_token_id=tokenizer.eos_token_id,
                length_penalty=0.5  # Further reduced - strongly favor longer outputs
            )
            # Heavily boost score by completeness
            score += completeness_score * 5.0  # Increased weight significantly
            score += diversity_score * 0.5
            
            sequences.append(
                BeamSearchSequence(
                    tokens=tokens,
                    logprobs=[],
                    cum_logprob=score,
                    text=cand.text
                )
            )

        # Sort beams
        beam_output = BeamSearchOutput(sequences=sequences)
        key_fn = create_sort_beams_key_function(
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=0.8  # Match the scoring penalty
        )
        sorted_beams = sorted(beam_output.sequences, key=key_fn, reverse=True)

        elapsed = time.time() - start_time
        logger.info(f"Beam search completed in {elapsed:.2f} seconds.")

        # Log candidate lengths for debugging
        for i, beam in enumerate(sorted_beams[:3]):
            req_count = beam.text.count('\n') + 1
            logger.info(f"Beam {i}: {len(beam.tokens)} tokens, ~{req_count} requirements")

        # Take the best completed beam and post-process
        response = sorted_beams[0].text
        response = post_process_response(response)
        
        logger.info(f"Final response length: {len(response)} chars")
        
        return response

    except Exception as e:
        logger.error("Error during vLLM response generation:")
        logger.error(traceback.format_exc())
        return "An error occurred while generating a response."


def has_excessive_repetition(text, ngram_size=3, threshold=0.3):
    """
    Check if text has excessive repetition of n-grams.
    """
    words = text.split()
    if len(words) < ngram_size:
        return False
    
    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngram = ' '.join(words[i:i+ngram_size])
        ngrams.append(ngram)
    
    if not ngrams:
        return False
    
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return unique_ratio < (1 - threshold)


def calculate_diversity_score(tokens):
    """
    Calculate a diversity score based on unique token ratio.
    Higher score = more diverse output.
    """
    if len(tokens) < 10:
        return 0.0
    
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    diversity = unique_tokens / total_tokens
    
    return diversity


def calculate_completeness_score(text):
    """
    Calculate completeness score based on number of requirements detected.
    Rewards outputs with more numbered requirements.
    """
    import re
    
    # Count numbered requirements (e.g., "1.", "2.", etc.)
    numbered_pattern = r'^\s*\d+[\.)]\s+'
    requirement_count = len(re.findall(numbered_pattern, text, re.MULTILINE))
    
    # Normalize score (more requirements = higher score, cap at 20)
    completeness = min(requirement_count / 10.0, 1.0)
    
    return completeness


def post_process_response(text):
    """
    Post-process the response to remove trailing repetitions.
    More conservative to preserve complete requirements.
    """
    # Remove common repetitive endings
    lines = text.split('\n')
    
    # Remove duplicate consecutive lines (but preserve intentional duplicates)
    cleaned_lines = []
    prev_line = None
    consecutive_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        # Allow empty lines
        if not line_stripped:
            cleaned_lines.append(line)
            prev_line = None
            consecutive_count = 0
            continue
        
        # Check if this line is identical to previous
        if line_stripped == prev_line:
            consecutive_count += 1
            # Only skip if we've seen it more than twice consecutively
            if consecutive_count > 2:
                continue
        else:
            consecutive_count = 0
        
        cleaned_lines.append(line)
        prev_line = line_stripped
    
    return '\n'.join(cleaned_lines)


# === Streamlit UI ===
st.set_page_config(page_title="Requirement Extractor", layout="wide")
st.title("📋 Requirement Extractor (vLLM + AWQ + Beam Search)")

user_input = st.text_area("Enter citation text below:", height=200, key="user_input")

col1, col2 = st.columns([1, 4])
with col1:
    beam_width = st.slider("Beam Width", 2, 8, 4, help="More beams = more diverse outputs")

if st.button("Generate Requirements"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        logger.info(f"Received input: {user_input[:100]}...") 
        with st.spinner("Extracting requirements..."):
            response = get_vllm_response(user_input, llm, tokenizer, beam_width=beam_width, max_tokens=2048)
            logger.info("Response successfully generated.")
            st.success("Extracted Requirements:")
            st.markdown(response)