This subfolder contains streamlit app to run inference on AWQ quantized model with following inference mechanisms.
**NOTE: vLLM is used for fast inference speed. vLLM version= 0.10.2. 
Beam serech is very much version specific in terms of syntax in vllm, so ensure the same version inorder to run these files.**

awq_streamlit.py: Uses simple top-p sampling with vLLM.


awq_streamlit_beam.py: Uses bulitin beam search instances from vllm.beam_search 


awq_streamlit_beam_manual.py: beam search instance, heuristic score-cumulative score(completeness etc.,  diversity = unique_tokens / total_tokens) to re-rank multiple outputs


awq_streamlit_speculative.py: Uses speculative decoding with n gram


awq_streamlit_speculative_draft.py: Uses speculative decoding with draft model (small model compared to base mdoel for fast inference)


**NOTE: vLLM speculative decoding is not tested yet.
Speculative decoding being a sampling method, beam search can't be used with it.**
