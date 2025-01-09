import os
import spacy
import pandas as pd
import torch
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import keyboard  # To detect escape key press

# --------------------------------------------------------------- MODEL DEFINITION --------------------------------------------------------------------------------------
# --- Configure Environment for Windows Symlink Compatibility ---
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlink warnings for Windows

# --- Hugging Face Authentication ---
login("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")  # Replace with your Hugging Face token

# --- Setup NLP and LLaMA Model ---
nlp = spacy.load("en_core_web_sm")  # Load spaCy for text preprocessing
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token="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")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token="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").to(device)

# Set up the text generation pipeline with LLaMA model
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, device=0 if device == torch.device("cuda") else -1)

# --------------------------------------------------------------- KNOWLEDGE BASE FROM EXCEL --------------------------------------------------------------------------------------
# --- Load Knowledge Base from Excel (.xlsx) ---
xlsx_path = "1.0 Datasets/1.1.0_ENG_Pidriš WEB description_small chunks.xlsx"  # Replace with your actual Excel file path
# Read Excel file
df = pd.read_excel(xlsx_path)
# Assuming the first column of the Excel file contains the text chunks (documents)
documents = df.iloc[:, 0].tolist()

# ------------------------------------------------------------- KNOWLEDGE BASE EMBEDDINGS AND ADDING TO FAISS --------------------------------------------------------------------------------------
# --- Setup Sentence Transformer and FAISS Index ---
model_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Load SentenceTransformer for embeddings
document_embeddings = model_encoder.encode(documents)  # Generate embeddings for each document

# Initialize FAISS index for fast document retrieval
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# --------------------------------------------------------------- FUNCTIONS DEFINITIONS --------------------------------------------------------------------------------------

# Step 1: Preprocess text for uniformity (for both query and documents)
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Step 2: Define a function to retrieve the most relevant document based on the user query
def retrieve_document(query):
    processed_query = preprocess_text(query)
    query_embedding = model_encoder.encode([processed_query])

    # Search in FAISS index for the closest document
    k = 3  # Retrieve top 3 most relevant documents
    distances, indices = index.search(query_embedding, k)

    # Retrieve the document text using the index
    retrieved_doc_index = indices[0][0]
    retrieved_doc_text = documents[retrieved_doc_index]
    return retrieved_doc_index, retrieved_doc_text  # Return the document index and text

# Step 3: Define function to generate response based on retrieved document
def generate_response(query, use_conversation_history=True):
    # Retrieve the relevant document for context
    retrieved_doc_index, retrieved_doc_text = retrieve_document(query)
    print(f"Retrieved Document Index: {retrieved_doc_index}")
    print(f"Document Preview: {retrieved_doc_text[:100]}...")  # Print the first 100 characters of the document for preview

    # Build context for model
    contextual_input = f"User query: {query}\nDocument context: {retrieved_doc_text}\nAnswer:"

    # Generate response using LLaMA model
    response = generator(
        contextual_input,

        ## Parameters that control the length of the output
        # max_length = 120,  # (int, optional, defaults to 20) — The maximum length the generated tokens can have.
        max_new_tokens = 512,  # (int, optional) — The maximum number of tokens to generate, ignoring the number of tokens in the prompt.
        # min_length = 20,  # (int, optional, defaults to 0) — The minimum length of the sequence to be generated.
        min_new_tokens = 10,  # (int, optional) — The minimum number of tokens to generate, ignoring the number of tokens in the prompt.
        # early_stopping = False,  # (bool or str, optional, defaults to False) — Stops generation earlier if all beams meet a stopping condition (e.g., generating an end-of-sequence token) before reaching the maximum token length.
        # max_time = 5.0,  # (float, optional) — The maximum amount of time (in seconds) allowed for computation.
        # stop_strings = ["\n"],  # (str or List[str], optional) — A string or list of strings that should terminate generation if the model outputs them.

        ## Parameters that control the generation strategy used
        do_sample = True,  # (bool, optional, defaults to False) — do_sample = True gives you variety and creativity, like randomly choosing a restaurant each time.
                                                                 # do_sample = False gives you predictability and reliability, like always going to the most popular restaurant based on recommendations.
        # num_beams = 4,  # (int, optional, defaults to 1) — num_beams = 1 means greedy decoding, where the model always picks the most likely next token at each step (like taking the safest path).
                                                         # num_beams > 1 means the model will explore multiple possible sequences in parallel and choose the best one, which can lead to more fluent and logical text.
        # num_beam_groups = 2,  # (int, optional, defaults to 1) — Number of groups to divide num_beams into for diversity. Imagine you're at a brainstorming session with a team. You have a total of 4 ideas (beams) that you want to explore, but you divide them into 2 groups, with each group coming up with different potential solutions. Each group works independently on their set of ideas, which can help you discover more creative and diverse solutions rather than having all the ideas come from one group.
        # penalty_alpha = 0.8,  # (float, optional) — When penalty_alpha is high, you're applying stronger self-discipline, meaning you'll catch those repeated phrases more quickly and force yourself to avoid them.
                                                  # When penalty_alpha is low, you're less strict and might let those repetitions slide through, resulting in more redundant or repetitive content.The value balancing the model confidence and the degeneration penalty in contrastive search decoding.
        dola_layers = "high",  # (str or List[int], optional) — The layers to use for DoLa decoding. The parameter can be set to "high" (for abstract, higher-level reasoning) or "low" (for syntax and simpler generation), or specific layers can be selected manually.

        # # Parameters that control the cache
        # use_cache = True,  # (bool, optional, defaults to True) — Whether the model should use past key/values attentions to speed up decoding.
        # cache_implementation = "static",  # (str, optional, defaults to None) — The cache class name for faster decoding.
        # cache_config = None,  # (CacheConfig or dict, optional, default to None) — Arguments for the cache class can be passed here.
        # return_legacy_cache = True,  # (bool, optional, default to True) — Whether to return the legacy format of the cache when used.

        # # Parameters for manipulation of the model output logits
        temperature = 0.7,  # (float, optional, defaults to 1.0) — Modulates the next token probabilities.
        top_k = 50,  # (int, optional, defaults to 50) — The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p = 0.90,  # (float, optional, defaults to 1.0) — Keeps the smallest set of tokens with probabilities that add up to top_p or higher.
        # min_p = 0.05,  # (float, optional) — Minimum token probability, scaled by the probability of the most likely token.
        # typical_p = 0.9,  # (float, optional, defaults to 1.0) — If you always pick the most typical or expected word or phrase, your story will follow well-established patterns. For example, if you're writing a romance story, you might default to the common phrase “They kissed under the moonlight.”
        # epsilon_cutoff = 0.001,  # (float, optional, defaults to 0.0) — If you're generating a formal email and you set a high epsilon_cutoff (e.g., 0.05), you would only allow tokens that are highly likely and formal in tone. This would help in ensuring that the generated text stays relevant and appropriate for the email context.
        # eta_cutoff = 0.001,  # (float, optional, defaults to 0.0) — Hybrid of locally typical sampling and epsilon sampling.
        # diversity_penalty = 0.5,  # (float, optional, defaults to 0.0) — Penalizes beams if they generate similar tokens to other beams.
        # repetition_penalty = 1.2,  # (float, optional, defaults to 1.0) — The parameter for repetition penalty. 1.0 means no penalty.
        # encoder_repetition_penalty = 1.0,  # (float, optional, defaults to 1.0) — This parameter applies an exponential penalty to any repeated token sequences generated by the model. The purpose is to discourage repetition of the same phrases, words, or sequences within the generated text. This is especially important in encoder-decoder models, where there is an interaction between the encoder (input processing) and decoder (output generation).
        # length_penalty = 1.0,  # (float, optional, defaults to 1.0) — Exponential penalty to length used in beam-based generation.
        # no_repeat_ngram_size = 2,  # (int, optional, defaults to 0) — Ensures no ngrams of the specified size repeat.
        # bad_words_ids = [[1, 2]],  # (List[List[int]], optional) — List of token ids that are not allowed to be generated.
        # force_words_ids = [[123, 456]],  # (List[List[int]] or List[List[List[int]]], optional) — Token ids that must be generated.
        renormalize_logits = True,  # (bool, optional, defaults to False) — This is useful when you're applying techniques like top-k sampling, top-p sampling, penalty adjustments, or constraints, ensuring the model continues generating valid, coherent output.
        # constraints = [],  # (List[Constraint], optional) — Custom constraints for the generation process.
        # forced_bos_token_id = 50256,  # (int, optional) — ID of the token to force as the first token after the decoder start.
        # forced_eos_token_id = 50257,  # (Union[int, List[int]], optional) — ID(s) of the token to force as the last generated token.
        # remove_invalid_values = False,  # (bool, optional, defaults to True) — Whether to remove nan and inf outputs of the model.
        # exponential_decay_length_penalty = (10, 0.8),  # (tuple(int, float), optional) — Exponentially increasing length penalty after certain tokens.
        # suppress_tokens = [3, 4],  # (List[int], optional) — List of tokens that will be suppressed during generation.
        # begin_suppress_tokens = [5],  # (List[int], optional) — Tokens that are suppressed at the beginning of the generation.
        # forced_decoder_ids = [[1, 123]],  # (List[List[int]], optional) — Mapping from generation indices to forced token indices.
        # sequence_bias = {(1, 2): 0.5},  # (Dict[Tuple[int], float], optional) — Bias term applied to specific token sequences.
        # token_healing = True,  # (bool, optional, defaults to False) — Heal tail tokens of prompts by replacing them.
        guidance_scale = 7.5,  # (float, optional) — The guidance scale for classifier-free guidance (CFG).
        low_memory = True,  # (bool, optional) — Switches to sequential beam search and sequential top-k for contrastive search.
        # watermarking_config = {},  # (BaseWatermarkingConfig or dict, optional) — Arguments used to watermark model outputs.
        # truncation=True,

        # # Parameters that define the output variables of generate
        num_return_sequences = 1,  # (int, optional, defaults to 1) — The number of independently computed returned sequences.
        # output_attentions = False,  # (bool, optional, defaults to False) — Whether to return the attention tensors of all layers.
        # output_hidden_states = False,  # (bool, optional, defaults to False) — Whether to return the hidden states of all layers.
        # output_scores = False,  # (bool, optional, defaults to False) — Whether to return the prediction scores.
        # output_logits = False,  # (bool, optional) — Whether to return the unprocessed prediction logit scores.
        # return_dict_in_generate = False,  # (bool, optional, defaults to False) — Whether to return a ModelOutput, not just generated text.
        #
        # # Special tokens that can be used at generation time
        # pad_token_id = 0,  # (int, optional) — The id of the padding token.
        # bos_token_id = 50256,  # (int, optional) — The id of the beginning-of-sequence token.
        # eos_token_id = 50257,  # (Union[int, List[int]], optional) — The id(s) of the end-of-sequence token(s).
        #
        # # Generation parameters exclusive to encoder-decoder models
        # encoder_no_repeat_ngram_size = 2,  # (int, optional, defaults to 0) — No ngrams of that size can repeat in encoder-decoder models.
        # decoder_start_token_id = 50258,  # (int or List[int], optional) — ID(s) of the token to start decoding.
        #
        # # Generation parameters exclusive to assistant generation
        # is_assistant = True,  # (bool, optional, defaults to False) — Whether the model is an assistant model.
        # num_assistant_tokens = 25,  # (int, optional, defaults to 20) — Number of speculative tokens generated by the assistant.
        # num_assistant_tokens_schedule = "constant",  # (str, optional, defaults to "constant") — Schedule for changing max assistant tokens.
        # assistant_confidence_threshold = 0.4,  # (float, optional, defaults to 0.4) — Confidence threshold for the assistant model.
        # prompt_lookup_num_tokens = 20,  # (int, optional) — The number of tokens to output as candidate tokens.
        # max_matching_ngram_size = 2,  # (int, optional) — Maximum ngram size to be considered for matching in the prompt.
        # assistant_early_exit = 5,  # (int, optional) — Early exit of the model will be used as an assistant.
        # assistant_lookbehind = 10,  # (int, optional, defaults to 10) — Looks back at the last assistant tokens for alignment.
        # target_lookbehind = 10,  # (int, optional, defaults to 10) — Looks back at the last target tokens for alignment.
        #
        # # Parameters related to performances and compilation
        # compile_config = None,  # (CompileConfig, optional) — Controls how the forward pass is compiled for performance gains.
        #
        # # Wild card: Additional generation kwargs
        # generation_kwargs = {},  # Additional generation kwargs will be forwarded to the model.
 )

    # Extract the answer from the generated text
    generated_text = response[0]["generated_text"]
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:")[-1].strip()
    else:
        generated_text = generated_text.strip()

    return generated_text

# --------------------------------------------------------------- MAIN LOOP --------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Chatbot ready! Type your questions. Press 'Esc' to exit.")

    while True:
        if keyboard.is_pressed("esc"):
            print("\nExiting chatbot. Goodbye!")
            break

        print("-----------------------------------------------------------------")
        # user_query = input("You: ")
        user_query = "What are the best hiking routes to visit in Pidriš village?"
        print("User Query:", user_query)

        # Generate and print the response
        chatbot_response = generate_response(user_query)
        print("[FINAL CHATBOT ANSWER]:", chatbot_response)