import json
import faiss
import random
import math
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from adversarial_decoding.strategies.emb_inv_decoding import EmbInvDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device
import torch
import sys

# Import components for reconstruction (GEIA)
from transformers import AutoTokenizer, AutoModelForCausalLM
from GEIA.reconstruction import load_projector, load_attacker, reconstruct

long_passages = False
add_noise = False

class NoisyEncoder(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        self.noise_level = kwargs.get('noise_level', 0)
        super().__init__(*args, **kwargs)

    def encode(self, text, add_special_tokens=True, **kwargs):
        embs = super().encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return embs + torch.randn_like(embs) * self.noise_level
    
def emb_inv_experiment(
    should_natural=False,
    encoder_name='gte',
    beam_width=5,
    max_steps=20,
    top_k=50,
    top_p=1,
    num_samples=None
):
    """
    Run an experiment to test retrieval-augmented generation.
    
    Args:
        should_natural (bool): Whether to enforce naturalness in the outputs
        beam_width (int): Width of the beam for beam search
        max_steps (int): Maximum number of steps for beam search
        top_k (int): Top-k parameter for sampling
        top_p (float): Top-p (nucleus sampling) parameter
    """
    print("Running Embedding Inversion experiment...")
    
    # Set up trigger and control text
    trigger = "tesla"
    control_text = "I hate {trig}."
    repetition_penalty = 1.5

    if encoder_name == 'gte':
        encoder = NoisyEncoder("../gte-base-local", device=file_device)
    elif encoder_name == 'gte-Qwen':
        encoder = NoisyEncoder("Alibaba-NLP/gte-Qwen2-1.5B-instruct", device=file_device, trust_remote_code=True)
    elif encoder_name == 'gtr':
        encoder = NoisyEncoder("sentence-transformers/gtr-t5-base", device=file_device)
    elif encoder_name == 'contriever':
        encoder = NoisyEncoder("facebook/contriever", device=file_device)
    
    # Initialize decoding strategy
    attack = EmbInvDecoding(
        encoder=encoder, 
        device=file_device,
        should_natural=should_natural,
        repetition_penalty=repetition_penalty
    )

    # Initialize GEIA components for reconstruction-guided initialization
    use_geia_init = True
    if use_geia_init:
        print("Initializing GEIA components for reconstruction...")
        # Paths - adjust these relative paths as necessary
        attacker_path = 'GEIA/qwen-LoRa-gte-msmarco/attacker_qwen25_3b_ms_marco_gte_base'
        tokenizer_path = '../Qwen2___5-3B'
        base_model_path = '../Qwen2___5-3B' 
        projector_path = 'GEIA/qwen-LoRa-gte-msmarco/projection_qwen25_3b_ms_marco_gte_base'
        
        # Load components
        geia_attacker = load_attacker(attacker_path, base_model_path=base_model_path)
        geia_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # Determine dimensions
        attacker_hidden_size = geia_attacker.config.hidden_size
        enc_dim = 768 # Default for GTE-base
        
        geia_projector = load_projector(projector_path, enc_dim, attacker_hidden_size)
    
    # Sample test prompts
    
    from datasets import load_dataset, load_from_disk

    if long_passages:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
        docs = ds['train'].shuffle(seed=42, buffer_size=1000).take(100)
        filtered_docs = []
        for doc in docs:
            doc_len = len(encoder.tokenizer.encode(doc['text']))
            if doc_len >= 512:
                filtered_docs.append(doc['text'])
        target_docs = filtered_docs[:10]
        print("filtered docs")
        print(len(filtered_docs))
    else:
        # marco_ds = load_dataset("microsoft/ms_marco", "v2.1")
        # Load from local disk (assumed saved via save_to_disk)
        marco_ds = load_from_disk("../ms_marco_v2.1_local")
        # Since it's loaded from disk, we might not need to shuffle and select if it was already processed, 
        # but to keep logic consistent with original code (which did shuffle and select on the full dataset):
        if 'train' in marco_ds:
             marco_ds = marco_ds['train']
        
        # If the loaded dataset is already a subset or fully processed, the shuffle/select might be redundant or error prone if indices are out of range.
        # However, assuming original flow:
        if len(marco_ds) > 1000:
             marco_ds = marco_ds.shuffle(seed=42).select(range(1000))
        random.seed(42)
        target_docs = [random.choice(doc['passages']['passage_text']) for doc in marco_ds]
    
    adv_texts = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = f'./data/emb_inv_attack_{"natural" if should_natural else "unnatural"}_{encoder_name}_{"long" if long_passages else "short"}_{"no_noise" if not add_noise else "noise"}_{timestamp}.json'

    if not long_passages:
        max_len_arr = [max_steps]
    else:
        max_len_arr = [16, 32, 64, 128, 256, 512]

    if add_noise:
        noise_levels = [0.001, 0.01, 0.1]
    else:
        noise_levels = [0]

    samples_per_target = len(max_len_arr) * len(noise_levels)
    if num_samples is not None and num_samples > 0:
        target_limit = max(1, math.ceil(num_samples / samples_per_target))
    else:
        target_limit = 5

    print(f"Target docs used: {target_limit}")
    print(f"Samples per target doc: {samples_per_target}")
    if num_samples is not None and num_samples > 0:
        print(f"Requested total samples: {num_samples}")
    else:
        print(f"Requested total samples: default ({target_limit * samples_per_target})")
    
    generated_count = 0
    # Run decoding for each prompt
    for full_target in target_docs[:target_limit]:
        if num_samples is not None and num_samples > 0 and generated_count >= num_samples:
            break
        for max_len in max_len_arr:
            if num_samples is not None and num_samples > 0 and generated_count >= num_samples:
                break
            for noise_level in noise_levels:
                if num_samples is not None and num_samples > 0 and generated_count >= num_samples:
                    break
                attack.encoder.noise_level = noise_level
                print("-" * 50)
                
                # --- Original Hard Truncation ---
                # target = encoder.tokenizer.decode(encoder.tokenizer.encode(full_target, add_special_tokens=False)[:max_len])
                
                # --- New Logic: Use full text or smart truncation ---
                # Option 1: No truncation (use full_target directly)
                # target = full_target
                
                # Option 2: Smart truncation (try to end on a sentence boundary within limit)
                tokens = encoder.tokenizer.encode(full_target, add_special_tokens=False)
                if len(tokens) <= max_len:
                    target = full_target
                else:
                    # Truncate to max_len first
                    truncated_tokens = tokens[:max_len]
                    partial_text = encoder.tokenizer.decode(truncated_tokens)
                    # Try to find the last sentence ending punctuation
                    import re
                    match = re.search(r'(.*[.!?])', partial_text)
                    if match:
                        target = match.group(1)
                    else:
                        # Fallback to hard truncation if no punctuation found
                        target = partial_text

                print(f"Processing prompt: {target}")
                
                # Set up combined scorer
                
                # Run decoding
                prompt = 'tell me a story'
                
                init_text = None
                if use_geia_init and geia_attacker and geia_projector:
                    try:
                        print("Running GEIA reconstruction for initialization...")
                        # Reconstruct text using GEIA (encoder is passed from current experiment)
                        # Note: reconstruction uses beam search internally
                        reconstructed = reconstruct(
                            text=target, 
                            encoder_model=encoder, 
                            encoder_tokenizer=encoder.tokenizer, 
                            projector=geia_projector, 
                            attacker_model=geia_attacker, 
                            attacker_tokenizer=geia_tokenizer
                        )
                        if isinstance(reconstructed, list):
                            reconstructed = reconstructed[0]
                        
                        init_text = reconstructed
                        print(f"GEIA Initialized Text: {init_text}")
                    except Exception as e:
                        print(f"GEIA reconstruction failed: {e}")
                        init_text = None

                
                for idx in range(1):
                    if long_passages:
                        if idx == 0:
                            current_top_k = top_k
                            current_max_len = 32
                        else:
                            current_top_k = 10
                            current_max_len = max_len
                    else:
                        current_top_k = top_k
                        current_max_len = max_len
                    best_cand = attack.run_decoding(
                        prompt=prompt,
                        target=target,
                        beam_width=beam_width,
                        max_steps=current_max_len,
                        top_k=current_top_k,
                        top_p=top_p,
                        should_full_sent=False,
                        verbose=False,
                        randomness=True,
                        init_text=init_text
                    )
                    print(best_cand.token_ids)
                    print([best_cand.seq_str], 'cos_sim:', best_cand.cos_sim, 'bleu_score:', sentence_bleu([target], best_cand.seq_str))
                    prompt = f"write a sentence similar to this: {best_cand.seq_str}"
                
                # Save results
                result_str = best_cand.seq_str
                adv_texts.append(result_str)
                print(f"Result: {result_str}")
                attack.llm_wrapper.template_example()
                bleu_score = sentence_bleu([target], result_str)
                
                append_to_target_dir(target_dir, {
                    'target': target,
                    'generation': result_str,
                    'cos_sim': best_cand.cos_sim or 0.0,
                    'naturalness': best_cand.naturalness or 0.0,
                    'bleu_score': bleu_score,
                    'beam_width': beam_width,
                    'max_steps': max_len,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repetition_penalty': repetition_penalty,
                    'encoder_name': encoder_name,
                    'noise_level': noise_level
                })
                generated_count += 1
                print(f"Generated samples: {generated_count}")

    print(f"Saved {generated_count} samples to {target_dir}")
