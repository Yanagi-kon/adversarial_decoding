from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import TrainerCallback
import numpy as np
import evaluate
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
import json
import torch
import random
import os
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

use_lora = True
curve_save_path = "./results/training_curves.png"
metric_history_path = "./results/metric_history.json"


def load_eval_encoder():
    candidates = ["../gte-base-local", "thenlper/gte-base"]
    for name in candidates:
        try:
            return SentenceTransformer(name, device=("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception as e:
            print(f"Failed to load eval encoder {name}: {e}")
    raise RuntimeError("Cannot load sentence embedding model for similarity evaluation.")


def compute_generation_metrics(model, tokenizer, eval_pairs, eval_encoder, max_new_tokens=32):
    if len(eval_pairs) == 0:
        return 0.0, 0.0

    smoothie = SmoothingFunction().method1
    preds = []
    refs = []
    bleu_scores = []

    model.eval()
    for generation, target in eval_pairs:
        prompt = f"Given the following text, predict the target:\n\nText: {generation}\n\nTarget:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = generated_text[len(prompt):].strip()
        preds.append(pred)
        refs.append(target)
        bleu_scores.append(
            sentence_bleu(
                [target.split()],
                pred.split(),
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie
            )
        )

    pred_emb = eval_encoder.encode(preds, convert_to_tensor=True, normalize_embeddings=True)
    ref_emb = eval_encoder.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
    cos_sim = F.cosine_similarity(pred_emb, ref_emb).mean().item()
    bleu4 = float(np.mean(bleu_scores))
    return cos_sim, bleu4


def plot_training_curves(trainer, metric_history, save_path):
    train_steps = []
    train_losses = []
    eval_epochs = []
    eval_losses = []

    for rec in trainer.state.log_history:
        if "loss" in rec and "eval_loss" not in rec and "step" in rec:
            train_steps.append(rec["step"])
            train_losses.append(rec["loss"])
        if "eval_loss" in rec and "epoch" in rec:
            eval_epochs.append(rec["epoch"])
            eval_losses.append(rec["eval_loss"])

    metric_epochs = [m["epoch"] for m in metric_history]
    metric_sims = [m["embedding_similarity"] for m in metric_history]
    metric_bleu4 = [m["bleu4"] for m in metric_history]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_steps, train_losses, label="train_loss")
    if len(eval_epochs) > 0:
        axes[0].plot(eval_epochs, eval_losses, marker="o", label="eval_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("step/epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(metric_epochs, metric_sims, marker="o")
    axes[1].set_title("Embedding Similarity")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cosine similarity")
    axes[1].grid(alpha=0.3)

    axes[2].plot(metric_epochs, metric_bleu4, marker="o")
    axes[2].set_title("BLEU-4")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("BLEU-4")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


class GenerationMetricCallback(TrainerCallback):
    def __init__(self, eval_pairs, tokenizer, eval_encoder, metric_history):
        self.eval_pairs = eval_pairs
        self.tokenizer = tokenizer
        self.eval_encoder = eval_encoder
        self.metric_history = metric_history

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        cos_sim, bleu4 = compute_generation_metrics(
            model=model,
            tokenizer=self.tokenizer,
            eval_pairs=self.eval_pairs,
            eval_encoder=self.eval_encoder
        )
        rec = {
            "epoch": float(state.epoch),
            "embedding_similarity": float(cos_sim),
            "bleu4": float(bleu4)
        }
        self.metric_history.append(rec)
        print(f"[epoch {state.epoch:.2f}] embedding_similarity={cos_sim:.4f}, bleu4={bleu4:.4f}")
        return control

# Load data
path = '/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_gte-Qwen_20250312_192926.json'
with open(path, 'r') as f:
    data = json.load(f)

# Format data as [generation, target] pairs
data = [[item['generation'], item['target']] for item in data]

# Shuffle data
random.seed(42)
random.shuffle(data)

# Split data: use last 20 for evaluation
train_data = data[:-20]
eval_data = data[-20:]

print(f"Training on {len(train_data)} samples, evaluating on {len(eval_data)} samples")

# Load tokenizer and model
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare model for training
if use_lora:
    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # rank
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
else:
    # For full fine-tuning, ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")

# Tokenization function
def preprocess_function(examples):
    # Format as instruction for the model with the target
    inputs = []
    targets = []
    for gen, target in zip(examples["generation"], examples["target"]):
        # Create prompt without the target
        prompt = f"Given the following text, predict the target:\n\nText: {gen}\n\nTarget: "
        inputs.append(prompt)
        targets.append(target)
    
    # Tokenize inputs and targets separately
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    target_tokens = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    # Initialize labels with -100s (ignored in loss calculation)
    all_labels = [[-100] * 512 for _ in range(len(model_inputs["input_ids"]))]
    
    # Combine input_ids and target_ids for the full sequence
    for i in range(len(model_inputs["input_ids"])):
        # Get the non-padding part of the target (exclude padding tokens)
        target_ids = [id for id in target_tokens["input_ids"][i] if id != tokenizer.pad_token_id]
        
        # Create the full sequence by concatenating input and target
        full_input_ids = model_inputs["input_ids"][i].copy()
        
        # Remove padding from the end of input to make room for target
        padding_count = 0
        while full_input_ids and full_input_ids[-1] == tokenizer.pad_token_id:
            full_input_ids.pop()
            padding_count += 1
        
        # Calculate where target starts in the full sequence
        target_start = len(full_input_ids)
        
        # Add target tokens to the input
        full_input_ids.extend(target_ids)
        
        # Create new attention mask (1 for tokens, 0 for padding)
        new_attention_mask = [1] * len(full_input_ids) + [0] * (512 - len(full_input_ids))
        
        # Pad to max length if needed
        if len(full_input_ids) < 512:
            full_input_ids.extend([tokenizer.pad_token_id] * (512 - len(full_input_ids)))
        elif len(full_input_ids) > 512:
            full_input_ids = full_input_ids[:512]
            new_attention_mask = new_attention_mask[:512]
        
        # Add target token ids to labels
        for j, token_id in enumerate(target_ids):
            if target_start + j < 512:  # Make sure we don't go beyond max length
                all_labels[i][target_start + j] = token_id
        
        # Update the input_ids and attention_mask with the full sequence
        model_inputs["input_ids"][i] = full_input_ids
        model_inputs["attention_mask"][i] = new_attention_mask
    
    model_inputs["labels"] = all_labels
    return model_inputs

# Convert to datasets format
train_dataset = Dataset.from_dict({
    "generation": [item[0] for item in train_data],
    "target": [item[1] for item in train_data]
})

eval_dataset = Dataset.from_dict({
    "generation": [item[0] for item in eval_data],
    "target": [item[1] for item in eval_data]
})

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

eval_encoder = load_eval_encoder()
metric_history = []

# Define training arguments
if use_lora:
    # LoRA training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        push_to_hub=False,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=5
    )
else:
    # Full fine-tuning arguments - smaller batch size and learning rate
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,  # Lower learning rate for full fine-tuning
        per_device_train_batch_size=16,  # Smaller batch size to fit in memory
        per_device_eval_batch_size=16,
        num_train_epochs=10,  # Fewer epochs for full fine-tuning
        weight_decay=0.01,
        push_to_hub=False,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=5
    )

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[GenerationMetricCallback(eval_data, tokenizer, eval_encoder, metric_history)],
)

# Evaluate the model before training to get baseline loss
print("\n===== Evaluating model before training =====")
eval_results = trainer.evaluate()
print(f"Initial model perplexity: {np.exp(eval_results['eval_loss']):.2f}")
print(f"Initial model loss: {eval_results['eval_loss']:.4f}")
print("=" * 50)

# Train the model
trainer.train()

plot_training_curves(trainer, metric_history, curve_save_path)
with open(metric_history_path, "w") as f:
    json.dump(metric_history, f, indent=2)
print(f"Training curves saved to {curve_save_path}")
print(f"Metric history saved to {metric_history_path}")

# Save the fine-tuned model
if use_lora:
    model.save_pretrained("./fine_tuned_model")
else:
    # For full fine-tuning, save the entire model
    model_to_save = model
    model_to_save.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Full model saved to ./fine_tuned_model")

# Test the model on a few examples from the evaluation set
print("\n===== Testing the model on evaluation examples =====")
model.eval()
for i in range(min(5, len(eval_data))):
    generation = eval_data[i][0]
    target = eval_data[i][1]
    
    print(f"\nProcessing example {i+1}...")
    prompt = f"Given the following text, predict the target:\n\nText: {generation}\n\nTarget:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Generating prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_target = generated_text[len(prompt):].strip()
    
    print(f"Example {i+1}:")
    print(f"Generation: {generation}")
    print(f"True Target: {target}")
    print(f"Predicted: {predicted_target}")
    baseline_bleu_score = sentence_bleu([target.split()], generation.split())
    bleu_score = sentence_bleu([target.split()], predicted_target.split())
    print(f"Baseline BLEU Score: {baseline_bleu_score:.2f}")
    print(f"BLEU Score: {bleu_score:.2f}")
    print("-" * 50)
