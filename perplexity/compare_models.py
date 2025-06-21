import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Helper Function (copy the function from Step 3 here) ---
def calculate_perplexity(model, tokenizer, text, max_length, stride, device="cuda"):
    # ... (code from above)
    model.to(device)
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        attention_mask = encodings.attention_mask[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    perplexity = torch.exp(torch.stack(nlls).sum() / seq_len)
    return perplexity.item()


# --- 1. Load your Test Data ---
# Load from a file. For a robust test, this file should contain thousands of tokens.
with open("legal_test_data.txt", "r", encoding="utf-8") as f:
    test_text = f.read()

# --- 2. Define Your Models ---
models_to_compare = {
    "Fine-Tuned Law Model": "./path/to/your/finetuned-model",
    "Base Model": "meta-llama/Llama-2-7b-hf", # Or whatever your base model is
    "HF Competitor Law Model": "nlpaueb/legal-bert-base-uncased" # Example: use a relevant competitor
}

# --- 3. Set Parameters ---
# Use the context window size of your models. Check the model's config.json.
# For Llama 2 it's 4096.
MAX_LENGTH = 4096 
STRIDE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("-" * 30)

# --- 4. Calculate and Compare ---
results = {}

for name, model_path in models_to_compare.items():
    print(f"Calculating perplexity for: {name} ({model_path})")
    
    # Load model and tokenizer
    # Add trust_remote_code=True if necessary for your model architecture
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Calculate perplexity
    perplexity_score = calculate_perplexity(model, tokenizer, test_text, MAX_LENGTH, STRIDE, DEVICE)
    results[name] = perplexity_score
    
    print(f"Perplexity: {perplexity_score:.4f}")
    print("-" * 30)

    # Clear memory
    del model
    del tokenizer
    torch.cuda.empty_cache()


# --- 5. Print Final Results ---
print("\n" + "="*30)
print("           Final Results")
print("="*30)
for name, score in sorted(results.items(), key=lambda item: item[1]):
    print(f"{name:<30} | Perplexity: {score:.4f}")
print("="*30)
print("(Lower is better)")