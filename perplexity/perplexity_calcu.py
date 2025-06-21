import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, text, max_length, stride, device="cuda"):
    """
    Calculates perplexity using a sliding window approach.

    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        text (str): The text from the test dataset.
        max_length (int): The max sequence length the model can handle.
        stride (int): The number of tokens to slide the window by.
        device (str): The device to run on ('cuda' or 'cpu').
    """
    model.to(device)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = [] # Negative Log Likelihoods
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        
        # Slice the input_ids and attention_mask
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        attention_mask = encodings.attention_mask[:, begin_loc:end_loc].to(device)
        
        # The target_ids are the same as input_ids for causal LMs
        target_ids = input_ids.clone()
        # For causal LMs, the loss is calculated on non-padded tokens.
        # The model automatically shifts the labels, so we don't need to.
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            # The 'loss' returned is the average NLL over the tokens in the sequence.
            # We multiply by the number of target tokens to get the total NLL.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    # Calculate the overall perplexity
    # We take the exponential of the average negative log likelihood
    total_nll = torch.stack(nlls).sum()
    avg_nll = total_nll / seq_len
    perplexity = torch.exp(avg_nll)

    return perplexity.item()