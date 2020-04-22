from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F


# Assume batch size = 1


DEV = 'gpu'
# COND = 'positive' #  sport funny'
COND = 'politics' #  sport funny'
TOP_K = 10
PREFIX = 'Author of the'
LENGTH = 100
WEIGHT = 0.1


def top_k_filtering(logits, top_k=1, filter_value=-float("Inf"), min_tokens_to_keep=1):
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    ids_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    ids_to_retain = torch.topk(logits, top_k)[1][0]
    logits[ids_to_remove] = filter_value
    return logits, ids_to_retain


def conditioning(logits, cond_ids, model, input_ids, ids_to_retain):
    input_ids = input_ids.repeat(TOP_K, 1)
    input_ids = torch.cat([input_ids, ids_to_retain.unsqueeze(1)], dim=-1)
    next_logits = model(input_ids)[0][:, -1]
    next_probs = F.softmax(next_logits, dim=-1)
    cond_logits = torch.log(torch.sum(next_probs[:, cond_ids], dim=-1))
    logits[:, ids_to_retain] += WEIGHT * cond_logits
    return logits


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
COND_IDS = tokenizer.encode(COND)

input_ids = torch.tensor([tokenizer.encode(PREFIX, add_special_tokens=True)])
for t in range(LENGTH):
    with torch.no_grad():
        logits = model(input_ids)[0][:, -1]
        logits, ids_to_retain = top_k_filtering(logits, TOP_K)
        probs = conditioning(logits, COND_IDS, model, input_ids, ids_to_retain)
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

print(tokenizer.decode(input_ids[0]))
