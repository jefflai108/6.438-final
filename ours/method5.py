from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


MODEL = 'gpt2-medium'
DEV = 'cuda'
TOP_K = 40
LENGTH = 100
WEIGHTS = [0.01, 0.01]
WEIGHTS = [0.02]

COND = 'positive politics'
COND = 'negative politics'
COND = 'positive'
COND = 'negative science'
COND = 'positive science'
COND = 'negative'

PREFIX = 'To conclude'
PREFIX = 'The potato'
PREFIX = 'The chicken tastes'


def top_k_filtering(logits, top_k=1, filter_value=-float("Inf"), min_tokens_to_keep=1):
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
    ids_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    ids_to_retain = torch.topk(logits, top_k)[1][0]
    logits[ids_to_remove] = filter_value
    return logits, ids_to_retain


tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV)
COND_IDS = torch.tensor([tokenizer.encode(COND)]).to(DEV)

embed = model.get_input_embeddings()
cond_embeds = embed(COND_IDS)[0]
for i in range(cond_embeds.shape[0]):
    embed.weight.data += WEIGHTS[i] * cond_embeds[i]


input_ids = torch.tensor([tokenizer.encode(PREFIX, add_special_tokens=True)]).to(DEV)
past = model(input_ids[:, :-1])[1]


for t in range(input_ids.shape[1]-1, LENGTH):  # +1 for the last time step of prefix

    with torch.no_grad():
        logits, past  = model(input_ids[:, -1:], past=past)
        logits = logits[:, 0]
        logits, ids_to_retain = top_k_filtering(logits, TOP_K)
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

print(tokenizer.decode(input_ids[0]))
