from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


MODEL = 'gpt2-medium'
DEV = 'cuda'
TOP_K = 35
LENGTH = 100
WEIGHTS = [0.01]
WEIGHTS = [0.01, 0.01]

COND = 'positive politics'
COND = 'negative politics'
COND = 'negative science'
COND = 'positive science'
COND = 'negative'
COND = 'positive'

PREFIX = 'To conclude'
PREFIX = 'The potato'
PREFIX = 'The chicken tastes'


def cat_past(past, cur_past, last=False):
    rtn_past = []
    for i in range(len(past)):
        if last:
            rtn_past.append(torch.cat([past[i], cur_past[i][..., -1:, :]], dim=3))
        else:
            rtn_past.append(torch.cat([past[i], cur_past[i]], dim=3))
    rtn_past = tuple(rtn_past)
    return rtn_past


def add_past(past, cond_past):
    assert len(past) == len(cond_past[0])
    past = list(past)  # or else 'tuple' object doesn't support item assignment
    for i in range(len(cond_past)):
        for j in range(len(past)):
            past[j] += WEIGHTS[i] * cond_past[i][j]
    past = tuple(past)
    return past


def top_k_filtering(logits, top_k=1, filter_value=-float("Inf"), min_tokens_to_keep=1):
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
    ids_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    ids_to_retain = torch.topk(logits, top_k)[1][0]
    logits[ids_to_remove] = filter_value
    return logits, ids_to_retain


tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV)
COND_IDS = tokenizer.encode(COND)

embed = model.get_input_embeddings()
cond_embeds = embed(torch.tensor([COND_IDS]).to(DEV))[0]
for i in range(cond_embeds.shape[0]):
    embed.weight.data += WEIGHTS[i] * cond_embeds[i]

cond_ids = torch.tensor([COND_IDS]).to(DEV)
cond_past = [None for i in range(cond_ids.shape[1])]


input_ids = torch.tensor([tokenizer.encode(PREFIX, add_special_tokens=True)]).to(DEV)
input_past = model(input_ids[:, :-1])[1]


for t in range(input_ids.shape[1]-1):
    with torch.no_grad():
        position_ids = torch.tensor([[t]]).to(DEV)
        for i in range(cond_ids.shape[1]):
            cur_past = model(cond_ids[:, i:i+1], position_ids=position_ids)[1]
            if cond_past[i] is None:
                cond_past[i] = cur_past
            else:
                cond_past[i] = cat_past(cond_past[i], cur_past)

for t in range(input_ids.shape[1]-1, LENGTH):  # +1 for the last time step of prefix

    with torch.no_grad():

        position_ids = torch.tensor([[t]]).to(DEV)
        past = add_past(input_past, cond_past)
        logits, cur_past = model(input_ids[:, -1:], past=past, position_ids=position_ids)
        logits = logits[:, 0]
        input_past = cat_past(input_past, cur_past, last=True)
        logits, ids_to_retain = top_k_filtering(logits, TOP_K)
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        for i in range(cond_ids.shape[1]):
            cur_past = model(cond_ids[:, i:i+1], position_ids=position_ids)[1]
            cond_past[i] = cat_past(cond_past[i], cur_past)

print(tokenizer.decode(input_ids[0]))
