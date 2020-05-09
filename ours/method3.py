from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# Assume batch size = 1
# Problems:
# 1. Next word prob != Classification prob
# 2. Negation is not considered (not good v.s. not bad)
# 3. Top k words don't contain valid words (The chicken tasts + negative)


MODEL = 'gpt2-medium'
DEV = 'cuda'
TOP_K = 100
LENGTH = 20
WEIGHT = 1
REDUCE = 'mean'

COND = 'positive science'
COND = 'negative science'
COND = 'negative politics'
COND = 'positive'
COND = 'negative'
COND = 'positive politics'

PREFIX = 'The potato'
PREFIX = 'To conclude'
PREFIX = 'The chicken tastes'


def merge_past(past, cur_past, last=False):
    assert len(past) == len(cur_past)
    rtn_past = []
    for i in range(len(past)):
        if last:
            rtn_past.append(torch.cat([past[i], cur_past[i][..., -1:, :]], dim=3))
        else:
            rtn_past.append(torch.cat([past[i], cur_past[i]], dim=3))
    rtn_past = tuple(rtn_past)
    return rtn_past


def top_k_filtering(logits, top_k=1, filter_value=-float("Inf"), min_tokens_to_keep=1):
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
    ids_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    ids_to_retain = torch.topk(logits, top_k)[1][0]
    logits[ids_to_remove] = filter_value
    return logits, ids_to_retain


def conditioning(logprobs, cond_ids, model, input_ids, ids_to_retain):
    input_ids = input_ids.repeat(TOP_K, 1)
    input_ids = torch.cat([input_ids, ids_to_retain.unsqueeze(1)], dim=-1)
    next_logits = model(input_ids)[0][:, -1]
    next_logprobs = F.log_softmax(next_logits, dim=-1)
    if REDUCE == 'max':
        cond_logprobs = torch.max(next_logprobs[:, cond_ids], dim=-1)[0]
    elif REDUCE == 'mean':
        cond_logprobs = torch.mean(next_logprobs[:, cond_ids], dim=-1)
    cond_logprobs /= (torch.max(cond_logprobs) - torch.min(cond_logprobs))
    cond_logprobs *= (torch.max(logprobs[:, ids_to_retain]) - torch.min(logprobs[:, ids_to_retain]))

    # print(logprobs[:, ids_to_retain])
    # print('-' * 80)
    # print(cond_logprobs)
    # plt.scatter(logprobs[0, ids_to_retain].cpu().numpy(), cond_logprobs.cpu().numpy())
    # plt.show()

    logprobs[:, ids_to_retain] += WEIGHT * cond_logprobs
    probs = torch.exp(logprobs)
    return probs


tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV)
COND_IDS = tokenizer.encode(COND)
cond_ids = torch.tensor([COND_IDS]).to(DEV)


input_ids = torch.tensor([tokenizer.encode(PREFIX, add_special_tokens=True)]).to(DEV)
input_past = model(input_ids[:, :-1])[1]

for t in range(input_ids.shape[1]-1, LENGTH):  # +1 for the last time step of prefix

    with torch.no_grad():
        position_ids = torch.tensor([[t]]).to(DEV)
        past = input_past
        for i in range(cond_ids.shape[1]):
            cur_past = model(cond_ids[:, i:i+1], position_ids=position_ids)[1]
            past = merge_past(past, cur_past)
        logits, cur_past = model(input_ids[:, -1:], past=past, position_ids=position_ids)
        logits = logits[:, 0]
        input_past = merge_past(input_past, cur_past, last=True)
        logits, ids_to_retain = top_k_filtering(logits, TOP_K)
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

print(tokenizer.decode(input_ids[0]))
