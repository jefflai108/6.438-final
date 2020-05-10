import torch
import torch.nn.functional as F


def repeat_past(past, top_k):
    past = list(past)
    for i in range(len(past)):
        past[i] = past[i].repeat(1, top_k, 1, 1, 1)
    past = tuple(past)
    return past


def top_k_filtering(logits, top_k=1, filter_value=-float("Inf"), min_tokens_to_keep=1):
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
    ids_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    ids_to_retain = torch.topk(logits, top_k)[1]
    logits[ids_to_remove] = filter_value
    return logits, ids_to_retain


def conditioning(tokenizer, logprobs, cond_ids, model, past, ids_to_retain, weights, cnt):
    cond_logits = model(torch.transpose(ids_to_retain, 0, 1), past=past)[0][:, -1]
    cond_logprobs = F.log_softmax(cond_logits, dim=-1)
    cond_logprobs = cond_logprobs[:, cond_ids] * weights[None, :] / cnt
    cond_logprobs = torch.mean(cond_logprobs, dim=-1)

    top_k = ids_to_retain.shape[1]
    # if top_k > 1 and torch.sum(cond_logprobs) > 0:
    #     cond_logprobs /= (torch.max(cond_logprobs) - torch.min(cond_logprobs))
    #     cond_logprobs *= (torch.max(logprobs[ids_to_retain]) - torch.min(logprobs[ids_to_retain]))

    cond_logprobs, _ = top_k_filtering(cond_logprobs, top_k//2)
    # print(_)
    # print('*' * 70)
    # print(tokenizer.decode(ids_to_retain[0][_]))
    logprobs[:, ids_to_retain[0]] += cond_logprobs[None]
    # logprobs, _ = top_k_filtering(logprobs, top_k//4)
    # print('*' * 70)
    # print(tokenizer.decode(_[0]))
    return logprobs


def cat_past(past, cur_past, last=False):
    rtn_past = []
    for i in range(len(past)):
        if last:
            rtn_past.append(torch.cat([past[i], cur_past[i][..., -1:, :]], dim=3))
        else:
            rtn_past.append(torch.cat([past[i], cur_past[i]], dim=3))
    rtn_past = tuple(rtn_past)
    return rtn_past


def add_past(past, cond_past, weights, cnt):
    assert len(past) == len(cond_past[0])
    past = list(past)  # or else 'tuple' object doesn't support item assignment
    for i in range(len(cond_past)):
        for j in range(len(past)):
            past[j] += weights[i] * cond_past[i][j] / (1 + sum(weights)) / cnt
    past = tuple(past)
    return past


