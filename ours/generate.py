import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from utils import cat_past, add_past, top_k_filtering, repeat_past, conditioning


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2-medium')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--length', type=int, default=100)
parser.add_argument('--embed_weights', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('--attn_weights', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('--cond_weights', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('--prefix', type=str, default='To conclude')
parser.add_argument('--condition', type=str, default='positive politics')
args = parser.parse_args()
args.cond_weights = torch.tensor(args.cond_weights).to(args.device)


tokenizer = GPT2Tokenizer.from_pretrained(args.model)
model = GPT2LMHeadModel.from_pretrained(args.model).to(args.device)
cond_ids = tokenizer.encode(args.condition)
cond_ids = torch.tensor([cond_ids]).to(args.device)

# Apply conditioning on embedding weights
embed = model.get_input_embeddings()
cond_embeds = embed(cond_ids)[0]
for i in range(cond_embeds.shape[0]):
    embed.weight.data += args.embed_weights[i] * cond_embeds[i]


input_ids = torch.tensor([tokenizer.encode(args.prefix, add_special_tokens=True)]).to(args.device)
input_past = model(input_ids[:, :-1])[1]


cond_past = [None for i in range(cond_ids.shape[1])]
for t in range(input_ids.shape[1]-1):
    with torch.no_grad():
        position_ids = torch.tensor([[t]]).to(args.device)
        for i in range(cond_ids.shape[1]):
            cur_past = model(cond_ids[:, i:i+1], position_ids=position_ids)[1]
            if cond_past[i] is None:
                cond_past[i] = cur_past
            else:
                cond_past[i] = cat_past(cond_past[i], cur_past)
past = add_past(input_past, cond_past, args.attn_weights)


for t in range(input_ids.shape[1]-1, args.length):  # +1 for the last time step of prefix

    with torch.no_grad():
        position_ids = torch.tensor([[t]]).to(args.device)
        # Apply conditioning on attention maps
        logits, cur_past = model(input_ids[:, -1:], past=past, position_ids=position_ids)
        assert logits.shape[1] == 1
        logits = logits[:, -1]
        logits, ids_to_retain = top_k_filtering(logits, args.top_k)

        input_past = cat_past(input_past, cur_past, last=True)
        for i in range(cond_ids.shape[1]):
            cur_past = model(cond_ids[:, i:i+1], position_ids=position_ids)[1]
            cond_past[i] = cat_past(cond_past[i], cur_past)
        past = add_past(input_past, cond_past, args.attn_weights)

        # Apply conditioning by conditioned next word probabilities
        logprobs = F.log_softmax(logits, dim=-1)
        probs = conditioning(logprobs, cond_ids[0], model, repeat_past(past, args.top_k), ids_to_retain, args.cond_weights)  # cond_ids is of batch size 1
        next_tokens = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)


print(tokenizer.decode(input_ids[0]))
