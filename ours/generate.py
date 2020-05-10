import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from utils import cat_past, add_past, top_k_filtering, repeat_past, conditioning


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2-medium')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--top_k', type=int, default=12)
parser.add_argument('--length', type=int, default=50)
parser.add_argument('--embed_weights', nargs='+', type=float,default=[0.04])
parser.add_argument('--attn_weights', nargs='+', type=float, default=[0.02])
parser.add_argument('--cond_weights', nargs='+', type=float, default=[0.20])
parser.add_argument('--prefix', type=str, default='To conclude')
parser.add_argument('--special_length', type=int, default=3)
parser.add_argument('--condition', type=str, default='positive politics')
parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
)
args = parser.parse_args()
args.cond_weights = torch.tensor(args.cond_weights).to(args.device)

print(args.condition)
print(args.prefix)

config = GPT2Config.from_pretrained(args.model)
config.torchscript = True  # in order to untie input and output embedding
tokenizer = GPT2Tokenizer.from_pretrained(args.model)
cond_ids = tokenizer.encode(args.condition)
comma = tokenizer.encode(args.prefix, add_special_tokens=True)
comma = comma.index(13) if 13 in comma else -1
cond_ids = torch.tensor([cond_ids]).to(args.device)
model = GPT2LMHeadModel.from_pretrained(args.model, config=config).to(args.device)
org_model = GPT2LMHeadModel.from_pretrained(args.model).to(args.device)

decode_outputs = []
for i in range(args.num_samples):
    
    # Apply conditioning on embedding weights
    out_embed = model.get_output_embeddings()
    embed = model.get_input_embeddings()
    cond_embeds = embed(cond_ids)[0]
    for i in range(cond_embeds.shape[0]):
        embed.weight.data += args.embed_weights[i] * cond_embeds[i]
    embed.weight.data /= (1 + sum(args.embed_weights))


    input_ids = torch.tensor([tokenizer.encode(args.prefix, add_special_tokens=True)]).to(args.device)
    input_past = model(input_ids[:, :-1])[1]
    org_past = org_model(input_ids[:, :-1])[1]


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
    past = add_past(input_past, cond_past, args.attn_weights, 1)

    cnt = 1
    t = input_ids.shape[1] - 1
    for _ in range(input_ids.shape[1]-1, args.length):  # +1 for the last time step of prefix

        with torch.no_grad():
            position_ids = torch.tensor([[t]]).to(args.device)
            # Apply conditioning on attention maps
            logits, cur_past = model(input_ids[:, -1:], past=past, position_ids=position_ids)
            assert logits.shape[1] == 1
            logits = logits[:, -1]
            logits[:, 50256] = -float('inf')   # don't generate <|endoftext|> token
            logits, ids_to_retain = top_k_filtering(logits, args.top_k)
            # print('*' * 70)
            # print(tokenizer.decode(ids_to_retain[0]))

            input_past = cat_past(input_past, cur_past, last=True)
            for i in range(cond_ids.shape[1]):
                cur_past = model(cond_ids[:, i:i+1], position_ids=position_ids)[1]
                cond_past[i] = cat_past(cond_past[i], cur_past)
            past = add_past(input_past, cond_past, args.attn_weights, cnt)

            logprobs = F.log_softmax(logits, dim=-1)

            # Apply conditioning by conditioned next word probabilities
            org_logits, org_past = org_model(input_ids[:, -1:], past=org_past, position_ids=position_ids)
            org_logits = org_logits[:, -1]
            org_logits[:, 50256] = -float('inf')   # don't generate <|endoftext|> token
            org_logprobs = F.log_softmax(org_logits, dim=-1)
            org_logprobs = conditioning(tokenizer, org_logprobs, cond_ids[0], org_model, repeat_past(org_past, args.top_k), ids_to_retain, args.cond_weights, cnt)  # cond_ids is of batch size 1

            probs = torch.exp((logprobs + org_logprobs) / 2)
            next_tokens = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        if cnt == args.special_length and comma != -1:   # remove "The following" prefix
            t -= comma
            input_ids = input_ids[:, comma:]
            input_past = model(input_ids[:, :-1])[1]
            org_past = org_model(input_ids[:, :-1])[1]
            for i in range(len(cond_past)):
                cond_past[i] = list(cond_past[i])
                for j in range(len(cond_past[i])):
                    cond_past[i][j] = cond_past[i][j][:, :, :, :-comma]
                cond_past[i] = tuple(cond_past[i])
            past = add_past(input_past, cond_past, args.attn_weights, cnt)

        args.top_k = min(16, args.top_k + 1)
        t += 1
        cnt += 1
    
    output = tokenizer.decode(input_ids[0])
    decode_outputs.append(output)

for i in decode_outputs:
    print(i)
