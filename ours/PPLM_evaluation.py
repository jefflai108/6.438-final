#!/usr/bin/env python
# coding: utf-8

import os
from transformers.modeling_gpt2 import GPT2LMHeadModel
# This downloads GPT-2 Medium, it takes a little while
_ = GPT2LMHeadModel.from_pretrained("gpt2-medium")
import sys
sys.path.insert(0, '/data/sls/scratch/clai24/seq2seq/nlg/6.438-final/')
from run_pplm import run_pplm_example


def pplm_examples():

    run_pplm_example(
        cond_text="The potato",
        num_samples=3,
        bag_of_words='military',
        length=50,
        stepsize=0.03,
        sample=True,
        num_iterations=3,
        window_length=5,
        gamma=1.5,
        gm_scale=0.95,
        kl_scale=0.01,
        verbosity='regular'
    )


    run_pplm_example(
        cond_text="Once upon a time",
        num_samples=10,
        discrim='sentiment',
        class_label='very_positive',
        length=50,
        stepsize=0.05,
        sample=True,
        num_iterations=10,
        gamma=1,
        gm_scale=0.9,
    kl_scale=0.02,
    verbosity='quiet'
)

# # Set up Evaluation Methods
# Evaluation with Perplexity
# Perplexity with a GPT-2:
# https://www.reddit.com/r/LanguageTechnology/comments/bucn53/perplexity_score_of_gpt2/
# https://github.com/huggingface/transformers/issues/4147

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch
import math
import numpy as np 

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
per_model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
per_model.eval() # no gradients
if torch.cuda.is_available():
    device = 'cuda'
else: device = 'cpu'
print('device is', device)
per_model.to(device) # run on GPU

def perplexity_score(sentence, model, tokenizer):
    # Encode a text inputs
    indexed_tokens = tokenizer.encode(sentence)
    # Convert indexed tokens in a PyTorch tensor
    input_ids = torch.tensor([indexed_tokens]) #.unsqueeze(0) 
    # If you have a GPU, put everything on cuda
    input_ids = input_ids.to(device)

    # Predict all tokens
    with torch.no_grad():
        outputs = model(input_ids, labels = input_ids)
    loss, logits = outputs[:2]

    return math.exp(loss)


# Evaluation with Distinct-1,2,3
# https://github.com/XinnuoXu/mmi_anti_pytorch/blob/master/diversity.py

SAMPEL_TIMES = 1 # sampled from the entire corpus
def diversity(sentences):
        import sys
        import random
        line_list = []
        for line in sentences:
                line_list.append(line.strip())
        d1, d2, d3 = 0.0, 0.0, 0.0
        for _ in range(0, SAMPEL_TIMES):
                uni_set, bi_set, tri_set = set(), set(), set()
                uni_num, bi_num, tri_num = 0, 0, 0
                for line in random.sample(line_list, min(2000, len(line_list))):
                        #print(line)
                        flist = line.split(" ")
                        for x in flist:
                                uni_set.add(x)
                                uni_num += 1
                        for i in range(0, len(flist)-1):
                                bi_set.add(flist[i] + "<XXN>" + flist[i + 1])
                                bi_num += 1
                        for j in range(0, len(flist)-2):
                                tri_set.add(flist[j] + "<XXN>" + flist[j + 1] + 
                                            "<XXN>" + flist[j + 2])
                                tri_num += 1
                d1 += len(uni_set) / float(uni_num)
                d2 += len(bi_set)  / float(bi_num)
                d3 += len(tri_set) / float(tri_num)
    
        #print("DIVERSE-1", d1 / SAMPEL_TIMES)
        #print("DIVERSE-2", d2 / SAMPEL_TIMES)
        #print("DIVERSE-3", d3 / SAMPEL_TIMES)
        #print("DISTINCT SENTENCES", len(set(line_list)) / float(len(line_list)))

        return (d1 / SAMPEL_TIMES), (d2 / SAMPEL_TIMES), (d3 / SAMPEL_TIMES)

# test evaluation methods
def evaluate_all(sentences):
    print('Evaluate by perplexity')
    print([perplexity_score(text, per_model, tokenizer) for text in sentences])
    print('\n')
    print('Evaluate by diversity')
    d1, d2, d3 = diversity(sentences)

    print("DIVERSE-1", d1)
    print("DIVERSE-2", d2)
    print("DIVERSE-3", d3)

def test_evaluation(a):
    evaluate_all(a)

# # Generate examples with our methods 
from generate import run_generate
def generate_text_with_our_methods():

    # loop over prefix and conditions and log the generated sentences and evaluation scores
    for condition in ["Religion", "Military", "Politics", "Science", "Legal", "Space", "Computers", "Technology"]:
        for prefix1 in ["The chiken", "The house", "The pizza", "The potato", "The lake"]:
            prefix2 = "The following is an article about " + condition + ". " + prefix1
            # print(prefix2) # added the following 
            for prefix in [prefix1, prefix2]: 
                print('Condition:', condition)
                print('Prefix:', prefix)
                perplexit_scores = []
                diversity_scores = []
                for k in range(20): # generate 100 samples per combination
                    a_lst = run_generate(prefix=prefix, condition=condition, length=100, device=device)
                    a_str = [''.join(a for a in a_lst)]
                    #print(a_str)
                    perplexit_scores.append(np.mean([perplexity_score(text, per_model, tokenizer) for text in a_str]))
                    diversity_scores.append(diversity(a_str))
                #evaluate_all(all_text)
                print('Perplexity score %.3f' % np.mean(perplexit_scores))
                print('Diversity score:')
                print('\t Dist-1: %.3f' % np.mean([a[0] for a in diversity_scores])) 
                print('\t Dist-2: %.3f' % np.mean([a[1] for a in diversity_scores])) 
                print('\t Dist-3: %.3f' % np.mean([a[2] for a in diversity_scores])) 
                print('********\n')

# # Generate examples with PPLM
def generate_text_with_pplm():
    for condition in ["military", "religion", "politics", "science", "legal", "space", "technology"]:
        for prefix1 in ["The chiken", "The house", "The pizza", "The potato", "The lake"]:
            for prefix in [prefix1]: 
                print('Condition:', condition)
                print('Prefix:', prefix)
                all_text = []
                for _ in range(1): # generate 100 samples per combination
                    generated_texts = run_pplm_example(
                                            cond_text=prefix,
                                            num_samples=20,
                                            bag_of_words=condition,
                                            length=100,
                                            stepsize=0.03,
                                            sample=True,
                                            num_iterations=3,
                                            window_length=5,
                                            gamma=1.5,
                                            gm_scale=0.95,
                                            kl_scale=0.01,
                                            verbosity='quiet'
                                        )
                    # get rid of the beginning '<|endoftext|>'
                    processed_texts = [sample.replace('<|endoftext|>','') for sample in generated_texts]
                    #print(processed_texts)
                    #print('')
                    print('Original perplexity score: %.3f' % perplexity_score(processed_texts[0], per_model, tokenizer))
                    avg_pplm_ppl = np.mean([perplexity_score(text, per_model, tokenizer) for text in processed_texts[1:]])
                    print('Perplexity score over %d samples: %.3f' % (len(processed_texts[1:]), avg_pplm_ppl))
                    diversity_scores = []
                    for text in processed_texts[1:]:
                        diversity_scores.append(diversity([text]))
                    print('Diversity score over %d samples:' % len(processed_texts[1:])) 
                    print('\t Dist-1 %.3f' % np.mean([a[0] for a in diversity_scores]))
                    print('\t Dist-2 %.3f' % np.mean([a[1] for a in diversity_scores]))
                    print('\t Dist-3 %.3f' % np.mean([a[2] for a in diversity_scores]))
                print('********\n')

if __name__ == '__main__':
    
    a=["""My dog died in February, after suffering from severe arthritis. He had been suffering with a terrible cold that was causing his skin to break. I couldn't afford a replacement dog and couldn't afford to have him taken to the vet. I knew the vet would be""",
  """My dog died after getting stuck in a tree... I don't wish this to happen to my favorite character from one of my favorite movies, \"The Magnificent Seven." "It's all sad when someone dies so well." The Magnificent Seven movie was""",
  """My name is Jeff. How are you?"""]
    test_evaluation(a)
    #generate_text_with_our_methods()
    generate_text_with_pplm()
