#!/usr/bin/env python
# coding: utf-8

import os
from transformers.modeling_gpt2 import GPT2LMHeadModel
# This downloads GPT-2 Medium, it takes a little while
_ = GPT2LMHeadModel.from_pretrained("gpt2-medium")
import sys
sys.path.insert(0, '/data/sls/scratch/clai24/seq2seq/nlg/6.438-final/')
from run_pplm import run_pplm_example

## Base examples 
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

## Set up Evaluation Methods
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
device = 'cpu'
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

## Generate examples with our methods for topics
from generate import run_generate
def generate_text_with_our_methods_topic():

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

## Generate examples with PPLM-BOW
def generate_text_with_pplm_bow():
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
                                            verbosity='quiet',
                                            device=device
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

##Generate examples with our methods for sentiments
def generate_text_with_our_methods_sentiment(sentiment_tokenizer, sentiment_classifier):

    for condition in ["negative", "positive"]:
        for prefix1 in ["The chiken", "The house", "The pizza", "The potato", "The lake"]:
            prefix2 = "The following is an article about " + condition + ". " + prefix1
            for prefix in [prefix1, prefix2]: 
                print('Condition:', condition)
                print('Prefix:', prefix)
                perplexit_scores = []
                diversity_scores = []
                sentiment_scores = []
                for k in range(20): # generate 100 samples per combination
                    a_lst = run_generate(prefix=prefix, condition=condition, length=100, device=device)
                    a_str = [''.join(a for a in a_lst)]
                    #print(a_str)
                    perplexit_scores.append(np.mean([perplexity_score(text, per_model, tokenizer) for text in a_str]))
                    diversity_scores.append(diversity(a_str))
                    sentiment_scores.append([sentiment_predict(text, sentiment_tokenizer, sentiment_classifier)['label'] == condition for text in a_str][0])
                print('Perplexity score %.3f' % np.mean(perplexit_scores))
                print('Sentiment acc. %.3f' % (sum(sentiment_scores) / len(sentiment_scores)))
                print('Diversity score:')
                print('\t Dist-1: %.3f' % np.mean([a[0] for a in diversity_scores])) 
                print('\t Dist-2: %.3f' % np.mean([a[1] for a in diversity_scores])) 
                print('\t Dist-3: %.3f' % np.mean([a[2] for a in diversity_scores])) 
                print('********\n')

## Generate examples with PPLM-Discrim
def generate_text_with_pplm_discrim(sentiment_tokenizer, sentiment_classifier):
    for condition in ["very_negative", "very_positive"]:
        for prefix1 in ["The chiken", "The house", "The pizza", "The potato", "The lake"]:
            for prefix in [prefix1]: 
                print('Condition:', condition)
                print('Prefix:', prefix)
                all_text = []
                for _ in range(1): # generate 100 samples per combination
                    generated_texts = run_pplm_example(
                                            cond_text=prefix,
                                            num_samples=20,
                                            discrim='sentiment',
                                            class_label=condition,
                                            length=100,
                                            stepsize=0.05,
                                            sample=True,
                                            num_iterations=10,
                                            gamma=1,
                                            gm_scale=0.9,
                                            kl_scale=0.02,
                                            verbosity='quiet',
                                            device=device
                                        )
                    # get rid of the beginning '<|endoftext|>'
                    processed_texts = [sample.replace('<|endoftext|>','') for sample in generated_texts]
                    #print(processed_texts)
                    #print('')
                    print('Original perplexity score: %f' % perplexity_score(processed_texts[0], per_model, tokenizer))
                    avg_pplm_ppl = np.mean([perplexity_score(text, per_model, tokenizer) for text in processed_texts[1:]])
                    print('Perplexity score over %d samples: %f' % (len(processed_texts[1:]), avg_pplm_ppl))
                    if condition == "very_negative": 
                        sent_label = 'negative'
                    else: sent_label = 'positive' 
                    avg_sen_clas = sum([sentiment_predict(text, sentiment_tokenizer, sentiment_classifier)['label'] == sent_label for text in processed_texts[1:]]) / len(processed_texts[1:])
                    print('External sentimnt classifier over %d samples: %.3f' % (len(processed_texts[1:]), avg_sen_clas))
                    diversity_scores = []
                    for text in processed_texts[1:]:
                        diversity_scores.append(diversity([text]))
                    print('Diversity score over %d samples:' % len(processed_texts[1:])) 
                    print('\t Dist-1 %f' % np.mean([a[0] for a in diversity_scores]))
                    print('\t Dist-2 %f' % np.mean([a[1] for a in diversity_scores]))
                    print('\t Dist-3 %f' % np.mean([a[2] for a in diversity_scores]))
                #evaluate_all(generated_texts)
                print('********\n')

## Sentiment classifier from Twitter
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

nltk.download('stopwords')

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

def setup_sentiment_classifier(
                KERAS_MODEL="model.h5", 
                WORD2VEC_MODEL="model.w2v",
                TOKENIZER_MODEL="tokenizer.pkl"
):

    ## load pretrained models   
    # Word2Vec
    w2v_model = gensim.models.word2vec.Word2Vec.load(WORD2VEC_MODEL)
    # Tokenizer
    import pickle 
    sentiment_tokenizer = Tokenizer()
    with open(TOKENIZER_MODEL, 'rb') as handle:
        sentiment_tokenizer = pickle.load(handle)
    vocab_size = len(sentiment_tokenizer.word_index) + 1
    # Embedding layer 
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in sentiment_tokenizer.word_index.items():
      if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    print(embedding_matrix.shape)

    embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False) 
    # Model 
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    # Loads the weights
    model.load_weights(KERAS_MODEL)

    return sentiment_tokenizer, model

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def sentiment_predict(text, sentiment_tokenizer, sentiment_classifier, include_neutral=False):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(sentiment_tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = sentiment_classifier.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  

if __name__ == '__main__':
    a=["""My dog died in February, after suffering from severe arthritis. He had been suffering with a terrible cold that was causing his skin to break. I couldn't afford a replacement dog and couldn't afford to have him taken to the vet. I knew the vet would be""",
  """My dog died after getting stuck in a tree... I don't wish this to happen to my favorite character from one of my favorite movies, \"The Magnificent Seven." "It's all sad when someone dies so well." The Magnificent Seven movie was""",
  """My name is Jeff. How are you?"""]
    test_evaluation(a)
    #generate_text_with_our_methods_topic()
    #generate_text_with_pplm_bow()
    sentiment_tokenizer, sentiment_classifier = setup_sentiment_classifier(
                        KERAS_MODEL="twitter_sentiment/model.h5", 
                        WORD2VEC_MODEL="twitter_sentiment/model.w2v", 
                        TOKENIZER_MODEL="twitter_sentiment/tokenizer.pkl")
    print(sentiment_predict('I am going home tonight', sentiment_tokenizer, sentiment_classifier))
    generate_text_with_pplm_discrim(sentiment_tokenizer, sentiment_classifier)
    #generate_text_with_our_methods_sentiment(sentiment_tokenizer, sentiment_classifier)
