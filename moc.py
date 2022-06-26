# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## load required packages
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys




def global_coherence_sos(summary, encoder, inspect = False, k = 1): 
  ## input:
  ### (1) summary: string, the summary to be evaluated
  ### (2) inspect: bool, to show a visualisation of the score, provide the sentence with lowest avg similarity
  ### (3) k: int, for inspection of k sentences with lowest similarity scores. Deafult is 1
  ## output: 
  ### (1) tuple of aggregation by mean, min, max

  ## sent tokenise first
  summary = re.sub('[.]+', '.', summary)
  list_sent = sent_tokenize(summary)
  ## print(list_sent)
 
  ## get sentence embeddings
  embeddings = encoder.encode(list_sent, convert_to_tensor=True)
  cosine_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

  ## compute the scores
  coh = []
  for i in range(len(list_sent)):
    for j in range((i+1),len(list_sent)):
      # print(i, j)
      coh.append(cosine_scores[i][j])
  # print(coh)
  score_mean = sum(coh)/len(coh)
  score_min = min(coh)
  score_max = max(coh)

  ## if the option is provides, 
  ## to provide some outputs
  if inspect == True:
    inspect_summary(list_sent, cosine_scores, k = k)

  return score_mean, score_min, score_max



def inspect_summary(list_sent, cosine_scores, k = 1):
  ## input:
  ### (1) list_sent: list, list of sentences in summary
  ### (2) k: int, number of sentence pairs I want to display. Default is 1.
  ## output:
  ### (1) The tokenised sentences
  ### (2) The sentence with the lowest average similarity with other sentences
  ### (3) k sentence pairs with lowest similari

  ## print the tokenised sentence
  ind = 0
  print("Sentences of summary:")
  for s in list_sent:
    print("S" + str(ind) + ": " + s)
    ind = ind + 1
  print("")
  print("") 

  lower_triang_df = np.where(np.tril(np.ones(cosine_scores.shape)).astype(bool),cosine_scores,np.nan)

  ## plot the similarity matrix
  pp = sns.heatmap(lower_triang_df)
  ticks = list(range(cosine_scores.shape[0]))
  ticks = ["S" + str(t) for t in ticks]

  # set x, y labels
  pp.set_xticklabels(ticks)
  pp.set_yticklabels(ticks) 
  plt.tight_layout()
  plt.title('Cosine Similarity of Sentence Pairs')

  ## get the sentence which have lowest similarity scores with other sentences
  lst_avg_scores_each_sentence = (cosine_scores.sum(axis=0)-cosine_scores.max(axis=0))/(cosine_scores.shape[0]-1)
  lst_avg_scores_each_sentence = lst_avg_scores_each_sentence.tolist()
  index_sentence_smallest_avg = lst_avg_scores_each_sentence.index(min(lst_avg_scores_each_sentence))

  print("Sentence least similar to other sentences in summary:")
  print("S" + str(index_sentence_smallest_avg) + ": " + list_sent[index_sentence_smallest_avg])
  print("")  
  print("") 

  # small N elements of array
  small_elems = sorted(cosine_scores.ravel())[:(2*k)] # Because symmetric
  small_elems = list(set(small_elems))
  # print(small_elems)

  print(str(k) + " sentence pair(s) with lowest similarity scores:")
  for e in small_elems:
    result = np.where(cosine_scores==e) 
    print("Similarity score: " + str(e))
    print("S" + str(result[0][0]) +": " + list_sent[result[0][0]])
    print("S" + str(result[0][1]) +": " + list_sent[result[0][1]])
    print("")

  ## end

