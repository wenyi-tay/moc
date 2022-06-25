# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:15:44 2022

@author: S3642603
"""

"""
This script trains sentence transformers with a triplet loss function.
As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime
from zipfile import ZipFile

import csv
import logging
import os

from supervisedTripletLoss import *
from evaluator import *
from sentencerTransformerEdited import *

import sys

for i in range(1, len(sys.argv)):
    print('argument:', i, 'value:', sys.argv[i])




# Read the dataset
model_name = 'all-MiniLM-L6-v2'
train_batch_size = sys.argv[1] #The larger you select this, the better the results (usually). But it requires more GPU memory
num_epochs = sys.argv[2] 
num_biz = "28000"
num_examples = 1
my_margin = sys.argv[3] 
my_temperature = sys.argv[4] #multiply by 0.01
my_alpha = sys.argv[5]  #multiply by 0.001
learning_rate = sys.argv[6] # multiply by 0.000001
num_run = 1

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

if my_alpha == 0:
  print("Using Triplet Loss")
  model_save_path = "output/training_tripletloss_" + str(num_biz) + "sentences_"  + str(num_examples) +"examples" + str(my_margin) + "margin" + str(num_epochs) + "epochs" + str(train_batch_size) + "batch" + str(learning_rate) + "lr" + "ReducedLabels" + str(num_run) + "run_" + model_name
else:
  print("Using Supervised Triplet Loss")
  model_save_path = "output/training_supervisedtripletloss_" + str(num_biz) + "sentences_"  + str(num_examples) +"examples" + str(my_margin) + "margin" + str(my_temperature)+"temp" +str(my_alpha)+ "alpha" + str(num_epochs) + "epochs" + str(train_batch_size) + "batch" + str(learning_rate) + "lr" + "ReducedLabels" + str(num_run) + "run_" + model_name


model = SentenceTransformerWenyi(model_name)


logger.info("Read Yelp Pair train dataset")
train_examples_setA = []
with open("/content/drive/MyDrive/data/yelp_triplets_28000sentences_clean_" +str(num_examples) + "examples_train.csv", encoding="utf-8") as fIn:
# with open("/content/drive/MyDrive/data/yelp_triplets_sentences_1examples_subset1.csv", encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn)
    for row in reader:
      # print(row)
      # train_examples_setA.append(InputExample(texts=[row['sent1'], row['sent2'], row["sent3"]], label = int(row['label'])))
      train_examples_setA.append(InputExample(texts=[row['sent1'], row['sent2'], row["sent3"]], label = int(row['biz1_type_first_clean_label'])))
      
      
logger.info("Read Yelp Pair dev dataset")
dev_examples_setA = []
with open("/content/drive/MyDrive/data/yelp_triplets_4000sentences_clean_" + str(num_examples) + "examples_dev.csv", encoding="utf-8") as fIn:
# with open("/content/drive/MyDrive/data/yelp_triplets_sentences_1examples_subset2.csv", encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn)
    for row in reader:
        # dev_examples_setA.append(InputExample(texts=[row['sent1'], row['sent2'], row["sent3"]], label = int(row['label'])))
        dev_examples_setA.append(InputExample(texts=[row['sent1'], row['sent2'], row["sent3"]], label = int(row['biz1_type_first_clean_label'])))
        
        
train_dataloader_setA = DataLoader(train_examples_setA, shuffle=True, batch_size=train_batch_size)
if my_alpha == 0:
  print("Using Triplet Loss")
  train_loss_setA  = losses.TripletLoss(model=model, 
                                        distance_metric=TripletDistanceMetric.COSINE,
                                        triplet_margin = my_margin)
else:
  print("Using Supervised Triplet Loss")
  train_loss_setA  = SupervisedTripletLoss(model=model, 
                                          distance_metric=TripletDistanceMetric.COSINE, 
                                          triplet_margin = my_margin,
                                          temperature = my_temperature * 0.01,
                                          alpha = my_alpha*0.001)


logger.info("Read Yelp Pair dev dataset")
if my_alpha == 0:
  print("Using Triplet Loss")
  evaluator = TripletEvaluatorWenyi.from_input_examples(dev_examples_setA, 
                                                        name = "tripletloss_" + str(num_biz) + "sentences_"  + str(my_margin) + "margin" + str(num_epochs) + "epochs" + str(train_batch_size) + "batch" + str(learning_rate) + "lr" + "ReducedLabels" + str(num_run) + "run_" + model_name,
                                                        triplet_margin = my_margin)
else:
  print("Using Supervised Triplet Loss")
  evaluator = TripletEvaluatorWenyi.from_input_examples(dev_examples_setA, 
                                                        name = "supervisedtripletloss_" + str(num_biz) + "sentences_"  + str(num_examples) +"examples" + str(my_margin) + "margin" + str(my_temperature)+"temp" +str(my_alpha)+ "alpha" + str(num_epochs) + "epochs" + str(train_batch_size) + "batch" + str(learning_rate) + "lr" + "ReducedLabels" + str(num_run) + "run_" + model_name,
                                                        triplet_margin = my_margin)

warmup_steps = int(len(train_dataloader_setA) * num_epochs * 0.1) #10% of train data
print(warmup_steps)



# Train the model
learning_rate = learning_rate * 0.000001 
print("Learning rate: " + str(learning_rate))


train_objectives = [(train_dataloader_setA, train_loss_setA)]
model.fit(train_objectives=train_objectives,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=500,
          warmup_steps=warmup_steps,
          optimizer_params= {'lr': learning_rate},
          output_path=model_save_path)



      
      
      
      
      
      
      
      
      
      
      
      
      
      
      