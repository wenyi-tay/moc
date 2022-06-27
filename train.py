# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:15:44 2022

@author: S3642603
"""

from sentence_transformers import InputExample, LoggingHandler, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
import csv
import logging
from supervisedTripletLoss import SupervisedTripletLoss, TripletDistanceMetric
from sentencerTransformerEdited import SentenceTransformerWenyi
import sys

#for i in range(1, len(sys.argv)):
#    print('argument:', i, 'value:', sys.argv[i])


# Read the dataset
model_name = 'all-MiniLM-L6-v2'
triplet_type = str(sys.argv[1]) # either review or sentence
train_batch_size = int(sys.argv[2]) #The larger you select this, the better the results (usually). But it requires more GPU memory
num_epochs = int(sys.argv[3])
my_margin = int(sys.argv[4])
my_temperature = int(sys.argv[5]) #multiply by 0.01
my_alpha = int(sys.argv[6])  #multiply by 0.001
learning_rate = int(sys.argv[7]) # multiply by 0.000001
fname_train = str(sys.argv[8])  #multiply by 0.001
fname_dev = str(sys.argv[9]) # multiply by 0.000001
num_run = 1

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


print("Which loss am I using?")
if my_alpha == 0:
  print("Using Triplet Loss")
  model_save_path = "output/training_tripletloss_" + triplet_type + "_" + str(my_margin) + "margin" + str(num_epochs) + "epochs" + str(train_batch_size) + "batch" + str(learning_rate) + "lr" + "ReducedLabels" + str(num_run) + "run_" + model_name
else:
  print("Using Supervised Triplet Loss")
  model_save_path = "output/training_supervisedtripletloss_" + triplet_type + "_" + str(my_margin) + "margin" + str(my_temperature)+"temp" +str(my_alpha)+ "alpha" + str(num_epochs) + "epochs" + str(train_batch_size) + "batch" + str(learning_rate) + "lr" + "ReducedLabels" + str(num_run) + "run_" + model_name


model = SentenceTransformerWenyi(model_name)


## read the training dataset
logger.info("Read Yelp Train Dataset")
train_examples_setA = []
with open(fname_train, encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn)
    for row in reader:
      train_examples_setA.append(InputExample(texts=[row['sent1'], row['sent2'], row["sent3"]], label = int(row['biz1_type_first_clean_label'])))

    
## read the development dataset    
logger.info("Read Yelp Dev Dataset")
dev_examples_setA = []
with open(fname_dev, encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn)
    for row in reader:
        dev_examples_setA.append(InputExample(texts=[row['sent1'], row['sent2'], row["sent3"]], label = int(row['biz1_type_first_clean_label'])))
        
logger.info("Set Dataloader")   
train_dataloader_setA = DataLoader(train_examples_setA, shuffle=True, batch_size=train_batch_size)

logger.info("Set Loss Function") 
if my_alpha == 0:
  train_loss_setA  = losses.TripletLoss(model=model, 
                                        distance_metric=TripletDistanceMetric.COSINE,
                                        triplet_margin = my_margin)
else:
  train_loss_setA  = SupervisedTripletLoss(model=model, 
                                          distance_metric=TripletDistanceMetric.COSINE, 
                                          triplet_margin = my_margin,
                                          temperature = my_temperature * 0.01,
                                          alpha = my_alpha*0.001)


logger.info("Set Evaluator")
if my_alpha == 0:
  evaluator = TripletEvaluator.from_input_examples(dev_examples_setA, 
                                                   name = "tripletloss_" + triplet_type + "_"  + str(my_margin) + "margin" + str(num_epochs) + "epochs" + str(train_batch_size) + "batch" + str(learning_rate) + "lr" + "ReducedLabels" + str(num_run) + "run_" + model_name)
else:
  evaluator = TripletEvaluator.from_input_examples(dev_examples_setA, 
                                                   name = "supervisedtripletloss_" + triplet_type + "_" + str(my_margin) + "margin" + str(my_temperature)+"temp" +str(my_alpha)+ "alpha" + str(num_epochs) + "epochs" + str(train_batch_size) + "batch" + str(learning_rate) + "lr" + "ReducedLabels" + str(num_run) + "run_" + model_name)

warmup_steps = int(len(train_dataloader_setA) * num_epochs * 0.1) #10% of train data
print(warmup_steps)



# Train the model
learning_rate = learning_rate * 0.000001 
# print("Learning rate: " + str(learning_rate))


train_objectives = [(train_dataloader_setA, train_loss_setA)]
model.fit(train_objectives=train_objectives,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=500,
          warmup_steps=warmup_steps,
          optimizer_params= {'lr': learning_rate},
          output_path=model_save_path) 
      
      
## end
      
      
      
      
      
      
      
      
      
      