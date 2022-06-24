# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 09:33:00 2022

@author: S3642603
"""

from torch.types import Device
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    
class SupervisedTripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:
    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).
    Margin is an important hyperparameter and needs to be tuned respectively.
    For further details, see: https://en.wikipedia.org/wiki/Triplet_loss
    :param model: SentenceTransformerModel
    :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric contains common distance metrices that can be used.
    :param triplet_margin: The negative should be at least this much further away from the anchor than the positive.
    Example::
        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model=model)
    """
    def __init__(self, 
                 model: SentenceTransformer, 
                 distance_metric=TripletDistanceMetric.EUCLIDEAN, 
                 triplet_margin: float = 3, 
                 temperature = 0.30,
                 alpha = 0.05):
        super(SupervisedTripletLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin
        self.temperature = temperature
        self.alpha = alpha

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name, 'triplet_margin': self.triplet_margin}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_anchor, rep_pos, rep_neg = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        ## The label determine whether the difference in the margin
        ## For same category but different polarity, margin is small
        ## For different category, regardless of the polarity, the margin is small
        ## set a default margin of 5
        ## margin is the additional margin for label = 1
        ## when label = 1 means the negative example is of a different category
        ## print(type(labels))


        rep = torch.cat([rep_anchor, rep_pos, rep_neg], dim=0)

        batch_size = rep.shape[0] ## 

        labels = torch.cat([labels, labels, labels], dim=0)
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device = "cuda")

        contrast_feature = rep
        anchor_feature = rep
        anchor_count = 2 ## we have two views

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print("Length of anchor dot product")
        # print(anchor_dot_contrast)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device = "cuda"),
            0
        )
        # print('Length of Logits Mask')
        # print(logits_mask)

        ## it produces 1 for the non-matching places and 0 for matching places i.e its opposite of mask
        mask = mask * logits_mask
        # compute log_prob with logsumexp

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()


        exp_logits = torch.exp(logits) * logits_mask

        ## log_prob = x - max(x1,..,xn) - logsumexp(x1,..,xn) the equation
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        supconloss = -1 * mean_log_prob_pos
        supconloss = supconloss.mean()
        #print('Supervised Contrastive Loss')
        #print(supconloss)


        triloss = F.relu(distance_pos - distance_neg + self.triplet_margin*0.1)
        triloss = triloss.mean()
        #print("Triplet Loss")
        #print(triloss)

        combined_loss  = self.alpha*supconloss + (1-self.alpha)*triloss
        #print(combined_loss)

        return combined_loss    