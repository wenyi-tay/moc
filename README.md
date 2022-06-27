# Metric for Opinion Consistency

Problem: Review summary evaluation for opinion consistency

Why it is an important problem to solve: We found that 8-20% of the abstractively generated summaries contain contrasting opinions. Contrasting opinions are opinions about the same target but opposite in sentiment polarity. For example, the service is good and the service is horrible. This happens within the same summary. 

## Test Dataset

We use the Yelp summarisation dataset in the Meansum paper. It consists of 200 businesses, 8 reviews and 200 human written summaries. The dataset can be obtained from https://github.com/sosuperic/MeanSum. The dataset is split into a development set and test set of 100 businesses each. We report the results on the test set only. 

We also use the Amazon summarisation dataset from the Copycat paper. It consists of 60 products, 8 reviews and 3 human written summaries per product. The dataset can be obtained from https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/. The dataset is split into 28 development and 32 test. We report the results on the test set only. 

## Systems

* Bertcent- We used the published outputs of the system. https://github.com/rktamplayo/PlanSum/tree/master/output
* Copycat- We used the published outputs of the system. https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/copycat/artifacts
* Denoisesum- We used the published outputs of the system. https://github.com/rktamplayo/DenoiseSum/tree/master/output
* Fewsum- We used the published outputs of the system. https://github.com/abrazinskas/FewSum/tree/master/artifacts
* Meansum- We used the published code to run Meansum. https://github.com/sosuperic/MeanSum
* Opiniondigest- We used the published code to run Opiniondigest https://github.com/megagonlabs/opiniondigest
* Plansum- We used the published outputs of the system. https://github.com/rktamplayo/PlanSum/tree/master/output


## Fine-tuning Dataset

We use Yelp Academic Dataset to create the fine-tuning datasets. We train our encoder with a supervised triplet loss objective. The triplet can be at review level or sentence level. The review level triplet is anchor review, positive review (a review of the same product and same sentiment polarity) and negative review ( a review of same product but different sentiment polarity). The triplet we use at sentence level is anchor sentence, positive sentence (a sentence in the same review) and negative sentence (a sentence from another review of the same product). 


