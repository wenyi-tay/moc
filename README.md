# Metric for Opinion Consistency

Problem: Review summary evaluation for opinion consistency

Why it is an important problem to solve: We found that 8-20% of the abstractively generated summaries contain contrasting opinions. Contrasting opinions are opinions about the same target but opposite in sentiment polarity. For example, the service is good and the service is horrible. This happens within the same summary. 

## Test Dataset

We use the Yelp summarisation dataset in the Meansum paper. It consists of 200 businesses, 8 reviews and 200 human written summaries. The dataset can be obtained from https://github.com/sosuperic/MeanSum. The dataset is split into a development set and test set of 100 businesses each. We report the results on the test set only. We use the development set only if the metrics require tuning. 

## Systems

* Bertcent- We used the published outputs of the system. https://github.com/rktamplayo/PlanSum/tree/master/output
* Copycat- We used the published outputs of the system. https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/copycat/artifacts
* Denoisesum- We used the published outputs of the system. https://github.com/rktamplayo/DenoiseSum/tree/master/output
* Fewsum- We used the published outputs of the system. https://github.com/abrazinskas/FewSum/tree/master/artifacts
* Meansum- We used the published code to run Meansum. https://github.com/sosuperic/MeanSum
* Opiniondigest- We used the published code to run Meansum. https://github.com/megagonlabs/opiniondigest
* Plansum- We used the published outputs of the system. https://github.com/rktamplayo/PlanSum/tree/master/output


## Fine-tuning Dataset

We use Yelp Academic Dataset to create the fine-tuning datasets. 


