# Metric for Opinion Consistency

Problem: Review summary evaluation for opinion cosistency

Why it is an important problem to solve: We found that 8-20% of the abstractively generated summaries contain contrasting opinions. Contrasting opinions are opinions about the same aspect but opposite in sentiment polarity. For example, the service is good and the service is horrible. This happens within the same summary. 

## Dataset

We will be using the Yelp summarisation dataset in the Meansum paper. It consists of 200 businesses, 8 reviews and 200 human written summaries. The dataset can be obtained from (insert link). The dataset is also split into a development set and test set of 100 businesses each. We report the results on the test set only. We use the development set only if the metrics require tuning. 

## Systems

We use the available code to run Meansum and Opiniondigest, and used the published outputs of the systems for Copycat and Denoisesum on the Yelp dataset.

