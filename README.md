# NLP-Tweets-Sentiment

Packages Used:
* scikit-learn
* numpy


This project take the work described in “Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis” (Wilson et al, 2005) and implement the features they discuss for use in the sentiment classification of tweets. Input tweets are classified as positive, negative, or neutral. Our approach uses the scikit-learn toolkit’s SGDClassifier as a MaxEnt model. In the experimentation phase, the model is evaluated on development split with different features oneach phase in order to discover which produce the highest Macro F1 score. Results are compared to a baseline model that only takes unigram features into account. 


Our final results show significant improvement within our domain when compared with a Unigram-only Maximum Entropy model where we see an absolute 10% improvement in Macro F1-scores (scaled by 100). 


Coded in Python and R. 
