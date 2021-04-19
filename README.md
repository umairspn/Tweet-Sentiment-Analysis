# NLP-Tweets-Sentiment

Packages Used:
* scikit-learn
* numpy


### Features Implemented

Lexical Features:
* Unigram
* Bigram
* Trigram

Non-Lexical Binary Features:
* Airline

Non-Lexical Real-Value Features:
* Number of users '@' mentioned 


### Description 
This project take the work described in “Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis” (Wilson et al, 2005) and implement the features they discuss for use in the sentiment classification of tweets. Input tweets are classified as positive, negative, or neutral. Our approach uses the scikit-learn toolkit’s SGDClassifier as a MaxEnt model. In the experimentation phase, the model is evaluated on development split with different features oneach phase in order to discover which produce the highest Macro F1 score. Results are compared to a baseline model that only takes unigram features into account. 



### Results 
Our final results show significant improvement within our domain when compared with a Unigram-only Maximum Entropy model where we see an absolute 10% improvement in Macro F1-scores (scaled by 100). 


### Code Details 

`foo_test.py` contains code for extracting multiple types of features (binary lexical features, binary non-lexical features, real-valued non-lexical features). In `sklearn`, it involves extra work in the initial data processing to create multiple directories and sub-directories for each kind of feature while ensuring that the filenames match for each document within each features' directory. See `foo_splits.r` for an example of such processing.

If you run the actual model, you'll see it's not great, but that's not really the point. If you'd like to see the script in action anyway, you need to follow a few steps:

* pip/conda install `scikit-learn` (this should install the other dependencies, `numpy` and `scipy`, automatically)
* download `foo_test.py`, `foo_splits.r`, `lyrics.csv`, and `lyrics_test.csv` to the same folder
* change the directory path in `foo_splits.r` to that folder
* In R, run `install.packages(c("readr","stringr"))` if you don't have them installed already (note: both of those packages are included in the `tidyverse` package collection)
* run `foo_splits.r` twice, once with `trfn <- "lyrics.csv"` and once with `trfn <- "lyrics_test.csv"`

After that, you should be able to run `foo_test.py` without issue.


### Tools and Programming Language 

Coded in Python and R. 
Anaconda -- PyCharm -- Jupyter 


