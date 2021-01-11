`foo_test.py` contains code for extracting multiple types of features (binary lexical features, binary non-lexical features,
 real-valued non-lexical features). In `sklearn`, it involves extra work in the initial data processing to create multiple 
 directories and sub-directories for each kind of feature while ensuring that the filenames match for each document within 
 each features' directory. See `foo_splits.r` for an example of such processing.

If you run the actual model, you'll see it's not great, but that's not really the point. If you'd like to see the script in action anyway, you need to follow a few steps:

* pip/conda install `scikit-learn` (this should install the other dependencies, `numpy` and `scipy`, automatically)
* download `foo_test.py`, `foo_splits.r`, `lyrics.csv`, and `lyrics_test.csv` to the same folder
* change the directory path in `foo_splits.r` to that folder
* In R, run `install.packages(c("readr","stringr"))` if you don't have them installed already (note: both of those packages are included in the `tidyverse` package collection)
* run `foo_splits.r` twice, once with `trfn <- "lyrics.csv"` and once with `trfn <- "lyrics_test.csv"`

After that, you should be able to run `foo_test.py` without issue.
