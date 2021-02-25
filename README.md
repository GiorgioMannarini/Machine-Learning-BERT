# Twitter Sentiment Analysis - EPFL course challenge

## Authors (rainbow-triangle 🌈)

* Giorgio Mannarini
* Maria Pandele
* Francesco Posa

## Introduction
This project performs supervised classification of tweets. It predicts if a
tweet message used to contain a positive :) or negative :( smiley, by
considering only the remaining text.We implement various methods to represent 
tweets (TF-IDF, Glove embeddings) and different machine learning algorithms to 
classify them, from more classical ones to recurrent neural networks and deep
learning.

In short, we compared: K-Nearest Neighbors, Naive Bayes, Logistic Regression,
Support Vector Machines (linear), Random Forest, Multi-layer Perceptron, Gated
Recurrent Unit, Bert. Moreover, we also make an ensemble based on voting
between all of them.  

For more details, read the [report.pdf](https://github.com/CS-433/cs-433-project-2-rainbow-triangle/blob/master/report.pdf).

### Results at a glance

Our best model was based on Bert (large-uncased) and had a 0.902 accuracy and 0.901 F1 score
on AIcrowd.

## Dependencies
To properly run our code you will have to install some dependencies. Our 
suggestion is to use a Python environment (we used Anaconda). 
GRU and Bert are built on TensorFlow, with Keras as a wrapper, while the 
baseline has been done in scikit-learn. In alphabetical order, you should have:

- joblib 0.17 `pip install joblib`
- nltk 3.5 `pip install nltk`
- numpy 1.18.5 `pip install numpy`
- pandas 1.1.2 `pip install pandas`
- tensorflow 2.3.1 `pip install --upgrade tensorflow`
- transformers 3.4.0  `pip install transformers`
- scikit-learn 0.23.2 `pip install -U scikit-learn`
- setuptools 50.3 `pip install setuptools`
- symspellpy 6.7 `pip install symspellpy`
- vaderSentiment 3.3.2 `pip install vaderSentiment`

## Project structure

This is scheleton we used when developing this project. We recommend this
structure since all the files' locations are based on it.

`classes/`: contains all our implementation

`logs/`: contains outputed logs during training

`preprocessed_data/`: we are saving/loading the preprocessed data here/from here

`submissions/`: contains AIcrowd submissions

`utility/`: contains helpful resources for preprocessing the tweets

`weights/`: contains saved weights

`Extract_emoticons.ipynb`: extracts emoticons from full dataset of tweets which
are later manually processed and translated to Glove specific tags

`constants.py`: defines constants used throughout preprocessing, training and
inference

`run.py`: main script, more details on how to use it in the next section

## How to run

There are several ways to run it. You can either re-run everything from data
preprocessing to training and inference. Or you can just load our already
trained models and make predictions. **Note**: all the requirements in terms of hardware are in the README in the weights folder.
**If you just want to reproduce our best
submission then skip to [Best submission on AIcrowd](#best-submission-on-aicrowd)
section.**

### Step 1. Download the raw data
Skip this section if you only want to make predictions.

Download the raw data from [https://www.aicrowd.com/challenges/epfl-ml-text-classification](https://www.aicrowd.com/challenges/epfl-ml-text-classification)
and put it in a new top level folder called `data`.
So you should have something like this:
```
├── data
│   ├── train_pos.txt
│   ├── train_neg.txt
│   ├── train_pos_full.txt
│   ├── train_neg_full.txt
│   └── test_data.txt
```

### Step 2. Download the GloVe File.
For our Recurrent Neural Network based on GRU, we use a Pre-Trained Embedding Layer, where each 100-dimensional GloVe vector has been obtained by the Stanford University on twitter data. Please download the file and put it in the data folder. If you didn't skip the previous step, you should have a structure like this: 

```
├── data
│   ├── train_pos.txt
│   ├── train_neg.txt
│   ├── train_pos_full.txt
│   ├── train_neg_full.txt
│   |── test_data.txt
|   └── glove.twitter.27B.100d.txt
```
Otherwise, you should have only the `glove.twitter.27B.100d.txt` in the data folder. This file is necessary even if you do not want to train the model again.
Download:
- Stanford: http://nlp.stanford.edu/data/glove.twitter.27B.zip (please use only the 100d file).
- Alternative (faster): https://drive.google.com/file/d/15p0lHVX1UxL3K9hn4SFZde_2LLeTMquW/view?usp=sharing

**Total required space**: 974 MB

### Step 3. Download the already preprocessed tweets
Skip this section if you did [Step 1](#step-1-download-the-raw-data) and want
to do your own preprocessing.

If you want to download the preprocessed tweets then download them from 
[this Drive link](https://drive.google.com/drive/folders/16izsD7W0SG3AF094cW0JpcfnPFRF1aXY?usp=sharing)
and save them into the top level [`preprocessed_data/`](https://github.com/CS-433/cs-433-project-2-rainbow-triangle/tree/master/preprocessed_data)
folder.  
**Total required space**: 365 MB  
So you should have something like this:
```
├── preprocessed_data
│   ├── baseline
│   │   ├── test_preprocessed.csv   
│   │   └── train_preprocessed.csv
│   ├── bert
│   │   ├── test_preprocessed.csv   
│   │   └── train_preprocessed.csv
│   ├── gru
│   │   ├── test_preprocessed.csv   
│   │   └── train_preprocessed.csv
│   └── README.md
```

### Step 4. Download the models
Skip this section if you want to re-train the models.

If you want to download the pretrained models (HIGHLY RECOMMENDED for the
deep learning models) then download them from [this Drive link](https://drive.google.com/drive/folders/1o_exDi-gA0X1kSBTl9qUPpEGWZBX-MFy?usp=sharing)
and save them into the top level [`weights/`](https://github.com/CS-433/cs-433-project-2-rainbow-triangle/tree/master/weights)
folder.  
**Total required space**: 6.21 GB  
So you should have something like this:
```
├── weights
│   ├── baseline
│   │   ├── model-KNN.joblib   
│   │   ├── model-Logistic-Regression.joblib   
│   │   ...
│   │   └── model-SVM.joblib
│   ├── bert
│   │   └── model
│   │       ├── config.json
│   │       └── tf_model.h5
│   ├── gru
│   └── README.md
```

### Step 5. The actual run
[`run.py`](https://github.com/CS-433/cs-433-project-2-rainbow-triangle/blob/master/run.py) is the
main script which performs the data preprocessing, training (with hyperparameter
tuning) and inference.

A detailed help can be found by running:
```
python3 run.py -h
```

There are 3 types of options to keep in mind `-lp` (load preprocessed\_data), 
`-lt` (load trained models), `bert`/`gru`/`mlp`...and so on.
For example, if you did [Step 1](#step-1-download-the-raw-data) and want to
re-train a Naive Bayer Classifier, then run:
```
python3 run.py nbc
```
If you downloaded any intermediary data (preprocessed data or model) then run:
```
python3 run.py nbc -lp -lt
```
If you downloaded preprocessed tweets but want to retrain the Naive Bayes
classifier then run:
```
python3 run.py nbc -lp
```

In all cases, the script will make a submission file and save it in the
[`submissions/`](https://github.com/CS-433/cs-433-project-2-rainbow-triangle/tree/master/submissions).


## Best submission on AIcrowd
Our best submission on AIcrowd was a model based on Bert. Since this is a highly
computationally expensive model, we recommend to download the preprocessed 
tweets and trained model.

* Download preprocessed tweets from [this Drive link](https://drive.google.com/drive/folders/16izsD7W0SG3AF094cW0JpcfnPFRF1aXY?usp=sharing)
in the top level [`preprocessed_data/`](https://github.com/CS-433/cs-433-project-2-rainbow-triangle/tree/master/preprocessed_data)
folder.  
**Total required space**: 365 MB
* Download the model from [this Drive link](https://drive.google.com/drive/folders/1o_exDi-gA0X1kSBTl9qUPpEGWZBX-MFy?usp=sharing)
in the top level [`weights/`](https://github.com/CS-433/cs-433-project-2-rainbow-triangle/tree/master/weights)
folder.  
**Total required space**: 6.21 GB
* Run:
```
python3 run.py bert -lp -lt
```
This will take between 30 minutes and one hour on a normal laptop.
