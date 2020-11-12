# Music-genre-classifier
COMP90049 - Intro to Machine Learning - 2020S2 - Assignment2

## Library used
- sklearn
- numpy
- pandas

## Report
- [report](./docs/COMP90049_Assignment_02.pdf)

## Kaggle
- https://www.kaggle.com/c/comp90049p2s2/leaderboard
- I am rank 176
- ![](./docs/kaggle.jpg)

## Peer review
- [peer reviews](./peer_review/)

## Result and feedback
- Final marks: 36 out of 40
- <img src="./docs/result.png"/>
- <img src="./docs/result4.jpg"/>
- <img src="./docs/result2.png"/>
- <img src="./docs/result3.jpg"/>
- <img src="./docs/result5.png"/>
- <img src="./docs/result6.png"/>
- [good report](./ExampleReport.pdf)

# how to run
- run all cells
- everything in report are in all cells
- ![](./readme/1.jpg)

# where to find functions
## TF-IDF
- section 5.3

## train, predict, evaluate functions
- defined in section 4
- called in section 6.3

## hyper-parameter tuning
- section 6.2

## error analysis
- section 6.4

## ensemble learning
- section 6.5

# For more details, read the jupyter notebook

## description
This file describes the data provided as part of  
COMP90049: Introduction to Machine Learning  
Project 2: Music Genre Prediction from Audio, Metadata, and Lyric Features!

The features and class labels are derived from the following published data sets

1. T. Bertin-Mahieux, D. P.W. Ellis, B. Whitman, and P. Lamere. The million song dataset. In Proceedings of the 12th International Conference on Music Information Retrieval (ISMIR),

2. A. Schindler and A. Rauber. Capturing the temporal domain in Echonest Features for improved classification effectiveness. In Proceedings of the 10th InternationalWorkshop on Adaptive Multimedia Retrieval (AMR), 2012.


### Data splits and format
The data set consists of audio, metadata, and text features for 8556 songs, as well as their
genre labels. The songs are split into a training set (7678 songs), development set (450 songs) and test set (428 songs).

For each data split, we provide a features.csv file, labels.csv file. Each file is in csv format 
(comma-separated values). 

### Features

The feature files (train_features.csv, valid_features.csv and test_features.csv) contains the following columns:

* trackID: unique identifier for each song (to be used for mapping songs to their labels)
* title: title of the song. Type: text.
* tags: A comma-separated list of tags representing the words that appeared in the lyrics of the song and are assigned by human annotators. Type: text / categorical.
* loudness: overall loudness in dB. Type: float / continuous.
* tempo: estimated tempo in beats per minute (BPM). Type: float / continuous.
* time_signature: estimated number of beats per bar. Type: integer.
* key: key the track is in. Type: integer/ nominal. 
* mode: major or minor. Type: integer / binary.
* duration: duration of the song in seconds. Type: float / continuous.
* vect_1 ... vect_148: 148 columns containing pre-computed audio features of each song. These features were pre-extracted from the 30 or 60 second snippets, and capture timbre, chroma, and mfcc aspects of the audio. Each feature takes a continuous value. Type: float / continuous.
 


### Labels

  
The label files (train_labels.csv and valid_labels.csv) contain the following columns:

* trackID: unique identifier for each song (to be used for mapping songs to their labels)
* genre: the genre label


All genre labels are taken from the following set of 8 genres:

1. Soul and Reggae
2. Pop
3. Punk
4. Jazz and Blues
5. Dance and Electronica
6. Folk
7. Classic Pop and Rock
8. Metal
