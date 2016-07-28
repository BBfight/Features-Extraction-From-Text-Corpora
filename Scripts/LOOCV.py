__author__ = 'Ben'

from feature_extraction_transcript import *
import csv
import os
import pandas as pd
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.neighbors as ng
import sklearn.naive_bayes as nb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# setting number of folds
folds = 30

# Here you put the paths of your files
posTrainPath = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/LOOCV/train/positive"
negTrainPath = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/LOOCV/train/negative"
posTestPath = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/LOOCV/test/positive"
negTestPath = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/LOOCV/test/negative"
trainPath = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/LOOCV/train/"
testPath = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/LOOCV/test/"

# min_frequency_ratio_grams_list = np.arange(1.1, 2.1, 0.1)
# min_count_list = np.arange(40, 220, 20)

# list where to store predictions for the statistical significance test
predictions = []

# for value in min_frequency_ratio_grams_list:
# for value in min_count_list:

POS = True
# the minimum ratio POS/NEG (NEG/POS) for general stuff
min_frequency_ratio_general = 1.1
# the minimum ratio POS/NEG (NEG/POS) for grams and POS stuff
min_frequency_ratio_grams = 1.1
# min_frequency_ratio_grams = value
# the minimum number of occurrences of any gram/POS to be considered
min_count = 100
# min_count = value
# the n-gram (n-POS) are computed up to n = up_to_what_gram
up_to_what_gram = 2

# running LOOCV various times
for c in range(5):
    i = 0
    result = []
    if c == 0:
        classifier = ng.KNeighborsClassifier(n_neighbors=3)
    elif c == 1:
        classifier = svm.SVC(kernel='poly')
    elif c == 2:
        classifier = lm.LogisticRegression()
    elif c == 3:
        classifier = nb.GaussianNB()

    # initialising counters for progress and correct predictions
    correct = 0
    progress = 0
    # beginning with the positive files
    for j in range(1, int(folds/2)+1):
        # moving one positive file from the train to the test
        os.rename(posTrainPath + '/p%i.txt' % j, posTestPath + '/p%i.txt' % j)
        # computes the features for all the sets needed.
        features = extract_features(posTrainPath, negTrainPath,min_frequency_ratio_general,min_frequency_ratio_grams, min_count,up_to_what_gram)
        # TRAIN
        posTrainValues = compute_features(posTrainPath,features,up_to_what_gram)
        negTrainValues = compute_features(negTrainPath,features,up_to_what_gram)
        # TEST(only positive values)
        posTestValues = compute_features(posTestPath,features,up_to_what_gram)

        with open(trainPath + "train.csv", 'w') as csvTrain:
            writer = csv.writer(csvTrain, delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in posTrainValues:
                writer.writerow(['POS']+line)
            for line in negTrainValues:
                writer.writerow(['NEG']+line)

        with open(testPath + "test.csv", 'w') as csvTest:
            writer = csv.writer(csvTest, delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in posTestValues:
                writer.writerow(['POS']+line)

        # loading train from just created csv
        train = pd.DataFrame.from_csv(trainPath + "train.csv", header=None, index_col=None)
        # loading test from just created csv
        test = pd.DataFrame.from_csv(testPath + "test.csv", header=None, index_col=None)
        # fitting classifier on data
        classifier.fit(train.ix[:, 1:train.columns.size], train.ix[:, 0])
        # predicting with the classifier
        prediction = classifier.predict(test.ix[0, 1:test.columns.size])
        if prediction == test.ix[0, 0]:
            correct += 1
        if c != 4:
            result.extend(prediction)
        else:
            result.extend('POS')

        # moving the file back to the original folder
        os.rename(posTestPath + '/p%i.txt' % j, posTrainPath + '/p%i.txt' % j)
        progress += 1
        print('progress is ', progress, ' over ', folds)

    # now handling the negative files
    for j in range(1, int(folds/2)+1):
        # moving one negative file from the train to the test
        os.rename(negTrainPath + '/n%i.txt' % j, negTestPath + '/n%i.txt' % j)
        # computes the features for all the sets needed.
        features = extract_features(posTrainPath, negTrainPath,min_frequency_ratio_general,min_frequency_ratio_grams, min_count,up_to_what_gram,POS)
        # TRAIN
        posTrainValues = compute_features(posTrainPath,features,up_to_what_gram)
        negTrainValues = compute_features(negTrainPath,features,up_to_what_gram)
        # TEST(only negative values)
        negTestValues = compute_features(negTestPath,features,up_to_what_gram)

        with open(trainPath + "train.csv", 'w') as csvTrain:
            writer = csv.writer(csvTrain, delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in posTrainValues:
                writer.writerow(['POS']+line)
            for line in negTrainValues:
                writer.writerow(['NEG']+line)

        with open(testPath + "test.csv", 'w') as csvTest:
            writer = csv.writer(csvTest, delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in negTestValues:
                writer.writerow(['NEG']+line)

        # loading train from just created csv
        train = pd.DataFrame.from_csv(trainPath + "train.csv", header=None, index_col=None)
        # loading test from just created csv
        test = pd.DataFrame.from_csv(testPath + "test.csv", header=None, index_col=None)
        # fitting classifier on data
        classifier.fit(train.ix[:, 1:train.columns.size], train.ix[:, 0])
        # predicting with the classifier
        prediction = classifier.predict(test.ix[0, 1:test.columns.size])
        if prediction == test.ix[0, 0]:
            correct += 1
        if c != 4:
            result.extend(prediction)
        else:
            result.extend('POS')

        # moving the file back to the original folder
        os.rename(negTestPath + '/n%i.txt' % j, negTrainPath + '/n%i.txt' % j)
        progress += 1
        print('progress is ', progress, ' over ', folds)

    # printing final accuracy over all 30 folds
    print(result)
    predictions.append(result)
    accuracy = correct/folds
    print("Run ", i+1, " terminated,", " accuracy over all folds is: ", accuracy)
    i += 1

# converting strings to numbers for the statistical significance test
for i in range(len(predictions)):
    predictions[i] = list(map(lambda x: 1 if x == 'POS' else 0, predictions[i]))

# Computing statistical significance
significance = st.f_oneway(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])
print('Correlation is: ', significance)

'''
### PLOTTING ###
plt.xlabel('Min Count')
# plt.xlabel('Min Frequency Ratio Grams')
plt.ylabel('Accuracy')
# plt.plot(min_frequency_ratio_grams_list, accuracies)
plt.plot(min_count_list, accuracies)
plt.show()
'''

'''
### RESULTS ###
LOGISTIC REGRESSION POS  => ACCURACY: 0.5
LOGISTIC REGRESSION NO POS => ACCURACY: 0.5
KNN POS => ACCURACY: 0.6
KNN NO POS => ACCURACY: 0.56
SVM LINEAR KERNEL POS => ACCURACY: 0.63
SVM LINEAR KERNEL NO POS => ACCURACY:0.66
SVM POLY KERNEL POS => ACCURACY: 0.63
SVM POLY KERNEL NO POS => FEATURES: 1300 ACCURACY: 0.7
NAIVE BAYES POS => ACCURACY: 0.53
NAIVE BAYES NO POS => ACCURACY: 0.56

BAG OF WORDS BASELINE NO POS SVM POLY KERNEL => FEATURES: 6000 ACCURACY: 0.8
ANOVA TEST NO POS => STATISTICS: 12.52 P-VALUE: 3.89*e^-9
ANOVA TEST POS => STATISTICS: 12.78  P-VALUE: 2.6*e^-9
'''