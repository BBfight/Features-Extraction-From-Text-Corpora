__author__ = 'ben'

from feature_extraction_topic import *
import csv
import pandas as pd
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.neighbors as ng
import sklearn.naive_bayes as nb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Here you put the paths of your files where you save the csv
trainPath = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/sentencesByTopicStratified/Train/"
testPath = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/sentencesByTopicStratified/Test/"

filesDict = {'listP': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/sentencesByTopicStratified/list.txt",
             'stackP': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/sentencesByTopicStratified/stack.txt",
             'treeP': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/sentencesByTopicStratified/tree.txt"}

nameDict = {'listP': 'list.txt', 'stackP': 'stack.txt', 'treeP': 'tree.txt'}

#min_frequency_ratio_grams_list = np.arange(1.1, 2.1, 0.1)

#for value in min_frequency_ratio_grams_list:

# the minimum ratio POS/NEG (NEG/POS) for grams and POS stuff
min_frequency_ratio_grams = 1.1
# min_frequency_ratio_grams = value
# True means it extracts also POS features
WITH_POS = False
# the minimum number of occurrences of gram/POS per number of token considered
min_count_per_length = [20, 10]
# the n-gram (n-POS) are computed up to n = up_to_what_gram
up_to_what_gram = len(min_count_per_length)

# list where to store predictions for the statistical significance test
predictions = []

# performing cross validation by taking the first 23 sentences of each topic and then training on all the others and
# trying to predict all the remaining. In the next step we slide of 23, thus taking for testing all the sentences
# from 23 to 46 for each topic and the rest for the training. This is done until all the sentences are covered
for c in range(5):
    otherLinesDict = {'listP': [], 'stackP': [], 'treeP': []}
    final_accuracies = []
    # initializing counters
    skip = 0
    limit = 23
    accuracies = []
    result = []
    for i in range(10):
        # creating train and test fold
        for key in filesDict:
            source = open(filesDict[key])
            counter = 0
            file = open(testPath + nameDict[key], 'w')
            for line in source.readlines():
                # writing test lines
                counter += 1
                if counter <= skip:
                    otherLinesDict[key].append(line)
                elif counter <= limit*(i+1):
                    file.write(line)
                    if counter == limit*(i+1):
                        file.close()
                else:
                    otherLinesDict[key].append(line)
            source.close()
            # writing train lines
            file = open(trainPath + nameDict[key], 'w')
            for line in otherLinesDict[key]:
                file.write(line)
            file.close()
        # updating counters
        skip += limit
        for key in otherLinesDict:
            otherLinesDict[key] = []

        # Here you compute the features (save them somewhere if needed)
        features = extract_features([trainPath + 'list.txt', trainPath + 'stack.txt', trainPath + 'tree.txt'], min_frequency_ratio_grams, min_count_per_length, WITH_POS)
        print('features extracted: ', len(features))

        ### HANDLING TRAINING ###

        # computes the features for all the sets needed.
        # TRAIN
        listPValues = compute_features(trainPath + 'list.txt', features,up_to_what_gram)
        stackPValues = compute_features(trainPath + 'stack.txt', features,up_to_what_gram)
        treePValues = compute_features(trainPath + 'tree.txt', features,up_to_what_gram)

        with open(trainPath + 'train.csv', 'w') as csvTrain:
            writer = csv.writer(csvTrain, delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in listPValues:
                writer.writerow(line+['LIST'])#'1, '+str(line)[1:-1])
            for line in stackPValues:
                writer.writerow(line+['STACK'])#'0, '+str(line)[1:-1])
            for line in treePValues:
                writer.writerow(line+['TREE'])

        ### HANDLING TEST ###

        # computes the features for all the sets needed.
        listPValues = compute_features(testPath + 'list.txt', features, up_to_what_gram)
        stackPValues = compute_features(testPath + 'stack.txt', features, up_to_what_gram)
        treePValues = compute_features(testPath + 'tree.txt', features, up_to_what_gram)

        with open(testPath + 'test.csv', 'w') as csvTest:
            writer = csv.writer(csvTest, delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in listPValues:
                writer.writerow(line+['LIST'])#'1, '+str(line)[1:-1])
            for line in stackPValues:
                writer.writerow(line+['STACK'])#'0, '+str(line)[1:-1])
            for line in treePValues:
                writer.writerow(line+['TREE'])

        ### PREDICTION TIME ###
        correct = 0
        # loading train from just created csv
        train = pd.DataFrame.from_csv(trainPath + "train.csv", header=None, index_col=None)
        # loading test from just created csv
        test = pd.DataFrame.from_csv(testPath + "test.csv", header=None, index_col=None)
        # fitting an classifier on data
        if c == 0:
            classifier = ng.KNeighborsClassifier(n_neighbors=3)
        elif c == 1:
            classifier = svm.SVC(kernel='poly')
        elif c == 2:
            classifier = lm.LogisticRegression()
        elif c == 3:
            classifier = nb.GaussianNB()
        classifier.fit(train.ix[:, 0:(train.columns.size-2)], train.ix[:, train.columns[-1]])
        # predicting with the classifier
        prediction = classifier.predict(test.ix[:, 0:(test.columns.size-2)])
        if c != 4:
            result.extend(prediction)
        else:
            result.extend(['TREE']*len(test))
        for j in range(len(prediction)):
            if prediction[j] == test.ix[j, test.columns[-1]]:
                correct += 1
        # computing accuracy for the fold
        accuracies.append(float(correct/len(prediction)))
        print('completed test ', i+1, ' over 10, accuracy is ', accuracies[i])

    # computing overall accuracy
    predictions.append(result)
    finalAccuracy = float(sum(accuracies)/10)
    final_accuracies.append(finalAccuracy)
    print('FINAL ACCURACY OVER ALL FOLDS: ', finalAccuracy)

# converting strings to numbers for the statistical significance test
for i in range(len(predictions)):
    predictions[i] = list(map(lambda x: 1 if x == 'TREE' else(2 if x == 'STACK' else 3), predictions[i]))

# Computing statistical significance
significance = st.f_oneway(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])
print('Correlation is: ', significance)

'''
### PLOTTING ###
plt.xlabel('Min Frequency Ratio Grams')
plt.ylabel('Accuracy')
plt.plot(min_frequency_ratio_grams_list, final_accuracies)
plt.show()
'''

'''
### RESULTS ###
LOGISTIC REGRESSION POS => FEATURES: 150 ACCURACY: 0.5625
LOGISTIC REGRESSION NO POS => ACCURACY: 0.507
KNN POS => ACCURACY: 0.425
KNN NO POS => ACCURACY: 0.4188
SVM LINEAR KERNEL POS => 0.478
SVM LINEAR KERNEL NO POS => 0.504
SVM POLY KERNEL POS => 0.337
SVM POLY KERNEL NO POS => 0.334
NAIVE BAYES POS => 0.514
NAIVE BAYES NO POS => 0.491

BAG OF WORDS BASELINE NO POS LOGISTIC REGRESSION => FEATURES: 530 ACCURACY: 0.584
ANOVA TEST NO POS => STATISTICS: 525.364 P-VALUE: 0.0
ANOVA TEST POS => STATISTICS: 461.8 P-VALUE: 3.83*e^-319
'''