__author__ = 'Ben'
# -*- coding: utf-8 -*-

import os
import glob
import string
import nltk
import re
import collections as cl

# appending source paths in a dictionary
pathsDict = {'trainPos': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/Large Movie Reviews/train preprocessed/pos",
             'trainNeg': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/Large Movie Reviews/train preprocessed/neg",
             'testPos': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/Large Movie Reviews/test preprocessed/pos",
             'testNeg': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/Large Movie Reviews/test preprocessed/neg"}

# counter
path = 0

# creating a dictionary of lists of lists
documentsPerFolder = {'trainPos': [], 'trainNeg': [], 'testPos': [], 'testNeg': []}
for folder in pathsDict:
    os.chdir(pathsDict[folder])
    for file in glob.glob("*.txt"):
        f = open(file, encoding='utf8')
        sentencesPerDocument = []
        for line in f.readlines():
            # removing final points
            while line[-1] == '.':
                line = line[:-1]
            sentencesPerDocument.extend(line.split('.'))
        documentsPerFolder[folder].append(sentencesPerDocument)
        f.close()
    path += 1
    print('finished parsing folder ', path)

path = 0
progress = 0
digits = set({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})
# replacing occurrences of numbers with he token <NUM> and removing html tags and punctuation
for folder in documentsPerFolder:
    for i in range(len(documentsPerFolder[folder])):
        for j in range(len(documentsPerFolder[folder][i])):
            sentence = documentsPerFolder[folder][i][j]
            # filtering and replacing
            sentence = sentence.replace('\t', ' ')
            sentence = re.sub('<[^>]+>', '', sentence)
            for k in sentence.split():
                if digits.intersection(set(k)) != set():
                    sentence = sentence.replace(k, 'NUM')
            # punctuation removal and pos tagging
            tokenizer = nltk.tokenize.TweetTokenizer()
            sentence = sentence.translate(sentence.maketrans("", "", string.punctuation))
            sentence = tokenizer.tokenize(sentence)
            sentence = nltk.pos_tag(sentence)
            tagCoupleList = []
            for k in sentence:
                singleCouple = '/'.join(k)
                tagCoupleList.append(singleCouple)
            sentence = ' '.join(tagCoupleList)
            documentsPerFolder[folder][i][j] = sentence
        # removing empty elements
        documentsPerFolder[folder][i] = filter(None, documentsPerFolder[folder][i])
        progress += 1
        print('pos tagged document', progress,' over ', len(documentsPerFolder[folder]))
    path += 1
    progress = 0
    print('pos tagged folder ', path)

print('started writing final files...')

# writing files in a different folder according to performance evaluation
destDict = {'trainPos': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/Large Movie Reviews/train preprocessed/pos/p_%i.txt",
            'trainNeg': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/Large Movie Reviews/train preprocessed/neg/n_%i.txt",
            'testPos': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/Large Movie Reviews/test preprocessed/pos/p_%i.txt",
            'testNeg': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/Large Movie Reviews/test preprocessed/neg/n_%i.txt"}
counter = {'trainPos': 0, 'trainNeg': 0, 'testPos': 12500, 'testNeg': 12500}

# writing final preprocessed files
for folder in destDict:
    for i in range(len(documentsPerFolder[folder])):
        counter[folder] += 1
        file = open(destDict[folder] % counter[folder], 'w',  encoding='utf8')
        file.writelines(documentsPerFolder[folder][i])
        #file.write("\n".join(documentsPerFolder[folder][i]))
        file.close()