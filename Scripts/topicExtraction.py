__author__ = 'Ben'

import os
import glob
import string
import pandas as pd
import numpy as np
import ntpath
import fileinput
import nltk
import re

#downloading resources
#nltk.download()

# storing the folders where the txt files are stored
folder = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/transcripts_preprocessed"
# initializing a dictionary containing an empty list per topic where to store sentences
topicDict = {'list': [], 'stack': [], 'tree': []}
# setting the topics(also with their plural forms)
topics = ['list', 'lists', 'stack', 'tree', 'trees']

# creating a list where to save the files pointers
txtFiles = []

# storing the pointers to the files into a list
os.chdir(folder)
for file in glob.glob("*.txt"):
    txtFiles.append(open(file, "r"))

# storing each sentence according to the topic in the relative list
for i in txtFiles:
    lines = []
    for line in i.readlines():
        for j in topics:
            if j in line:
                topic = j
        # checking that the sentence has at least three words and does not contain
        # one of the topics explicitly, otherwise we discard it
        if line[0] == 'S' and not(topic in line) and (len(line.split())-1) >= 3:
            line = line.split(' ', 1)[1]
            for key in topicDict.keys():
                if key in topic:
                    topicDict[key].append(line)

print('writing files...')

# writing files in a different folder according to the relative topic
destDict = {'tree': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/sentencesByTopic/tree.txt",
            'stack': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/sentencesByTopic/stack.txt",
            'list': "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/sentencesByTopic/list.txt"}

# writing each sentences list(specific topic) into a different file
for i in topicDict.keys():
        file = open(destDict[i], 'w')
        for line in topicDict[i]:
            file.write(line)
            file.write('\n')
        file.close()
        for line in fileinput.FileInput(destDict[i], inplace=1):
            cleanedLine = line.strip()
            if cleanedLine:
                print(cleanedLine)

# closing files
for i in txtFiles:
    i.close()
