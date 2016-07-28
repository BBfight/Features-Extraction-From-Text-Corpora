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

# setting the folder where the txt files are stored
folder = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/transcripts"

# creating a list to save the files
txtFiles = []

# saving the pointers to the files into a list
os.chdir(folder)
for file in glob.glob("*.txt"):
    txtFiles.append(open(file, "r"))

# creating a list of lists in order to save each line of a document, for each document, in a list
linesPerDocument = []
for i in txtFiles:
    lines = []
    for line in i.readlines():
        if line[0] != '@' and not('%' in line):
            lines.append(line)
    linesPerDocument.append(lines)

# eliminating useless numbers and creating identifiers for student and instructors, discarding the event sentences
for i in range(len(linesPerDocument)):
    for j in range(len(linesPerDocument[i])):
        sentence = linesPerDocument[i][j]
        if '*' in sentence:
            sentence = sentence.split('*', 1)[1]
            if sentence[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                sentence = sentence.replace(sentence[:4], 'STU ')
            else:
                sentence = sentence.replace(sentence[:4], 'TEA ')
        linesPerDocument[i][j] = sentence


# removing occurrences of unknown characters, replacing unknown words with the token <UNK> and  numbers with <NUM>
blacklist = ['@', '#', '/', 'xxx', 'xx']
replaceList = ['(', ')', '<', '>', '+', '_']
for i in range(len(linesPerDocument)):
    for j in range(len(linesPerDocument[i])):
        sentence = linesPerDocument[i][j]
        for r in replaceList:
            sentence = sentence.replace(r, '')
        sentence = sentence.replace('\t', ' ')
        sentence = sentence.replace('OK', 'okay')
        sentence = sentence.replace('Im', 'I am')
        sentence = sentence.replace("we'll", 'we will')
        sentence = re.sub('\[[^\]]+\]', '', sentence)
        for k in sentence.split(" "):
            for b in blacklist:
                if b in k:
                    sentence = sentence.replace(k, '<UNK>')
            if k.isdigit():
                sentence = sentence.replace(k, '<NUM>')
        linesPerDocument[i][j] = sentence

# removing punctuation and adding pos tagging to sentences
for i in range(len(linesPerDocument)):
    for j in range(len(linesPerDocument[i])):
        tokenizer = nltk.tokenize.TweetTokenizer()
        sentence = linesPerDocument[i][j]
        sentence = sentence.translate(sentence.maketrans("", "", string.punctuation))
        sentence = sentence.split(' ', 1)
        marker = sentence[0]
        sentence = sentence[1]
        sentence = tokenizer.tokenize(sentence)
        sentence = nltk.pos_tag(sentence)
        tagCoupleList = []
        for k in sentence:
            singleCouple = '/'.join(k)
            tagCoupleList.append(singleCouple)
        sentence = marker + ' ' + ' '.join(tagCoupleList)
        linesPerDocument[i][j] = sentence

# opening the prediction csv where performances labels are stored
result = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/id_average-gain_total_length_performance.csv',
                              header=0, index_col=None)

# writing files in a different folder according to performance label
fileCountPos = 0
fileCountNeg = 0
destPathPos = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/positive_preprocessed/p%d.txt"
destPathNeg = "C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/negative_preprocessed/n%d.txt"

for i in range(len(linesPerDocument)):
    indexResult = np.where(result['id'] == ntpath.basename(txtFiles[i].name).split('.')[0])[0][0]
    if result['performance'][indexResult] == 'positive':
        fileCountPos += 1
        file = open(destPathPos % fileCountPos, 'w')
        for j in range(len(linesPerDocument[i])):
            file.write(linesPerDocument[i][j])
            file.write('\n')
        file.close()
        for line in fileinput.FileInput(destPathPos % fileCountPos, inplace=1):
            cleanedLine = line.strip()
            if cleanedLine:
                print(cleanedLine)
    else:
        fileCountNeg += 1
        file = open(destPathNeg % fileCountNeg, 'w')
        for j in range(len(linesPerDocument[i])):
            file.write(linesPerDocument[i][j])
            file.write('\n')
        file.close()
        # removing blank lines from txt
        for line in fileinput.FileInput(destPathNeg % fileCountNeg, inplace=1):
            cleanedLine = line.strip()
            if cleanedLine:
                print(cleanedLine)

# closing files
for i in txtFiles:
    i.close()
