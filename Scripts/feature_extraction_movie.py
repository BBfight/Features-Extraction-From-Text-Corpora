# feature extraction function
from functools import reduce
import os
import glob
from collections import defaultdict

TESTING = False


def split_pos(element):
    temp = element.split('/')
    return (temp[0],temp[1])


def open_files_in_path(path):
    # creating a list to save the files
    linesPerDocument = []
    os.chdir(path)
    for file in glob.glob("*.txt"):
        tempFile = open(file, "r")
        # creating a list (file) of lists (lines) of sentences
        # where a sentence is a list of couples (word,tag)
        lines = []

        counter = 0
        for line in tempFile.readlines():
            counter += 1
            wordsPerLine = line.split()
            if len(wordsPerLine) > 0:
                lines.append(list(map(split_pos,wordsPerLine)))

        if counter:
            linesPerDocument.append(lines)
        # closing the file
        tempFile.close()

    # returning the newly created list
    return linesPerDocument


def ngrams_sentence(words, n=2, padding=True):
    """Compute n-grams with optional padding"""

    pad = [] if not padding else ['<S>']*(n-1)
    grams = pad + words + pad
    return list( (tuple(grams[i:i+n]) for i in range(0, len(grams) - (n - 1))))


def ngrams_file(file, n=2, padding=True):
    return list(reduce(lambda x,y: x + y,list(map(lambda x: ngrams_sentence(x,n,padding),file))))


def ngrams_collection(collection, n=2, padding=True):
    """

    :param collection: the data set to transform
    :param n: what gram you want. I.e. n=2 means we want 2-grams
    :param padding: padding = True if we want start and end tokens per line.
    :return: a collection of n-grams
    """
    return reduce(lambda x,y: x + y,list(map(lambda x: ngrams_file(x,n,padding),collection)))


def count_appearances(collection):
    """

    It creates a dictionary to count the appearances of the elements of the collection
    """
    counts = defaultdict(int)
    for ng in collection:
        counts[ng] += 1
    return counts


def extract_only_element(collection,element = 0):
    # It extracts either only the words or only the POS tags. It returns a collection like the previous one
    return list(map(lambda x:
                    list(map(lambda y:
                             list(map(lambda z: z[element],y)),x)),collection))


def extract_features(paths, min_frequency_ratio_general,min_frequency_ratio_grams, min_count_per_length,WITH_POS = True):
    """Extracts relevant features from the datasets contained in the paths

        input:
            paths: a list of paths of different classes of documents
            min_frequency_ratio: the minimum value accepted for FEA1/FEA2
                (eg. the count of a unigram in neg divided by the count of that unigram in pos must be at least min_frequency_ratio to be considered)
            min_count_per_length: the minimum value accepted for FEA per gram length
                (eg. the count of a unigram in pos must be at least min_count to be considered)
            WITH_POS: if True, we extract features for POS tags as well as words. If False, we only extract word features

        returns:
            a list of strings, each of them representing a feature
    """
    # our result will be a list of features
    resultFeatures = []

    # opens all the files, parses them by line and puts them into different lists (files) of lists (lines) of texts,
    # then closes them. "text" is a list of couples (word,tag)
    files = []
    for path in paths:
        files.append(open_files_in_path(path))

    # now we run the algorithm

    # now we start extracting features of different types
    # COUNTS OF GENERAL THINGS (like words, different words, average lengths, number of sentences ecc.)
    resultFeatures += extract_general_features(files, min_frequency_ratio_general)

    # n-GRAM COUNTS
    resultFeatures += extract_gram_features(files,  min_frequency_ratio_grams, min_count_per_length)

    # n-POS COUNTS
    if WITH_POS:
        resultFeatures += extract_pos_features(files, min_frequency_ratio_grams, min_count_per_length)

    # returns the list of features
    return resultFeatures


def extract_general_features(files, min_frequency_ratio):

    resultFeatures = []
    classes = len(files)
    documentCounts = []
    for file in files:
        documentCounts.append(len(file))

    # average count of sentences per document
    sentenceCounts = []
    for c in range(classes):
        sentenceCounts.append(0)
        for item in files[c]:
            sentenceCounts[c] += len(item)

    averageSentenceCounts = []
    for c in range(classes):
        averageSentenceCounts.append(sentenceCounts[c]/documentCounts[c])

    # if at least one of them is statistically different, then I take that feature.
    VALIDFEATURE = False
    for c1 in range(classes-1):
        for c2 in range(c1+1,classes):
            if (averageSentenceCounts[c1] / averageSentenceCounts[c2]) >= min_frequency_ratio or\
                (averageSentenceCounts[c2] / averageSentenceCounts[c1]) >= min_frequency_ratio:
                VALIDFEATURE = True
                break
            if VALIDFEATURE:
                break

    if VALIDFEATURE:
        resultFeatures.append("COUNT SENTENCES")

    # average count of tokens per document/sentence
    wordCounts = []
    for c in range(classes):
        wordCounts.append(0)
        for item in files[c]:
            for line in item:
                wordCounts[c] += len(line)

    averageWordCountPerDocument = []
    for c in range(classes):
        averageWordCountPerDocument.append(wordCounts[c]/documentCounts[c])

    VALIDFEATURE = False
    for c1 in range(classes-1):
        for c2 in range(c1+1,classes):
            if (averageWordCountPerDocument[c1] / averageWordCountPerDocument[c2]) >= min_frequency_ratio or\
                (averageWordCountPerDocument[c2] / averageWordCountPerDocument[c1]) >= min_frequency_ratio:
                VALIDFEATURE = True
                break
            if VALIDFEATURE:
                break

    if VALIDFEATURE:
        resultFeatures.append("COUNT WORDS DOCUMENT")

    # sentences now
    averageWordCountPerSentence = []
    for c in range(classes):
        averageWordCountPerSentence.append(averageWordCountPerDocument[c]/averageSentenceCounts[c])

    VALIDFEATURE = False
    for c1 in range(classes-1):
        for c2 in range(c1+1,classes):
            if (averageWordCountPerSentence[c1] / averageWordCountPerSentence[c2]) >= min_frequency_ratio or\
                (averageWordCountPerSentence[c2] / averageWordCountPerSentence[c1]) >= min_frequency_ratio:
                VALIDFEATURE = True
                break
            if VALIDFEATURE:
                break

    if VALIDFEATURE:
        resultFeatures.append("COUNT WORDS SENTENCE")

    return resultFeatures


def extract_gram_features(files, min_frequency_ratio, min_count_per_length):
    up_to_what_gram = len(min_count_per_length)
    resultFeatures = []

    classes = len(files)
    documentCounts = []
    for file in files:
        documentCounts.append(len(file))

    # here I only want the words without the POS tag
    onlyWords = []
    for file in files:
        onlyWords.append(extract_only_element(file,0))

    for n in range(1,up_to_what_gram+1):
        wordsGrams = []
        for onlyWord in onlyWords:
            wordsGrams.append(ngrams_collection(onlyWord,n))

        wordsCounts = []
        for wordsGram in wordsGrams:
            wordsCounts.append(count_appearances(wordsGram))

        for c1 in range(classes):
            for c, ng in sorted(((c, ng) for ng, c in wordsCounts[c1].items()), reverse=True):
                # as it is sorted, the first time we have c< min_c, we know we won't find any more interesting features
                if c < min_count_per_length[n-1] : break
                for c2 in range(classes):
                    if c1 == c2: continue #of course I don't check for the same classes

                    cother = wordsCounts[c2][ng]
                    if cother == 0 or c/cother >= min_frequency_ratio:
                        newFeature = "COUNT GRAM " + str(n)
                        for i in range(n):
                            newFeature += " " + ng[i]
                        resultFeatures.append(newFeature)

    # There might be many repetitions, so I transform it into a set and then into a list again
    return sorted(set(resultFeatures))


def extract_pos_features(files, min_frequency_ratio, min_count_per_length):
    up_to_what_gram = len(min_count_per_length)
    resultFeatures = []

    classes = len(files)
    documentCounts = []
    for file in files:
        documentCounts.append(len(file))

    # here I only want the words without the POS tag
    onlyWords = []
    for file in files:
        onlyWords.append(extract_only_element(file,1))

    for n in range(1,up_to_what_gram+1):
        wordsGrams = []
        for onlyWord in onlyWords:
            wordsGrams.append(ngrams_collection(onlyWord,n))

        wordsCounts = []
        for wordsGram in wordsGrams:
            wordsCounts.append(count_appearances(wordsGram))

        for c1 in range(classes):
            for c, ng in sorted(((c, ng) for ng, c in wordsCounts[c1].items()), reverse=True):
                if c < min_count_per_length[n-1] : break
                for c2 in range(classes):
                    if c1 == c2: continue #of course I don't check for the same classes

                    cother = wordsCounts[c2][ng]
                    if cother == 0 or c/cother >= min_frequency_ratio:
                        newFeature = "COUNT POS " + str(n)
                        for i in range(n):
                            newFeature += " " + ng[i]
                        resultFeatures.append(newFeature)

    # There might be many repetitions, so I transform it into a set and then into a list again
    return sorted(set(resultFeatures))


# NOW WE COMPUTE THE FEATURES, GIVEN A LIST OF FEATURES TO COMPUTE

def compute_features_of_document(document,features,up_to_what_gram):
    # document holds the files

    # Here I hold the grams and pos counts
    speakerStatistics = {}

    # Here I hold the count of sentences and words
    generalStatistics = defaultdict(int)



    generalStatistics["SENTENCES"] += len(document)

    for line in document:
        generalStatistics["WORDS"] += len(line)

    # here I only want the words without the POS tag
    # note: it is a list of documents (of size 1)
    onlyWords = extract_only_element([document],0)
    onlyPOS = extract_only_element([document],1)


    for n in range(1,up_to_what_gram+1):
        wordsGram = ngrams_collection(onlyWords,n)
        wordsCount = count_appearances(wordsGram)
        speakerStatistics["GRAM "+str(n)] = wordsCount

        posGram = ngrams_collection(onlyPOS,n)
        posCount = count_appearances(posGram)
        speakerStatistics["POS "+str(n)] = posCount

    # now I can iterate over every feature to compute its result
    documentResult = []
    for feature in features:
        splitF = feature.split()
        if splitF[0] == 'COUNT':
            if splitF[1] == 'SENTENCES':
                documentResult.append(generalStatistics["SENTENCES"])
            elif splitF[1] == 'WORDS':
                if splitF[2] == 'DOCUMENT':
                    documentResult.append(generalStatistics["WORDS"])
                elif splitF[2] == 'SENTENCE':
                    documentResult.append(generalStatistics["WORDS"]/generalStatistics["SENTENCES"])
            elif splitF[1] == 'GRAM':
                documentResult.append(speakerStatistics["GRAM "+ splitF[2]][tuple(splitF[3:])])
            elif splitF[1] == 'POS':
                documentResult.append(speakerStatistics["POS "+ splitF[2]][tuple(splitF[3:])])

    return documentResult


def compute_features(corpusPath,features,up_to_what_gram):
    """

    :param corpusPath: the path of the corpus we are going where we are going to compute
    :param features: the string representation of the features we want to compute
    :return: a list of vectors, each of them representing the value of the features computed for one file
    """

    resultComputedFeatures = []

    corpusFiles = open_files_in_path(corpusPath)

    for document in corpusFiles:
        resultComputedFeatures.append(compute_features_of_document(document,features,up_to_what_gram))

    return resultComputedFeatures

