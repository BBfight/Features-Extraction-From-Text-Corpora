# FILE FOR TRANSCRIPT DATA SET FEATURE EXTRACTION

from functools import reduce
import os
import glob
from collections import defaultdict

TESTING = False


def split_pos(element):
    temp = element.split('/')
    return (temp[0],temp[1])


def speaker_equality(sp1,sp2):
    """sp1 must be the one we want to find (so can be ALL)

        sp2 cannot be ALL
    """

    return True if sp1 == 'ALL' else sp1 == sp2


def open_files_in_path(path):
    # creating a list to save the files
    txtFiles = []
    os.chdir(path)
    for file in glob.glob("*.txt"):
        txtFiles.append(open(file, "r"))
    # creating a list (file) of lists (lines) of couples (tag,sentence)
    # where a sentence is a list of couples (word,tag)
    linesPerDocument = []
    for txtFile in txtFiles:
        lines = []
        for line in txtFile.readlines():
            wordsPerLine = line.split()
            if len(wordsPerLine) > 1:
                lines.append((wordsPerLine[0], list(map(split_pos,wordsPerLine[1:]))))
        linesPerDocument.append(lines)
        # closing the file
        txtFile.close()

    # returning the newly created list
    return linesPerDocument


def find_speakers(file):
    speakerSet = set(line[0] for line in file)
    return list(speakerSet)


def ngrams_sentence(words, n=2, padding=True):
    """Compute n-grams with optional padding"""

    pad = [] if not padding else ['<S>']*(n-1)
    grams = pad + words + pad
    return list( (tuple(grams[i:i+n]) for i in range(0, len(grams) - (n - 1))))


def ngrams_file(file, n=2, padding=True):
    return list(reduce(lambda x,y: x + y,list(map(lambda x: ngrams_sentence(x,n,padding),file))))


def ngrams_collection(collection, n=2, padding=True):
    return reduce(lambda x,y: x + y,list(map(lambda x: ngrams_file(x,n,padding),collection)))


def count_appearances(collection):
    counts = defaultdict(int)
    for ng in collection:
        counts[ng] += 1
    return counts


def extract_only_element(collection,element = 0):
    return list(map(lambda x:
                    list(map(lambda y:
                             list(map(lambda z: z[element],y)),x)),collection))


def extract_features(pos_path, neg_path, min_frequency_ratio_general,min_frequency_ratio_grams, min_count,up_to_what_gram,WITH_POS = True):
    """Extracts relevant features from the datasets contained in the paths

        input:
            pos_path: the path where the positive training set is
            neg_path: the path where the negative training set is
            min_frequency_ratio: the minimum value accepted for FEA1/FEA2
                (eg. the count of a unigram in neg divided by the count of that unigram in pos must be at least min_frequency_ratio to be considered)
            min_count: the minimum value accepted for FEA
                (eg. the count of a unigram in pos must be at least min_count to be considered)

        returns:
            a list of strings, each of them representing a feature
    """
    # our result will be a list of features
    resultFeatures = []

    # opens all the pos and neg files, parses them by line and puts them into 2 different lists (files) of lists (lines) of couples (SPEAKER_TAG, text),
    # then closes them. "text" is a list of couples (word,tag)
    posFiles = open_files_in_path(pos_path)
    negFiles = open_files_in_path(neg_path)

    # we want to extract features for different partitions: ALL & type specific (STU & TEA in our corpus)
    # to do that we parse a file (one should be enough) and retrieve the unique starting tags (STU & TEA in our corpus)
    oneRandomFile = posFiles[0]  # it's a random file...
    speakerTags = find_speakers(oneRandomFile)  # gives me STU & TEA
    speakerTags.append('ALL')  # I add ALL, which means I consider everything
    # now we run the algorithm for these 3 different tags
    for speaker in speakerTags:
        # we extract and concatenate the features extracted for speaker
        resultFeatures += extract_features_of_speaker(posFiles, negFiles, speaker, min_frequency_ratio_general,min_frequency_ratio_grams, min_count,up_to_what_gram,WITH_POS)

    # there might also be some features that I want to extract by considering the fact that more tags exist.
    resultFeatures += extract_features_between_speakers(posFiles, negFiles, speakerTags[:-1], min_frequency_ratio_general, min_count)

    # returns the list of features
    return resultFeatures


def filter_files_by_tag(files, speaker):
    if speaker == 'ALL':
        return list(map(lambda x: list(map(lambda z: z[1], x)), files))
    return list(map(
        lambda x: list(map(lambda z: z[1],
            filter(lambda y: y[0] == speaker, x))), files))


def extract_features_of_speaker(posFiles, negFiles, speaker, min_frequency_ratio_general,min_frequency_ratio_grams, min_count,up_to_what_gram,WITH_POS = True):
    """Extracts relevant features from the datasets contained in pos/neg Files regarding speaker


        input:
            pos_path: the path where the positive training set is
            neg_path: the path where the negative training set is
            speaker: the ID of the speaker we will consider
            min_frequency_ratio: the minimum value accepted for FEA1/FEA2
                (eg. the count of a unigram in neg divided by the count of that unigram in pos must be at least min_frequency_ratio to be considered)
            min_count: the minimum value accepted for FEA
                (eg. the count of a unigram in pos must be at least min_count to be considered)

        returns:
            a list of strings, each of them representing a feature

        features must be parsable! so an output feature must follow the style:
            "speaker TYPE_OF_FEATURE OBJECT_OF_FEATURE [ARGS*]"
            TYPE_OF_FEATURE: COUNT, FREQUENCY, EXISTANCE...
            OBJECT_OF_FEATURE: GRAM (ARGS: number (like 2), name1, name2,...), POS (ARGS as before)
            ex: "STU COUNT GRAM 2 ok then", "ALL FREQUENCY POS 2 N V"
    """

    # our result will be a list of features
    resultFeatures = []

    # we want to only consider the lines starting with tag speaker. So we get rid of all the rest
    posRelevantFiles = filter_files_by_tag(posFiles, speaker)
    negRelevantFiles = filter_files_by_tag(negFiles, speaker)
    # now pos/neg RelevantFiles are a list (files) of lists (lines) of text [it will have to be tokenized & stuff]

    # now we start extracting features of different types
    # COUNTS OF GENERAL THINGS (like words, different words, average lenghts, number of sentences ecc.)
    resultFeatures += extract_general_features(posRelevantFiles, negRelevantFiles, min_frequency_ratio_general, min_count,
                                               speaker)  # speaker is needed to know the first word of the features' names

    # n-GRAM COUNTS/FREQUENCIES/EXISTANCE
    resultFeatures += extract_gram_features(posRelevantFiles, negRelevantFiles,  min_frequency_ratio_grams, min_count,up_to_what_gram, speaker)

    # n-POS COUNTS/FREQUENCIES/EXISTANCE
    if WITH_POS:
        resultFeatures += extract_pos_features(posRelevantFiles, negRelevantFiles, min_frequency_ratio_grams, min_count, up_to_what_gram, speaker)

    # returns the list of features
    return resultFeatures


def extract_features_between_speakers(posFiles, negFiles, speakerTags, min_frequency_ratio, min_count):
    """Does nothing atm"""

    resultFeatures = []
    posDocumentCount = len(posFiles)
    negDocumentCount = len(negFiles)
    # holds the files for every speaker
    posDocs ={}
    negDocs = {}

    # holds counters for every speaker
    posSentences = defaultdict(int)
    negSentences = defaultdict(int)
    posWords = defaultdict(int)
    negWords = defaultdict(int)

    for speaker in speakerTags:
        posDocs[speaker] = filter_files_by_tag(posFiles, speaker)
        negDocs[speaker] = filter_files_by_tag(negFiles, speaker)

        for item in posDocs[speaker]:
            posSentences[speaker] += len(item)
            for line in item:
                posWords[speaker] += len(line)

        for item in negDocs[speaker]:
            negSentences[speaker] += len(item)
            for line in item:
                negWords[speaker] += len(line)

    for speaker1 in range(0,len(speakerTags)-1):
        for speaker2 in range(speaker1+1,len(speakerTags)):
            #sentences per doc
            avgPosSentS1 = posSentences[speakerTags[speaker1]] / posDocumentCount
            avgPosSentS2 = posSentences[speakerTags[speaker2]] / posDocumentCount
            avgNegSentS1 = negSentences[speakerTags[speaker1]] / negDocumentCount
            avgNegSentS2 = negSentences[speakerTags[speaker2]] / negDocumentCount

            # words per doc
            avgPosWordDocS1 = posWords[speakerTags[speaker1]] / posDocumentCount
            avgPosWordDocS2 = posWords[speakerTags[speaker2]] / posDocumentCount
            avgNegWordDocS1 = negWords[speakerTags[speaker1]] / negDocumentCount
            avgNegWordDocS2 = negWords[speakerTags[speaker2]] / negDocumentCount

            # words per sentence
            avgPosWordSenS1 = avgPosWordDocS1 / avgPosSentS1
            avgPosWordSenS2 = avgPosWordDocS2 / avgPosSentS2
            avgNegWordSenS1 = avgNegWordDocS1 / avgNegSentS1
            avgNegWordSenS2 = avgNegWordDocS2 / avgNegSentS2

            # sentences S1/S2
            if (avgPosSentS1/avgPosSentS2) / (avgNegSentS1/avgNegSentS2) >= min_frequency_ratio\
                    or (avgNegSentS1/avgNegSentS2) / (avgPosSentS1/avgPosSentS2) >= min_frequency_ratio:
                if TESTING:
                    print("Ratio sentences "+ speakerTags[speaker1]+" "+speakerTags[speaker2]+" is "+ str(avgPosSentS1/avgPosSentS2)\
                                                                                             + " vs " + str(avgNegSentS1/avgNegSentS2))
                resultFeatures.append("MIX RATIO SENTENCES " +speakerTags[speaker1]+" "+speakerTags[speaker2])
            """
            # sentences S2/S1
            if (avgPosSentS2/avgPosSentS1) / (avgNegSentS2/avgNegSentS1) >= min_frequency_ratio\
                    or (avgNegSentS2/avgNegSentS1) / (avgPosSentS2/avgPosSentS1) >= min_frequency_ratio:
                resultFeatures.append("MIX RATIO SENTENCES " +speakerTags[speaker2]+" "+speakerTags[speaker1])
            """
            # words per doc S1/S2
            if (avgPosWordDocS1/avgPosWordDocS2) / (avgNegWordDocS1/avgNegWordDocS2) >= min_frequency_ratio\
                    or (avgNegWordDocS1/avgNegWordDocS2) / (avgPosWordDocS1/avgPosWordDocS2) >= min_frequency_ratio:
                resultFeatures.append("MIX RATIO WORDS DOCUMENT " +speakerTags[speaker1]+" "+speakerTags[speaker2])
            """
            # words per doc S2/S1
            if (avgPosWordDocS2/avgPosWordDocS1) / (avgNegWordDocS2/avgNegWordDocS1) >= min_frequency_ratio\
                    or (avgNegWordDocS2/avgNegWordDocS1) / (avgPosWordDocS2/avgPosWordDocS1) >= min_frequency_ratio:
                resultFeatures.append("MIX RATIO WORDS DOCUMENT " +speakerTags[speaker2]+" "+speakerTags[speaker1])
            """
            # words per sentence S1/S2
            if (avgPosWordSenS1/avgPosWordSenS2) / (avgNegWordSenS1/avgNegWordSenS2) >= min_frequency_ratio\
                    or (avgNegWordSenS1/avgNegWordSenS2) / (avgPosWordSenS1/avgPosWordSenS2) >= min_frequency_ratio:
                resultFeatures.append("MIX RATIO WORDS SENTENCE " +speakerTags[speaker1]+" "+speakerTags[speaker2])
            """
            # words per sentence S2/S1
            if (avgPosWordSenS2/avgPosWordSenS1) / (avgNegWordSenS2/avgNegWordSenS1) >= min_frequency_ratio\
                    or (avgNegWordSenS2/avgNegWordSenS1) / (avgPosWordSenS2/avgPosWordSenS1) >= min_frequency_ratio:
                resultFeatures.append("MIX RATIO WORDS SENTENCE " +speakerTags[speaker2]+" "+speakerTags[speaker1])
            """
    return resultFeatures


def extract_general_features(posRelevantFiles, negRelevantFiles, min_frequency_ratio, min_count,
                                               speaker):

    resultFeatures = []
    posDocumentCount = len(posRelevantFiles)
    negDocumentCount = len(negRelevantFiles)

    # average count of sentences per document
    posSentenceCount = 0
    negSentenceCount = 0

    for item in posRelevantFiles:
        posSentenceCount += len(item)

    for item in negRelevantFiles:
        negSentenceCount += len(item)

    posAverageSentenceCount = posSentenceCount / posDocumentCount
    negAverageSentenceCount = negSentenceCount / negDocumentCount

    if TESTING:
        print("Average number of sentences for "+ speaker+ " in pos is "+ str(posAverageSentenceCount)+
              " while in neg is "+ str(negAverageSentenceCount))
    posSenOverNeg = posAverageSentenceCount / negAverageSentenceCount
    negSenOverPos = negAverageSentenceCount / posAverageSentenceCount

    if TESTING:
        print("Percentages: POS/NEG: "+ str(posSenOverNeg)+"; NEG/POS: "+ str(negSenOverPos))

    if posSenOverNeg >= min_frequency_ratio or negSenOverPos >= min_frequency_ratio:
        resultFeatures.append(speaker + " COUNT SENTENCES")

    # average count of tokens per document/sentence
    posWordCount = 0
    negWordCount = 0

    for item in posRelevantFiles:
        for line in item:
            posWordCount += len(line)

    for item in negRelevantFiles:
        for line in item:
            negWordCount += len(line)

    posAverageWordCountPerDocument = posWordCount / posDocumentCount
    negAverageWordCountPerDocument = negWordCount / negDocumentCount

    if TESTING:
        print("Average number of words per document for "+ speaker+ " in pos is "+ str(posAverageWordCountPerDocument)+
              " while in neg is "+ str(negAverageWordCountPerDocument))
    posWordPerDocOverNeg = posAverageWordCountPerDocument / negAverageWordCountPerDocument
    negWordPerDocOverPos = negAverageWordCountPerDocument / posAverageWordCountPerDocument

    if TESTING:
        print("Percentages: POS/NEG: "+ str(posWordPerDocOverNeg)+"; NEG/POS: "+ str(negWordPerDocOverPos))

    if posWordPerDocOverNeg >= min_frequency_ratio or negWordPerDocOverPos >= min_frequency_ratio:
        resultFeatures.append(speaker + " COUNT WORDS DOCUMENT")

    posAverageWordCountPerSentence = posAverageWordCountPerDocument / posAverageSentenceCount
    negAverageWordCountPerSentence = negAverageWordCountPerDocument / negAverageSentenceCount

    if TESTING:
        print("Average number of words per sentence for "+ speaker+ " in pos is "+ str(posAverageWordCountPerSentence)+
              " while in neg is "+ str(negAverageWordCountPerSentence))
    posWordPerSenOverNeg = posAverageWordCountPerSentence / negAverageWordCountPerSentence
    negWordPerSenOverPos = negAverageWordCountPerSentence / posAverageWordCountPerSentence

    if TESTING:
        print("Percentages: POS/NEG: "+ str(posWordPerSenOverNeg)+"; NEG/POS: "+ str(negWordPerSenOverPos))

    if posWordPerSenOverNeg >= min_frequency_ratio or negWordPerSenOverPos >= min_frequency_ratio:
        resultFeatures.append(speaker + " COUNT WORDS SENTENCE")

    return resultFeatures


def extract_gram_features(posRelevantFiles, negRelevantFiles, min_frequency_ratio, min_count, up_to_what_gram, speaker):

    resultFeatures = []
    posDocumentCount = len(posRelevantFiles)
    negDocumentCount = len(negRelevantFiles)

    # here I only want the words without the POS tag
    posOnlyWords = extract_only_element(posRelevantFiles,0)
    negOnlyWords = extract_only_element(negRelevantFiles,0)

    for n in range(1,up_to_what_gram+1):
        posWordsGram = ngrams_collection(posOnlyWords,n)
        negWordsGram = ngrams_collection(negOnlyWords,n)

        posWordsCount = count_appearances(posWordsGram)
        negWordsCount = count_appearances(negWordsGram)

        for c, ng in sorted(((c, ng) for ng, c in posWordsCount.items()), reverse=True):
            if c < min_count : break

            cneg = negWordsCount[ng]
            if cneg == 0 or c/cneg >= min_frequency_ratio:
                if TESTING:
                    print(str(n)+"-gram word "+str(ng)+" appears "+str(c)+" times in positive docs"
                          +" and "+str(cneg)+" times in negative docs: frequency of "+str(c/cneg))
                newFeature = speaker + " COUNT GRAM " + str(n)
                for i in range(n):
                    newFeature += " " + ng[i]
                resultFeatures.append(newFeature)

        for c, ng in sorted(((c, ng) for ng, c in negWordsCount.items()), reverse=True):
            if c < min_count : break

            cpos = posWordsCount[ng]
            if cpos == 0 or c/cpos >= min_frequency_ratio:
                if TESTING:
                    print(str(n)+"-gram word "+str(ng)+" appears "+str(c)+" times in negative docs"
                          +" and "+str(cpos)+" times in positive docs: frequency of "+str(c/cpos))
                newFeature = speaker + " COUNT GRAM " + str(n)
                for i in range(n):
                    newFeature += " " + ng[i]
                resultFeatures.append(newFeature)

    return resultFeatures

def extract_pos_features(posRelevantFiles, negRelevantFiles, min_frequency_ratio, min_count, up_to_what_gram, speaker):

    resultFeatures = []
    posDocumentCount = len(posRelevantFiles)
    negDocumentCount = len(negRelevantFiles)

    # here I only want the POS tags without the words
    posOnlyWords = extract_only_element(posRelevantFiles,1)
    negOnlyWords = extract_only_element(negRelevantFiles,1)

    for n in range(1,up_to_what_gram+1):
        posWordsGram = ngrams_collection(posOnlyWords,n)
        negWordsGram = ngrams_collection(negOnlyWords,n)

        posWordsCount = count_appearances(posWordsGram)
        negWordsCount = count_appearances(negWordsGram)

        for c, ng in sorted(((c, ng) for ng, c in posWordsCount.items()), reverse=True):
            if c < min_count : break

            cneg = negWordsCount[ng]
            if cneg == 0 or c/cneg >= min_frequency_ratio:
                newFeature = speaker + " COUNT POS " + str(n)
                for i in range(n):
                    newFeature += " " + ng[i]
                resultFeatures.append(newFeature)

        for c, ng in sorted(((c, ng) for ng, c in negWordsCount.items()), reverse=True):
            if c < min_count : break

            cpos = posWordsCount[ng]
            if cpos == 0 or c/cpos >= min_frequency_ratio:
                newFeature = speaker + " COUNT POS " + str(n)
                for i in range(n):
                    newFeature += " " + ng[i]
                resultFeatures.append(newFeature)

    return resultFeatures


# NOW WE COMPUTE THE FEATURES, GIVEN A LIST OF FEATURES TO COMPUTE

def filter_single_file_by_tag(file, speaker):
    if speaker == 'ALL':
        return list(map(lambda z: z[1], file))
    return list(map(lambda z: z[1],
            filter(lambda y: y[0] == speaker, file)))


def compute_features_of_document(document,features,speakerTags,up_to_what_gram):
    # Holds the files for every speaker
    speakerDocs ={}

    # Here I hold the grams and pos counts
    speakerStatistics = {}

    # Here I hold the count of sentences and words
    speakerGeneralStatistics = defaultdict(int)

    for speaker in speakerTags:
        speakerDocs[speaker] = filter_single_file_by_tag(document, speaker)


        speakerGeneralStatistics[speaker+" SENTENCES"] += len(speakerDocs[speaker])

        for line in speakerDocs[speaker]:
            speakerGeneralStatistics[speaker+" WORDS"] += len(line)

        # here I only want the words without the POS tag
        # note: it is a list of documents (of size 1)
        onlyWords = extract_only_element([speakerDocs[speaker]],0)
        onlyPOS = extract_only_element([speakerDocs[speaker]],1)


        for n in range(1,up_to_what_gram+1):
            wordsGram = ngrams_collection(onlyWords,n)
            wordsCount = count_appearances(wordsGram)
            speakerStatistics[speaker+" GRAM "+str(n)] = wordsCount

            posGram = ngrams_collection(onlyPOS,n)
            posCount = count_appearances(posGram)
            speakerStatistics[speaker+" POS "+str(n)] = posCount

    # now I can iterate over every feature to compute its result
    documentResult = []
    for feature in features:
        splitF = feature.split()
        if splitF[0] == 'MIX':
            if splitF[1] == 'RATIO':
                if splitF[2] == 'SENTENCES':
                    documentResult.append(speakerGeneralStatistics[splitF[3]+" SENTENCES"]/speakerGeneralStatistics[splitF[4]+" SENTENCES"])
                elif splitF[2] == 'WORDS':
                    if splitF[3] == 'DOCUMENT':
                        documentResult.append(speakerGeneralStatistics[splitF[4]+" WORDS"]/speakerGeneralStatistics[splitF[5]+" WORDS"])
                    elif splitF[3] == 'SENTENCE':
                        documentResult.append((speakerGeneralStatistics[splitF[4]+" WORDS"]/speakerGeneralStatistics[splitF[4]+" SENTENCES"])
                                              /(speakerGeneralStatistics[splitF[5]+" WORDS"]/speakerGeneralStatistics[splitF[5]+" SENTENCES"]))
        else:
            if splitF[1] == 'COUNT':
                if splitF[2] == 'SENTENCES':
                    documentResult.append(speakerGeneralStatistics[splitF[0]+" SENTENCES"])
                elif splitF[2] == 'WORDS':
                    if splitF[3] == 'DOCUMENT':
                        documentResult.append(speakerGeneralStatistics[splitF[0]+" WORDS"])
                    elif splitF[3] == 'SENTENCE':
                        documentResult.append(speakerGeneralStatistics[splitF[0]+" WORDS"]/speakerGeneralStatistics[splitF[0]+" SENTENCES"])
                elif splitF[2] == 'GRAM':
                    documentResult.append(speakerStatistics[splitF[0]+" GRAM "+ splitF[3]][tuple(splitF[4:])])
                elif splitF[2] == 'POS':
                    documentResult.append(speakerStatistics[splitF[0]+" POS "+ splitF[3]][tuple(splitF[4:])])


    return documentResult


def compute_features(corpusPath,features,up_to_what_gram):
    """

    :param corpusPath: the path of the corpus we are going where we are going to compute
    :param features: the string representation of the features we want to compute
    :return: a list of vectors, each of them representing the value of the features computed for one file
    """

    # We need the data sets for every speaker

    resultComputedFeatures = []

    corpusFiles = open_files_in_path(corpusPath)
    speakerTags = find_speakers(corpusFiles[0])  # gives me STU & TEA
    speakerTags.append('ALL')

    for document in corpusFiles:
        resultComputedFeatures.append(compute_features_of_document(document,features,speakerTags,up_to_what_gram))

    return resultComputedFeatures


