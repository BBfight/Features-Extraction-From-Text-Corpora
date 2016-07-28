__author__ = 'ettore'
# THIS FILE IS NOT USED. IT IS JUST TO EXPLAIN HOW TO USE feature_extraction_transcript.py
from feature_extraction_transcript import *
import csv

# Here you put the paths of your files
posPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/train/positive"
negPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/train/negative"
posTestPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/test/positive"
negTestPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/test/negative"
trainPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/train/"
testPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/test/"

# posPath & negPath are the paths of the training sets
# the minimum ratio POS/NEG (NEG/POS) for general stuff
min_frequency_ratio_general = 10
# the minimum ratio POS/NEG (NEG/POS) for grams and POS stuff
min_frequency_ratio_grams = 1.2
# the minimum number of occurences of any gram/POS to be considered
min_count = 100
# the n-gram (n-POS) are computed up to n = up_to_what_gram
up_to_what_gram = 3


for fold in range (1,4):
    # Here you compute the features (save them somewhere if needed)

    features = extract_features(posPath+str(fold), negPath+str(fold),min_frequency_ratio_general,min_frequency_ratio_grams, min_count,up_to_what_gram)

    # If you want to see the features extracted
    for feat in features:
        print(feat)

    # computes the features for all the sets needed.
    # TRAIN
    posTrainValues = compute_features(posPath+str(fold),features,up_to_what_gram)
    negTrainValues = compute_features(negPath+str(fold),features,up_to_what_gram)
    # TEST
    posTestValues = compute_features(posTestPath+str(fold),features,up_to_what_gram)
    negTestValues = compute_features(negTestPath+str(fold),features,up_to_what_gram)

    # Save the results somewhere, profit

    # If you want to see some results
    #for line in posTrainValues:
    #    print(line)


    with open(trainPath + "train"+str(fold)+".csv", 'w') as csvTrain:
        writer = csv.writer(csvTrain, delimiter = ',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in posTrainValues:
            writer.writerow(['POS']+line)#'1, '+str(line)[1:-1])
        for line in negTrainValues:
            writer.writerow(['NEG']+line)#'0, '+str(line)[1:-1])

    with open(testPath + "test"+str(fold)+".csv", 'w') as csvTest:
        writer = csv.writer(csvTest, delimiter = ',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in posTestValues:
            writer.writerow(['POS']+line)#'1, '+str(line)[1:-1])
        for line in negTestValues:
            writer.writerow(['NEG']+line)#'0, '+str(line)[1:-1])
