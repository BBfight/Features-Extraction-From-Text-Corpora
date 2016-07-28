from feature_extraction_movie import *
import csv

# Here you put the paths of your files
posPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/movieReviews/v_0.2/train/pos"
negPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/movieReviews/v_0.2/train/neg"
posTestPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/movieReviews/v_0.2/test/pos"
negTestPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/movieReviews/v_0.2/test/neg"
trainPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/movieReviews/v_0.2/train/"
testPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/movieReviews/v_0.2/test/"

# posPath & negPath are the paths of the training sets
# the minimum ratio POS/NEG (NEG/POS) for general stuff
min_frequency_ratio_general = 1.1
# the minimum ratio POS/NEG (NEG/POS) for grams and POS stuff
min_frequency_ratio_grams = 1.1
# True means it extracts also POS features
WITH_POS = False

# the minimum number of occurrences of gram/POS per number of token considered
min_count_per_length = [1000,800]#,200]
# the n-gram (n-POS) are computed up to n = up_to_what_gram
up_to_what_gram = len(min_count_per_length)


# Here you compute the features (save them somewhere if needed)


features = extract_features([posPath, negPath],min_frequency_ratio_general,min_frequency_ratio_grams, min_count_per_length,WITH_POS)
# If you want to see the features extracted
for feat in features:
    print(feat)

# computes the features for all the sets needed.
# TRAIN
posTrainValues = compute_features(posPath,features,up_to_what_gram)
print("Computed pos train features.")
negTrainValues = compute_features(negPath,features,up_to_what_gram)
print("Computed neg train features.")
# TEST
posTestValues = compute_features(posTestPath,features,up_to_what_gram)
print("Computed pos test features.")
negTestValues = compute_features(negTestPath,features,up_to_what_gram)
print("Computed neg test features.")

# Save the results somewhere, profit

# If you want to see some results
#for line in posTrainValues:
    #print(line)

csv_name = "fr_"+str(min_frequency_ratio_grams)+"_GRAMS:"+str(up_to_what_gram)+"_POS:"+str(WITH_POS)
with open(trainPath + "train_"+csv_name+".csv", 'w') as csvTrain:
    writer = csv.writer(csvTrain, delimiter = ',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for line in posTrainValues:
        writer.writerow(line+['POS'])
    for line in negTrainValues:
        writer.writerow(line+['NEG'])

with open(testPath + "test_"+csv_name+".csv", 'w') as csvTest:
    writer = csv.writer(csvTest, delimiter = ',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for line in posTestValues:
        writer.writerow(line+['POS'])
    for line in negTestValues:
        writer.writerow(line+['NEG'])
