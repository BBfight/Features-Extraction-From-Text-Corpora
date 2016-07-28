#FILE NOT USED! IT JUST EXPLAINS HOW TO USE feature_extraction_topic.py
__author__ = 'ettore'

from feature_extraction_topic import *
import csv

# Here you put the paths of your files
#where you save the csv
trainPath = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/topics/"

listP = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/topics/list.txt"
stackP = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/topics/stack.txt"
treeP = "/Users/ettore/Documents/Università/SNLP/Projectzor/transcripts dataset/topics/tree.txt"

# posPath & negPath are the paths of the training sets
# the minimum ratio POS/NEG (NEG/POS) for grams and POS stuff
min_frequency_ratio_grams = 1.1
# True means it extracts also POS features
WITH_POS = False

# the minimum number of occurrences of gram/POS per number of token considered
min_count_per_length = [20]#,10]
# the n-gram (n-POS) are computed up to n = up_to_what_gram
up_to_what_gram = len(min_count_per_length)


# Here you compute the features (save them somewhere if needed)

features = extract_features([listP,stackP,treeP],min_frequency_ratio_grams, min_count_per_length,WITH_POS)
# If you want to see the features extracted
for feat in features:
    print(feat)

# computes the features for all the sets needed.
# TRAIN
listPValues = compute_features(listP,features,up_to_what_gram)
print("Computed listP features.")
stackPValues = compute_features(stackP,features,up_to_what_gram)
print("Computed stackP features.")
treePValues = compute_features(treeP,features,up_to_what_gram)
print("Computed treeP features.")

# Save the results somewhere, profit

# If you want to see some results
#for line in posTrainValues:
    #print(line)

csv_name = "fr_"+str(min_frequency_ratio_grams)+"_GRAMS:"+str(up_to_what_gram)+"_POS:"+str(WITH_POS)
with open(trainPath + "train_"+csv_name+".csv", 'w') as csvTrain:
    writer = csv.writer(csvTrain, delimiter = ',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for line in listPValues:
        writer.writerow(line+['LIST'])
    for line in stackPValues:
        writer.writerow(line+['STACK'])
    for line in treePValues:
        writer.writerow(line+['TREE'])