__author__ = 'Ben'

import pandas as pd

#rading the initial csv files
session_topic_gain = pd.DataFrame.from_csv("C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/session_topic_gain.csv",
                                           header=0, index_col=None)

# saving the candidates set from the dataframe
candidates = set(session_topic_gain.id)

# setting the sentiment threshold
threshold = 0.25 #estimated in another script

# creating a data frame where to store values
final_csv = pd.DataFrame(columns=['id', 'average_gain', 'total_length','performance'])

# finding for every candidate the relative sessions and total length and the average gain
for candidate in candidates:
    total_length = 0
    total_gain = 0
    counter = 0
    for index,row in session_topic_gain.iterrows():
        if row['id'] == candidate:
            counter += 1
            total_length += row['length']
            total_gain += row['gain']            
    average_gain = float(total_gain/counter)
    # comparing average score and threshold in order to set the performance
    if average_gain > threshold:
        final_csv.loc[len(final_csv)+1] = [candidate,average_gain,total_length,'positive']
    else:
        final_csv.loc[len(final_csv)+1] = [candidate,average_gain,total_length,'negative']

# saving final dataframe to csv
final_csv.to_csv("C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/id_average-gain_total_length_performance.csv")
