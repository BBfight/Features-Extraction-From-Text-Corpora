__author__ = 'Ben'

import pandas as pd
import matplotlib.pyplot as plt
import statistics as st

# reading the csv files with the average_gain and total_length of sessions
id_averageGain_totalLength = pd.DataFrame.from_csv("C:/Users/Ben/Google Drive/UIC/SNLP/Project/Datasets/transcripts dataset/id_average-gain_total_length.csv",
                                                   header=0, index_col=None)

# plotting the average-gain for students
plt.plot(id_averageGain_totalLength.id.index, id_averageGain_totalLength.average_gain)

# computing the mean of the average_gain
avg = st.mean(id_averageGain_totalLength.average_gain)
print('average is ', avg)

# computing the median of the average_gain
mdn = st.median(id_averageGain_totalLength.average_gain)
print('median is ',mdn)

# initialising counters for positive and negative instances according to the threshold chosen
npos_avg = 0 
nneg_avg = 0
npos_mdn = 0
nneg_mdn = 0

# checking how many positive and negative with avg as threshold
for index,row in id_averageGain_totalLength.iterrows():
        if row['average_gain'] >= avg:
            npos_avg += 1
        else:
            nneg_avg += 1
    
# checking how many positive and negative with mdn as threshold
for index,row in id_averageGain_totalLength.iterrows():
        if row['average_gain'] >= mdn:
            npos_mdn += 1
        else:
            nneg_mdn += 1    
 
print('Considering the average as threshold: ', npos_avg, 'positive, ', nneg_avg, ' negative')
print('Considering the median as threshold: ', npos_mdn, 'positive, ', nneg_mdn, ' negative')
