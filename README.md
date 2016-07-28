##  Project Strcture

In this file it is briefly described what each of the project files contains and does:

* Files "feature_extraction_...py" are the libraries
* Files "main_....py" are some sample usages of these libraries

* "feature_extraction_transcript.py" is used for the feature extraction of different speakers for the transcript dataset. "main_transcript_example.py" is an example usage. The real testing has been performed with a cross validation, with another script.

* "feature_extraction_topic.py" is used for the feature extraction regarding the topic classification for the transcript dataset, focusing on students only. "main_topic_example.py" is an example usage. The real testing has again been performed with a cross validation, with another script.

* "feature_extraction_movie.py" is used for the feature extraction regarding the movie sentiment reviews of the movie data set. "main_movie.py" is the script that I actually used to create train and test values to feed the libsvm classifier with KNIME.

* "preprocessingDialogue.py" is used to perform the preprocessing procedure described in the report on the transcripts dataset

* "LOOCV.py" is used to perform the leave one out crossvalidation on the already preprocessed transcripts dataset

* "preprocessingReview.py" is used to perform the preprocessing procedure described in the report on the movie reviews dataset"

* "topicExtraction.py" is used to extract the students sentences relative to the various topics from the transcripts dataset

* "CrossValidationTopics.py" is used to perform the 10-fold crossvalidation on the various students sentences divided by topic, extracted from the transcripts dataset already preprocessed

* "sentiment_threshold.py" is used to compute the threshold to determine the performance for each dialogue(positive or negative)

* "sentiment_score_duration_computation.py" is used to compute the average gain for each dialogue in the transcript dataset and to determine the performance label according to an already estimated threshold 