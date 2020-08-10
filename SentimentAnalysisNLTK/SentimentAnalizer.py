import csv,os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
Filename=os.path.join("DATA","DataoftheHeart_LakeDistrictsurvey.csv")
with open(Filename, mode='r') as infile:
    reader = list(csv.DictReader(infile))
    desiredkey=list(reader[0].keys())[17]
    for row in reader:
        text=row.get(desiredkey,"N/A")

        # Calling the polarity_scores method on sid and passing in the message_text outputs a dictionary with negative, neutral, positive, and compound scores for the input text
        scores = sid.polarity_scores(message_text)

        # Here we loop through the keys contained in scores (pos, neu, neg, and compound scores) and print the key-value pairs on the screen

        for key in sorted(scores):
            print('{0}: {1}, '.format(key, scores[key]), end='')