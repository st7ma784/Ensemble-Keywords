import csv,os
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
scores=list()
sid=SentimentIntensityAnalyzer()
Filename=os.path.join("DATA","DataoftheHeart_LakeDistrictsurvey.csv")
with open(Filename, mode='r') as infile:
    reader = list(csv.DictReader(infile))
    desiredkey=list(reader[0].keys())[17]

    for id,row in enumerate(reader):
        text=row.get(desiredkey,"N/A")
        scores.append(sid.polarity_scores(text))

totals=defaultdict(int)

for score in scores:
    for key,value in score.items():
        totals[key] += value
for key in totals:
    totals[key]=totals[key]/len(scores)
print(totals)