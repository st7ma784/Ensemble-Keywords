import nltk,gensim,os
import gensim.downloader as api
import shutil
import datetime

now = str(datetime.datetime.now())[:19]
now = now.replace(":","_")

path="/app/UTILS/model/"
filename="model.gz"
os.makedirs(path, exist_ok = True)
inpath=api.load('glove-wiki-gigaword-50', return_path=True)
shutil.copy(inpath,os.path.join(path,filename))
print(os.path.join(path,filename))
nltk.download('vader_lexicon')


from gensim.models import doc2vec
from collections import namedtuple

# Load data
from gensim.test.utils import common_texts

# Train model (set min_count = 1, if you want the model to work with the provided example data set)
filename="sentsmodel.bin"
model = doc2vec.Doc2Vec(common_texts, size = 100, window = 300, min_count = 1, workers = 4)
model.save(os.path.join(path,filename))