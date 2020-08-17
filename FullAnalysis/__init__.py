import nltk,gensim,os
import gensim.downloader as api
import shutil
import datetime

now = str(datetime.datetime.now())[:19]
now = now.replace(":","_")

path="/app/model/"
filename="model.wv"
os.makedirs(path, exist_ok = True)
path=api.load('glove-wiki-gigaword-50', return_path=True))
shutil.copy(path,os.path.join(path,filename))
nltk.download('vader_lexicon')