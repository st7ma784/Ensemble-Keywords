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