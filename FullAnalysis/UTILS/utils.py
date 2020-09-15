from __future__ import unicode_literals

import csv,os,random, datetime,dash, itertools, base64
from collections import OrderedDict,defaultdict
import numpy as np
from io import BytesIO,StringIO
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
import dash_core_components as dcc
import dash_html_components as html
from functools import partial
from itertools import repeat
import math
import numpy
import spacy
import tqdm
import networkx as nx
DEBUG=bool(os.environ['DEBUG'])
from spacy.tokens import Token
from spacy.matcher import Matcher
import os
from collections import OrderedDict,defaultdict

from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
class TextRank4Keyword():
    """Extract keywords from text"""
    wordtypes=["ADJ","ABSTNOUN","INTRANVERB","TRANVERB","ADV","PRPN","VERB","NOUN","INTJ"]

    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight
        with open(os.path.join("UTILS","DATA",'abstractnounlist.txt'),'r') as file:
            ABST=file.readlines()
        self.wordtypes=["ADJ","ABSTNOUN","INTRANVERB","TRANVERB","ADV","PRPN","VERB","NOUN","INTJ"]
        self.ABSTLIST=[word.lower().replace("\n","") for word in ABST]
        self.knowledgebase=set()
        #print(self.ABSTLIST)
    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    def check_verb(self, token):
        if token.pos_ == 'VERB':
            indirect_object = False
            direct_object = False
            for item in token.children:
                if(item.dep_ == "iobj" or item.dep_ == "pobj"):
                    indirect_object = True
                if (item.dep_ == "dobj" or item.dep_ == "dative"):
                    direct_object = True
            if indirect_object and direct_object:
                return 'TRANVERB'
            elif direct_object and not indirect_object:
                return 'TRANVERB'
            elif not direct_object and not indirect_object:
                return 'INTRANVERB'
            else:
                return 'VERB'
        elif token.pos_ == 'NOUN':
            #print(token.text)
            if token.text.lower() in self.ABSTLIST:
                #print("Found to be ABST" + token.text)
                return 'ABSTNOUN'
            else:
                return 'NOUN'
        else:
            return token.pos_
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos == could implement with matcher for more complex structures"""
        sentences = []
        lemma_tags = {"NNS", "NNPS","VBD","VBG","VBN","VBP","VBZ"}
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                pos=self.check_verb(token)
                #check for compound noun.
                if (pos in candidate_pos or token.tag_ in candidate_pos) and token.is_stop is False:

                    text=token.text    
                    if token.tag_ in lemma_tags:   #de pluralize nouns
                        text = token.lemma_
                    if lower is True:
                        selected_words.append(text.lower())
                    else:
                        selected_words.append(text)
            sentences.append(selected_words)
        return sentences
    
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    def reversetemplate(self,poem):
        sentences = []
        for sent in poem.sents:
            selected_words = []
            for token in sent:
                pos=self.check_verb(token)
                if pos in self.wordtypes:
                    text=pos 
                    selected_words.append(text.upper())
                else:
                    text=token.text
                    #if len(text)>1 or text=="a" or text=="I" or token.is_punct or token.is_stop: 
                    selected_words.append(text.lower())

            sentences.append(" ".join(selected_words))
        return "\n".join(sentences)

    def get_keywords(self, number=None):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        if number is not None:
            return list(node_weight.items())[:number]
        else:
            return list(node_weight.items())
    def getknowledge(self,doc):
        if DEBUG:
            print(doc.text)
            print("Dependencies", [(t.text, t.dep_, t.head.text) for t in  filter(lambda w: w.dep_ in ("compound","amod", "dobj"), doc)])
            print("WordTypes",[(self.check_verb(t),self.check_verb(t.head)) for t in  filter(lambda w: w.dep_ in ("compound","amod", "dobj"), doc)])
        self.knowledgebase=set((t.text, t.dep_, t.head.text) for t in doc)


        return set((t.text, t.dep_, t.head.text,self.check_verb(t),self.check_verb(t.head)) for t in doc)
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        #createTensorBOARD from vocab??


        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight

def buildTemplateFromText(rawpoem):
    poem=rawpoem.replace('(<a).*(>).*(</a>)', '')
    # poem = poem.replace('(&amp)', '')
    # poem = poem.replace('(&gt)', '')
    # poem = poem.replace('(&lt)', '')
    # poem = poem.replace('(\xa0)', ' ') 
    tr4w=TextRank4Keyword()
    template=tr4w.reversetemplate(nlp(poem))
    return template

def buildPoem(df,Column,poem,metaphor=1):
    wordtypes=TextRank4Keyword.wordtypes
    if DEBUG:
        print(wordtypes)
    newdf = df[~df[Column].isnull()]
    wordlist={wtype:createWordList(newdf[Column],wtype) for wtype in wordtypes}#:parralelproc(wordtypes,newdf[Column],createWordList)#create our wordlist
    wordgraph=pd.DataFrame()
    wordgraph["relations"]=buildknowledgebase(newdf[Column])
    wordgraph["start"]=wordgraph["relations"].apply(lambda x: x[0])
    wordgraph["target"]=wordgraph["relations"].apply(lambda x: x[2])
    wordgraph["edgetype"]=wordgraph["relations"].apply(lambda x: x[1])
    wordgraph["starttype"]=wordgraph["relations"].apply(lambda x: x[3])
    wordgraph["targettype"]=wordgraph["relations"].apply(lambda x: x[4])
    #wordgraph.drop("relations",axis=1)
    kg_df=pd.DataFrame({"source":wordgraph["start"],"target":wordgraph["target"],"edge":wordgraph["edgetype"]})
    G=nx.from_pandas_edgelist(kg_df, "source", "target",edge_attr=True, create_using=nx.MultiDiGraph())
    pos=nx.spring_layout(G)
    #nx.draw(G, with_labels=True, pos=pos, edge_cmap=plt.cm.Blues, )
    
    
    if metaphor==1:
        #poem=poem.split("\n")
        finished=[]
        Token.set_extension("used", default=False,force=True)
        doc=nlp(poem)
        out=[]
        matcher = Matcher(nlp.vocab)
        out=[token.text for token in doc]

        def Match(matcher, doc, i, matches):
            spans = [(start,end, doc[start : end]) for ent_id, start, end in matches]
            for (start,end,words) in spans:
                covered=0
                target=len(words)
                #print(" Output of {0} is {1}".format(words,BetterTraverseGraph(graphtuples,wordlist,words)))
                while covered<target:
                    found=TraverseGraph(wordgraph,wordlist,words[covered:target])
                    if covered+len(found)>target:
                        found=found[:target-covered]
                    out[start+covered:start+covered+len(found)]=found
                    covered+=len(found)
        pattern=[{"TEXT": {"IN":wordtypes},'OP':'+'}]
        matcher.add('WORDTYPES', [pattern], on_match=Match)
        matches = matcher(doc)
        poem=" ".join(out)
    if metaphor!=1:
        for wtype in wordtypes:
            if wtype in poem:
                while wtype in poem:
                    wchoice=weighted_random(wordlist[wtype])
                    if wchoice is None:
                        print(wordlist[wtype])
                        poem=poem.replace(wtype,"Oh",1) #because we've found a place for interjections. GRR
                    else:
                        poem=poem.replace(wtype,wchoice,1)
    

    return poem

def getRoutes(graphtuples,TypeList,start=None):

    possibletuples=graphtuples.loc[graphtuples["starttype"]==TypeList[0].text]
    if start is not None:
        possibletuples=possibletuples.loc["start"==start]
    if len(TypeList)>1:
        return possibletuples.loc[any(possibletuples["target"]==traversegraph[graphtuples,wordlist,TypeList[1:],startnodes,outlist]["start"])]
    else:
        return possibletuples    

def TraverseGraph(graphtuples,wordlist,TypeList, startnodes=None,outlist=[]):
    '''
    start=startnodes
    for i in range(0,len(TypeList)):
        df=getRoutes[graphtuples,TypeList[i:],start]
        possiblelist=list(filter(lambda word: word[0] in set(df["start"].values.tolist()), wordlist[TypeList[i].text]))
        start=weighted_random(possiblelist)
        outlist.append(start)
    '''
    if ((startnodes is None) or (len(startnodes)<1)):
        return TraverseGraph(graphtuples,wordlist,TypeList,graphtuples.loc[graphtuples["starttype"] == TypeList[0].text]["start"].values.tolist(),outlist=outlist)
    possiblelist=itertools.chain.from_iterable([list(filter(lambda word: word[0] in startnodes, wordlist[typ])) for typ in wordlist])
    AddedWord=weighted_random(possiblelist)
    if len(TypeList)<=1: # end case our types is just defining the word choices we've been passed. none left to find
        return outlist+[AddedWord]
    else: #not first case nor last so there are words left to find and we've added one. 
        #what if nextsteps is empty 
        nextsteps =graphtuples.loc[graphtuples["start"] == AddedWord].loc[graphtuples["targettype"] == TypeList[1].text]["target"].values.tolist()#df[~df[Column].isnull()].values.tolist()
        while len(nextsteps)<=0:
            possiblelist=list(filter(lambda word: not(word[0] == AddedWord), possiblelist))
            if len(possiblelist)==0:
                return outlist
            AddedWord=weighted_random(possiblelist)
            nextsteps =graphtuples.loc[graphtuples["start"] == AddedWord].loc[graphtuples["targettype"] == TypeList[1].text]["target"].values.tolist()#df[~df[Column].isnull()].values.tolist()
        return TraverseGraph(graphtuples,wordlist,TypeList[1:],nextsteps,outlist)
        #    print("Failed to find tuple starting with {0} to type {1}".format(AddedWord,TypeList[0]))
def BetterTraverseGraph(graphtuples,wordlist,TypeList, startnodes=None,outlist=[]):
    if len(TypeList)==1:
        return graphtuples.loc[(graphtuples["starttype"] == TypeList[0].text) ]#| graphtuples["targettype"] == TypeList[0].text ]
    if len(TypeList)==2:
        return graphtuples.loc[graphtuples["starttype"] == TypeList[0].text].loc[graphtuples["targettype"] == TypeList[1].text]
    if len(TypeList)>2:
        startdf=graphtuples.loc[graphtuples["starttype"] == TypeList[0].text]
        finaldf=graphtuples.loc[graphtuples["targettype"] == TypeList[len(TypeList)-1].text]
        link=BetterTraverseGraph(graphtuples,wordlist,TypeList[1:-1],startnodes,outlist)
        startdf=startdf.loc[any(startdf["target"]==link["start"])]
        finaldf=finaldf.loc[any(finaldf["start"]==link["target"])]
      
        
def parralelproc(params,df,func,n_cores=os.cpu_count()):
    #print(params)
    results=[]
    with Pool(n_cores) as pool:
        
        if len(params)>1:
            #df_split = np.array_split(params, n_cores)
            results=pool.starmap(func, zip(repeat(df),params))
        else:
            df_split=np.array_split(df,n_cores)
            result=pool.starmap(func, zip(df_split,repeat(params[0])))
            results.append(list(itertools.chain.from_iterable(result)))
    
    #print(results)
    return dict(zip(params,results)) 

def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')  
    return ReviewText

def extractdf(contents,filename):
    #print(contents)
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(StringIO(decoded.decode('utf-8')),header=0)
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        return pd.read_excel(BytesIO(decoded),header=0)

def weighted_random(pairs):
    total = sum(pair[1] for pair in pairs)
    r = random.uniform(0,total)
    for (name, weight) in pairs:
        r= r- float(weight)
        if r <= 0.1: return name
    return pairs[-1][0]

def parse_contents(contents, filename, date):
    global df,TextFields,GroupFields,keys
    content_type, content_string = contents.split(',')
    if DEBUG:
        print("Reading in ")
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(BytesIO(decoded))
        
    except Exception as e:
        if DEBUG:
            print(e)
        return (html.Div([
            'There was an error processing this file.'
        ]),None)
    returnlist=[html.H5(filename),html.H6(datetime.datetime.fromtimestamp(date))]
    if DEBUG:
        returnlist=returnlist+[
            html.Hr(),  # horizontal line
            html.Div('Raw Content'),
            html.Pre(contents[0:200] + '...', style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })
        ]
    return (html.Div(returnlist),df)
def readtextfile(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        TEXT = "\n".join(StringIO(decoded.decode('utf-8')).read())
        TEXT=buildTemplateFromText(TEXT)
    else:
        TEXT="None"
    return TEXT
def buildknowledgebase(df):
    tr4w = TextRank4Keyword()
    try:
        iterable=df.apply(lambda x:tr4w.getknowledge(nlp(x)))
    except: #probs a series
        iterable=df.apply(lambda x:itertools.chain.from_iterable(tr4w.getknowledge(nlp(str(i))) for i in x.values.tolist()))
    return list(itertools.chain.from_iterable(iterable))
def createWordList(df,param):
    tr4w = TextRank4Keyword()
    toprankslist=list(df.apply(lambda x:TextRankAnalyse(tr4w,x,param)))
    return list(itertools.chain.from_iterable(toprankslist))

def TextRankAnalyse(tr4w, text,wtype):
    out=list()
    try:
        tr4w.analyze(text, candidate_pos = [wtype], window_size=4, lower=False)#
        out=tr4w.get_keywords()
    except Exception as e: #text is series
        print(e)
        out=list()
        for i in text.values.tolist():
            if len(str(i))>17:   #I presume this is therefore a good text to learn on. 
                tr4w.analyze(str(i), candidate_pos = [wtype], window_size=4, lower=False)
                out=out+tr4w.get_keywords()
    return out

def CreateTensorBoard(df, Strings,Text,wordmodel, out_loc=".", name="spaCy_vectors"):
    
    import tensorflow as tf
    from tensorboard.plugins import projector

    from tensorboard.plugins.projector import (
        visualize_embeddings,
        ProjectorConfig,
    )
    tf.compat.v1.disable_eager_execution()

    Texts=" ".join(Text).split("\n")
    Columns=filter(lambda x:df[x].map(lambda x: len(str(x))).max()>100, df)
    tensors={}
    for column in Columns:
        Strings=list(df[column].map(lambda text: Response(str(text).lower())).values.tolist())
        Strings=Strings+Texts
        embeddings={String:wordmodel.infer_vector(String.split()) for String in Strings}
        embeddings_vectors = np.stack(embeddings.values(), axis=0)
        tensors[column]=tf.Variable(embeddings_vectors, name=column)
   
    Strings=Strings+Texts
    embeddings={String:wordmodel.infer_vector(String.split()) for String in Strings}
    embeddings_vectors = np.stack(embeddings.values(), axis=0)

    '''
        # Create some variables.
    emb = tf.Variable(embeddings_vectors, name='word_embeddings')

    # Add an op to initialize the variable.
    init_op = tf.compat.v1.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.compat.v1.train.Saver([emb])

    # Later, launch the model, initialize the variables and save the
    # variables to disk.
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
    '''
    sess = tf.compat.v1.InteractiveSession()
    emb=tf.Variable(embeddings_vectors, name='embeddings') 
    init_op=tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver([emb])
    saver2=tf.compat.v1.train.Saver(tensors.values())
    words = list(embeddings.keys())
    with open(os.path.join(out_loc, 'metadata.tsv'), 'w') as f:
        [f.write(word + "\n") for word in words]
    #meta_file = "{}.tsv".format('metadata')
    sess.run(init_op)
    save_path = saver.save(sess, os.path.join(out_loc, "{}.ckpt".format(name)))
    save2_path=saver2.save(sess, os.path.join(out_loc, "{}.ckpt".format(name)))
    print("Model saved in path: %s" % save_path)
    '''    
    summary_writer = tf.summary.FileWriter(out_loc, graph=sess.graph, )
    # Link the embeddings into the config
    config = ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = name
    embed.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    visualize_embeddings(summary_writer, config)
    '''
    # Tell the projector about the configured embeddings and metadata file
    #visualize_embeddings(writer, config)
    
    if DEBUG:

    # Save session and print run command to the output
        print("Saving Tensorboard Session...")
    
    
    
    os.system(
    "tensorboard --logdir {0}/ --host 0.0.0.0 --port 6006".format(out_loc) 
    ) 
    if DEBUG:
        print("Done. Run `tensorboard --logdir={0}` to view in Tensorboard".format(out_loc))
    return 6006