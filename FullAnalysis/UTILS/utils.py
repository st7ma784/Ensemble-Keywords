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

import tensorflow as tf
from tensorboard.plugins import projector

from tensorboard.plugins.projector import (
    visualize_embeddings,
    ProjectorConfig,
)

DEBUG=False

import spacy,os
from collections import OrderedDict,defaultdict

from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight
        with open(os.path.join("UTILS","DATA",'abstractnounlist.txt'),'r') as file:
            ABST=file.readlines()
        self.wordtypes=["ADJ","ABSTNOUN","INTRANVERB","TRANVERB","INTJ","ADV","PRPN","VERB","NOUN"]
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
                return token.pos_
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
                    if len(text)>1 or text=="a" or text=="I": 
                        selected_words.append(text.lower())

            sentences.append(" ".join(selected_words))
        return "\n".join(sentences)

    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        return list(node_weight.items())[:number]

    def getknowledge(self,doc):
        print(doc.text)
        print("Dependencies", [(t.text, t.dep_, t.head.text) for t in  filter(lambda w: w.dep_ in ("compound","amod", "dobj"), doc)])
        print("WordTypes",[(self.check_verb(t),self.check_verb(t.head)) for t in  filter(lambda w: w.dep_ in ("compound","amod", "dobj"), doc)])
        self.knowledgebase=set((t.text, t.dep_, t.head.text) for t in doc)

        '''
        # merge entities and noun chunks into one token
        spans = list(doc.ents) + list(doc.noun_chunks)
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)

        relations = []
        for money in filter(lambda w: w.ent_type_ == "MONEY", doc):
            if money.dep_ in ("attr", "dobj"):
                subject = [w for w in money.head.lefts if w.dep_ == "nsubj"]
                if subject:
                    subject = subject[0]
                    relations.append((subject, money))
            elif money.dep_ == "pobj" and money.head.dep_ == "prep":
                relations.append((money.head.head, money))
        return relations


        print("Processing %d texts" % len(TEXTS))

        for text in TEXTS:
            doc = nlp(text)
            relations = extract_currency_relations(doc)
            for r1, r2 in relations:
                print("{:<10}\t{}\t{}".format(r1.text, r2.ent_type_, r2.text))
        '''
        return set((t.text, t.dep_, t.head.text) for t in doc)
        
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
    r = random.uniform(0, total)
    for (name, weight) in pairs:
        r -= weight
        if r <= 0: return name

def parse_contents(contents, filename, date):
    global df,TextFields,GroupFields,keys
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(BytesIO(decoded))
        
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    returnlist=[html.H5(filename),html.H6(datetime.datetime.fromtimestamp(date)),]
    if DEBUG:
        returnlist=returnlist.extend([
            html.Hr(),  # horizontal line
            html.Div('Raw Content'),
            html.Pre(contents[0:200] + '...', style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })
        ])
    return html.Div(returnlist)
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
    return list(itertools.chain.from_iterable(df.apply(lambda x:tr4w.getknowledge(nlp(x)))))

def createWordList(df,param):
    tr4w = TextRank4Keyword()
    toprankslist=list(df.apply(lambda x:TextRankAnalyse(tr4w,x,param)))
    return list(itertools.chain.from_iterable(toprankslist))

def TextRankAnalyse(tr4w, text,wtype):
    tr4w.analyze(text, candidate_pos = [wtype], window_size=4, lower=False)
    return tr4w.get_keywords(10)

def CreateTensorBoard(Strings, out_loc=".", name="spaCy_vectors"):
    meta_file = "{}.tsv".format(name)
    out_meta_file = os.path.join(out_loc, meta_file)

    strings_stream = tqdm.tqdm(Strings, total=len(Strings), leave=False)
    queries = [w for w in strings_stream if model.vocab.has_vector(w)]
    vector_count = len(queries)

    print(
        "Building Tensorboard Projector metadata for ({}) vectors: {}".format(
            vector_count, out_meta_file
        )
    )

    tf_vectors_variable = numpy.zeros((vector_count, model.vocab.vectors.shape[1]))
    with open(out_meta_file, "wb") as file_metadata:
        # Define columns in the first row
        file_metadata.write("Text\tFrequency\n".encode("utf-8"))
        vec_index = 0
        for text in tqdm.tqdm(queries, total=len(queries), leave=False):
            text = "<Space>" if text.lstrip() == "" else text
            lex = model.vocab[text]
            tf_vectors_variable[vec_index] = model.vocab.get_vector(text)
            file_metadata.write(
                "{}\t{}\n".format(text, math.exp(lex.prob) * vector_count).encode(
                    "utf-8"
                )
            )
            vec_index += 1
    print("Running Tensorflow Session...")
    sess = tf.InteractiveSession()
    tf.Variable(tf_vectors_variable, trainable=False, name=name)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(out_loc, sess.graph)

    # Link the embeddings into the config
    config = ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = name
    embed.metadata_path = meta_file

    # Tell the projector about the configured embeddings and metadata file
    visualize_embeddings(writer, config)

    # Save session and print run command to the output
    print("Saving Tensorboard Session...")
    saver.save(sess, path.join(out_loc, "{}.ckpt".format(name)))
    print("Done. Run `tensorboard --logdir={0}` to view in Tensorboard".format(out_loc))
