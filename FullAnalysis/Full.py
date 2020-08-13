import csv,os,random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import OrderedDict,defaultdict
import numpy as np
import spacy
from bokeh.plotting import figure, output_file, show
import plotly.graph_objects as go
from chart_studio.plotly import plot,iplot
from textblob import TextBlob
import pandas as pd
import numpy as np
import chart_studio.plotly as py
from multiprocessing.pool import ThreadPool as Pool
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

def weighted_random(pairs):
    total = sum(pair[1] for pair in pairs)
    r = random.uniform(0, total)
    for (name, weight) in pairs:
        r -= weight
        if r <= 0: return name


class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight
        with open(os.path.join("DATA",'abstractnounlist.txt'),'r') as file:
            ABST=file.readlines()
        self.ABSTLIST=[word.lower().replace("\n","") for word in ABST]
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

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        return list(node_weight.items())[:number]

        
        
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




def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')  
    return ReviewText

def CreateGraph(df):

    trace1 = go.Scatter(
        x=df['Age group'], y=df['polarity'], mode='markers', name='points',
        marker=dict(color='rgb(102,0,0)', size=2, opacity=0.4)
    )
    trace2 = go.Histogram2dContour(
        x=df['Age group'], y=df['polarity'], name='density', ncontours=20,
        colorscale='Hot', reversescale=True, showscale=False
    )
    trace3 = go.Histogram(
        x=df['Age group'], name='Age density',
        marker=dict(color='rgb(102,0,0)'),
        yaxis='y2'
    )
    trace4 = go.Histogram(
        y=df['polarity'], name='Sentiment Polarity density', marker=dict(color='rgb(102,0,0)'),
        xaxis='x2'
    )
    data = [trace1, trace2, trace3, trace4]

    layout = go.Layout(
        showlegend=False,
        autosize=False,
        width=600,
        height=550,
        xaxis=dict(
            domain=[0, 0.85],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            domain=[0, 0.85],
            showgrid=False,
            zeroline=False
        ),
        margin=dict(
            t=50
        ),
        hovermode='closest',
        bargap=0,
        xaxis2=dict(
            domain=[0.85, 1],
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            domain=[0.85, 1],
            showgrid=False,
            zeroline=False
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show(renderer="png")
    
    #py.plot(fig, filename='2dhistogram-2d-density-plot-subplots')


def CreateGraph2(df):
    AgeGroups=df['Age group'].unique()
    Traces={}
    for age in AgeGroups:
        r=random.randint(0,255)
        g=random.randint(0,255)
        b=random.randint(0,255)
        Traces[age]=go.Box(
            y=df.loc[df['Age group'] == age]['polarity'],
            name = 'age',
            marker = dict(
                color = ''.join(['rgb(',str(r),", ", str(g), ", ", str(b),' )'])
            )
        )
    
    data = list(Traces.values())
    layout = go.Layout(
        title = "Sentiment by age group"
    )
    fig = go.Figure(data=data,layout=layout)
    fig.show(renderer="png")
    #py.plot(fig, filename='2dhistogram-2d-density-plot-subplots')

wordtypes=["ADJ","ABSTNOUN","INTRANVERB","TRANVERB","INTJ","ADV","PRPN","VERB","NOUN"]
wordlist={wtype:list() for wtype in wordtypes}
def createWordList(df):
    tr4w = TextRank4Keyword()
    df.apply(lambda x:TextRankAnalyse(tr4w,x))

def TextRankAnalyse(tr4w, text):
    global wordlist
    for wtype in wordtypes:
        tr4w.analyze(text, candidate_pos = [wtype], window_size=4, lower=False)
        wordlist[wtype]+=tr4w.get_keywords(10)
def parralelproc(df,func,n_cores=os.cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    pool.map(func, df_split)
    pool.close()
    pool.join()
def createpoem(poem):
    for wtype in wordtypes:
        while wtype in poem:
                wchoice=weighted_random(wordlist[wtype])
                if wchoice is None:
                    poem=poem.replace(wtype,"Oh",1) #because we've found a place for interjections. GRR
                else:
                    poem=poem.replace(wtype,wchoice,1)
    return poem
def main():
    global wordtypes

    sid=SentimentIntensityAnalyzer()
    sentences=["The ADJ NOUN ADV TRANVERBs the NOUN.",
    "ADJ, ADJ NOUN ADV TRANVERBs a ADJ, ADJ NOUN.",
    "ABSTNOUN is a ADJ NOUN.",
    "INTJ, ABSTNOUN!",
    "NOUNs INTRANVERB!",
    "The NOUN INTRANVERBs like a ADJ NOUN.",
    "NOUNs INTRANVERB like ADJ NOUN.",
    "Why does the NOUN INTRANVERB?",
    "INTRANVERB ADV like a ADJ NOUN.",
    "ABSTNOUN, ABSTNOUN, and ABSTNOUN.",
    "Where is the ADJ NOUN?",
    "All NOUNs TRANVERB ADJ, ADJ NOUN.",
    "Never TRANVERB a NOUN.",
    ]
    
    poem="\n".join(random.sample(sentences,10))
    words=[]
    Filename=os.path.join("DATA","DataoftheHeart_LakeDistrictsurvey.csv")
    totals=defaultdict(int)

    polaritykeys=sid.polarity_scores("").keys()
    with open(Filename, mode='r') as infile:
        reader = list(csv.DictReader(infile))
        desiredkey=list(reader[0].keys())[17]
    print("From: "+desiredkey)
    df = pd.read_csv(Filename) # open file
    df = df[~df[desiredkey].isnull()] #filter out empty rows
    df[desiredkey] = preprocess(df[desiredkey]) #clean text up a bit
    parralelproc(df[desiredkey],createWordList)#create our wordlist
    #print(wordlist)
    print(createpoem(poem))
    for key in polaritykeys:
        df[key]=df[desiredkey].map(lambda text: sid.polarity_scores(text)[key]) # create rows for sentiment [pos, nue, neg, compound]
        print(key + " score for column : " + str(sum(df[key])))  # lets just sanity check those scores. 
    df['polarity'] = df[desiredkey].map(lambda text: TextBlob(text).sentiment.polarity) # lets make a single polarity value using textblob
    df['review_len'] = df[desiredkey].astype(str).apply(len)
    df['word_count'] = df[desiredkey].apply(lambda x: len(str(x).split()))
    CreateGraph2(df)
    for word,score in wordlist['ADJ']:
        totals[word]=totals.get(word,0)+score
    totals=OrderedDict(sorted(totals.items(), key=lambda t: t[1], reverse=True))
if __name__=="__main__":
    main()
