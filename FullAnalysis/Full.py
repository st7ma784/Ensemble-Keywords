import csv,os,random,spacy, datetime,dash, itertools, base64
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import OrderedDict,defaultdict
import numpy as np
from io import BytesIO,StringIO
from bokeh.plotting import figure, output_file, show
import plotly.graph_objects as go
from chart_studio.plotly import plot,iplot
from textblob import TextBlob
import pandas as pd
import numpy as np
import chart_studio.plotly as py
from multiprocessing.pool import ThreadPool as Pool
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, render_template, send_file
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from functools import partial
from itertools import repeat
DEBUG=True
nlp = spacy.load('en_core_web_sm')

wordtypes=["ADJ","ABSTNOUN","INTRANVERB","TRANVERB","INTJ","ADV","PRPN","VERB","NOUN"]
wordlist={wtype:list() for wtype in wordtypes}
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
df=None
TextFields=[10,12,13,14,15,17,19,20,21,27] # if unique values greater than  15? or max len? 
TextFields=[]
GroupFields=['Age group','Postcode / Zip','Country','Gender'] # these can be made with defining unique() values < 15
GroupFields=[]
keys=[]
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

def polarity(df,Column,Group):
    df['polarity'] = df[Column].map(lambda text: TextBlob(text).sentiment.polarity)

    Groups=df[Group].unique()
    Traces={}
    for grp in Groups:
        r=random.randint(0,255)
        g=random.randint(0,255)
        b=random.randint(0,255)
        Traces[g]=go.Box(
            y=df.loc[df[Group] == grp]['polarity'],
            name = grp,
            marker = dict(
                color = ''.join(['rgb(',str(r),", ", str(g), ", ", str(b),' )'])
            )
        )
    Figures=OrderedDict(sorted(Traces.items(), key=lambda t: str(t[0])[:2], reverse=False))

    data = list(Figures.values())
    layout = go.Layout(
        title = "Sentiment by group"
    )
    fig = go.Figure(data=data,layout=layout)
    #img=BytesIO(fig.to_image())
    #img.seek(0)

    return fig

def createWordList(df,param):
    tr4w = TextRank4Keyword()
    toprankslist=list(df.apply(lambda x:TextRankAnalyse(tr4w,x,param)))
    return list(itertools.chain.from_iterable(toprankslist))
def TextRankAnalyse(tr4w, text,wtype):
    global wordlist
    tr4w.analyze(text, candidate_pos = [wtype], window_size=4, lower=False)
    return tr4w.get_keywords(10)
    
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

def main():
    global wordtypes

    app = dash.Dash(__name__)
    server = app.server
    sid=SentimentIntensityAnalyzer()
    Filename=os.path.join("DATA","DataoftheHeart_LakeDistrictsurvey.csv")
    #df = pd.read_csv(Filename) # open file
    #keys=list(df.keys())
    Functions={"polarity":polarity}
    TextFields=[10,12,13,14,15,17,19,20,21,27]
    GroupFields=['Age group','Postcode / Zip','Country','Gender']
    app.layout = html.Div([    
        dcc.Upload(id='upload-data',children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '100%', 'border':'1px'},),    
        html.Div(id='output-data-upload'),
        html.P('Graph Types:'),
        dcc.Dropdown(id='Function-select', options=[{'label': function, 'value': function} for function in Functions],style={'width': '100\%'}),
        html.P('Text entries to process:'),
        dcc.Dropdown(id='Column-select',style={'width': '100\%'}),
        html.P('Grouped By'),
        dcc.Dropdown(id='Groupby-select',  style={'width': '100\%'}),
        dcc.Graph('Boxplot-graph', config={'displayModeBar': False}),
        html.P('find common words of type:'),
        dcc.Dropdown(id='WordType', options=[{'label': WordType, 'value': WordType} for WordType in wordlist], style={'width': '100\%'}),
        html.P('Overall:'),
        html.Div(id='TextOut'),
        html.P('view most common words across by group:'),
        dcc.Dropdown(id='group-dropdown',style={'width': '100\%'}),
        html.Div(id='filterTextOut'),
        html.P('Here\'s a poem generated with responses in this column'),
        html.Div(id='PoemOut'),
        # Hidden div inside the app that stores the dataset
        html.Div(id='intermediate-value', style={'display': 'none'})
    ])
    

    @app.callback([Output('output-data-upload', 'children'),Output('Column-select', 'options'),Output('Groupby-select', 'options'),Output('intermediate-value', 'children'),],[Input('upload-data', 'contents')],[State('upload-data', 'filename'),State('upload-data', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = parse_contents(list_of_contents, list_of_names, list_of_dates)
        df=extractdf(list_of_contents, list_of_names)
        keys=list(df.keys())
        print(keys)
        TextFields=filter(lambda x:df[x].map(lambda x: len(str(x))).max()>100, keys)
        GroupFields=filter(lambda x:len(df[x].unique())<15,keys)
        Texts=[{'label': key, 'value': key} for key in TextFields]
        Groups=[{'label': Group, 'value': Group} for Group in GroupFields]
        df=df.to_json(date_format='iso', orient='split')
        return children,Texts,Groups,df

    @app.callback(Output('Boxplot-graph', 'figure'),[Input('Function-select', 'value'),Input('Column-select','value'),Input('Groupby-select','value'),Input('intermediate-value', 'children')])
    def update_graph(Function,Column,Group,jsondf):
        df = pd.read_json(jsondf, orient='split')
        newdf = df[~df[Column].isnull()] #filter out empty rows
        newdf[Column] = preprocess(newdf[Column]) #clean text up a bit
        return globals()[Function](newdf,Column,Group)
    @app.callback(Output('TextOut', component_property='children'),[Input('Column-select','value'),Input('WordType','value'),Input('intermediate-value', 'children')])
    def update_keywords(Column,Type,jsondf):
        df = pd.read_json(jsondf, orient='split')
        newdf = df[~df[Column].isnull()] #filter out empty rows
        newdf[Column] = preprocess(newdf[Column]) #clean text up a bit
        wordlist=parralelproc([Type],newdf[Column],createWordList)#create our wordlist
        totals=defaultdict(int)
        for (word,score) in wordlist[Type]:
            totals[word]=totals.get(word,0)+score
    
        return 'Output: {}'.format(sorted(totals.items(), key=lambda t: t[1], reverse=True)[:10]) 
    @app.callback(Output('PoemOut', component_property='children'),[Input('Column-select','value'),Input('intermediate-value', 'children')])
    def update_poem(Column,jsondf):
        df = pd.read_json(jsondf, orient='split')
        poem="\n".join(random.sample(sentences,10))
        newdf = df[~df[Column].isnull()] #filter out empty rows
        newdf[Column] = preprocess(newdf[Column]) 
        wordlist=parralelproc(wordtypes,newdf[Column],createWordList)#create our wordlist

        for wtype in wordtypes:
            while wtype in poem:
                wchoice=weighted_random(wordlist[wtype])
                if wchoice is None:
                    poem=poem.replace(wtype,"Oh",1) #because we've found a place for interjections. GRR
                else:
                    poem=poem.replace(wtype,wchoice,1)
        return 'Output: {}'.format(poem)
    @app.callback(Output('group-dropdown', 'options'),[Input('Groupby-select', 'value'),Input('intermediate-value', 'children')])
    def update_date_dropdown(name,jsondf):
        df = pd.read_json(jsondf, orient='split')
        newdf = df[~df[name].isnull()]
        return [{'label': i, 'value': i} for i in newdf[name].unique()]
    @app.callback(Output('filterTextOut', component_property='children'),[Input('Column-select','value'),Input('WordType','value'),Input('group-dropdown', 'value'),Input('Groupby-select', 'value'),Input('intermediate-value', 'children')])
    def update_keywordsbygroup(Column,Type,name,group,jsondf):
        df = pd.read_json(jsondf, orient='split')
        newdf = df[~df[Column].isnull()] #filter out empty rows
        newdf=newdf[newdf[group]==name]
        newdf[Column] = preprocess(newdf[Column]) #clean text up a bit
        wordlist=parralelproc([Type],newdf[Column],createWordList)#create our wordlist
        totals=defaultdict(int)
        for (word,score) in wordlist[Type]:
            totals[word]=totals.get(word,0)+score
    
        return 'Output: {}'.format(sorted(totals.items(), key=lambda t: t[1], reverse=True)[:10]) 
        return [{'label': i, 'value': i} for i in newdf[name].unique()]


    app.run_server(host='0.0.0.0',debug=DEBUG, port=8050)


if __name__=="__main__":
    main()
    