import gensim
from collections import OrderedDict,defaultdict
from bokeh.plotting import figure, output_file, show
import plotly.graph_objects as go
from chart_studio.plotly import plot,iplot
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import chart_studio.plotly as py
import random
import gensim.downloader as api
from gensim.models import Word2Vec,KeyedVectors
from gensim.test.utils import datapath
from App import app
from gensim.models import doc2vec
from UTILS.utils import *

DEBUG=bool(os.environ['DEBUG'])
'''
from gensim.models import Sent2Vec
sents = Sent2Vec(common_texts, size=100, min_count=1)'''
model=KeyedVectors.load_word2vec_format(datapath("/app/UTILS/model/model.gz"), binary=False)
docmodel=doc2vec.Doc2Vec.load("/app/UTILS/model/sentsmodel.bin")                    
def Sentence(text,Text):
    return gensim.summarization.textcleaner.split_sentences(text)
    #could use summarize and then word sim? or combine with Keywords 
def Response(text):
    return text
def Word(text):
    return text.lower().split()


def sentiment(df,Column,Group,Text):
    if Column=='*':
        Column=filter(lambda x:df[x].map(lambda x: len(str(x))).max()>100, df)
    newdf = df[~df[Column].isnull()] #filter out empty rows
    #filter by group
    print(Text)
    df["TextEmbedding"] = df[Column].map(lambda text: Response(str(text).lower()))
    port=CreateTensorBoard(list(df["TextEmbedding"].values.tolist()),Text,docmodel)
    return "Opened Tensorboard at port {}.".format(port)
    '''
    Traces={}
    if Group != 'None':
        Groups=df[Group].unique()

        for grp in Groups:
            r=random.randint(0,255)
            g=random.randint(0,255)
            b=random.randint(0,255)
            Traces[grp]=go.Box(
                y=df.loc[df[Group] == grp][Granularity],
                name = grp,
                marker = dict(
                    color = ''.join(['rgb(',str(r),", ", str(g), ", ", str(b),' )'])
                )
            )
    Traces["all"]=go.Box(
                y=df[Granularity],
                name = "All",
                marker = dict(
                    color = ''.join(['rgb(',str(random.randint(0,255)),", ", str(random.randint(0,255)), ", ", str(random.randint(0,255)),' )'])
                )
            )
    
    Figures=OrderedDict(sorted(Traces.items(), key=lambda t: str(t[0])[:2], reverse=False))

    data = list(Figures.values())
    layout = go.Layout(
        title = "Sentiment compared to input string"
    )
    fig = go.Figure(data=data,layout=layout)
    return fig'''

def run(df): 

    TextFields=list(filter(lambda x:df[x].map(lambda x: len(str(x))).max()>100, df))+["*"]
    GroupFields=list(filter(lambda x:len(df[x].unique())<15,df))+['None']
    #OptionFields=["Word","Sentence","Response"]
    Texts=[{'label': key, 'value': key} for key in TextFields]
    Groups=[{'label': key, 'value': key} for key in GroupFields]
    #Options=[{'label': key, 'value': key} for key in OptionFields]
    OutPut=[    
        html.P('Text entries to process:'),
        dcc.Dropdown(id='sentimenttext-select',options=Texts,style={'width': '100\%'}),
        html.P('Grouped By'),
        dcc.Dropdown(id='sentimentgroup-select',options=Groups,style={'width': '100\%'}),
        html.P('Similarity comparison of '),
        #dcc.Dropdown(id='sentimentgranularity-select',options=Options,style={'width': '100\%'}),
       # html.P('Sentiment to compare to'),
        dcc.Textarea(
            id='sentimentext-entry',
            placeholder='Enter a value...',
            value='There is much hope for the future',
        ),
        html.Button('Submit', id='sentimenttextarea-button', n_clicks=0),
        html.Div(id='sentimentresult'),

        #dcc.Graph('sentimentgraph', config={'displayModeBar': False}),

    ]
    #need to take input string
    #need to pull in granularity [paragraph, sentence, word]
    return html.Div(OutPut)