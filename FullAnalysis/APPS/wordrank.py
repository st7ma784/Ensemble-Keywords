from textblob import TextBlob
from collections import OrderedDict,defaultdict
from bokeh.plotting import figure, output_file, show
import plotly.graph_objects as go
from chart_studio.plotly import plot,iplot
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from textblob import TextBlob
import pandas as pd
import numpy as np
import chart_studio.plotly as py
import random
from App import app
from UTILS.utils import * 
DEBUG=os.environ['DEBUG'] 

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
"ABSTNOUN, ABSTNOUN, and ABSTNOUN",
"Where is the ADJ NOUN?",
"All NOUNs TRANVERB ADJ, ADJ NOUN.",
"Never TRANVERB a NOUN.",
]



def wordrankpoem(newdf,Column,poem):
    if Column=="*":
            Column=list(filter(lambda x:newdf[x].map(lambda x: len(str(x))).max()>100, newdf))
    newdf = newdf[~newdf[Column].isnull()] #filter out empty rows
    if poem=="None":
        poem=random.sample(sentences,10)
        poem="\n".join(poem)
    poem=buildPoem(newdf[Column],poem)
    return poem

def wordrankpersubgroup(Column,Type,name,group,df):
    newdf = df[~df[Column].isnull()] #filter out empty rows
    if name!= "None" and name!="All":
        newdf=newdf[newdf[group]==name]
    if Column=="*":
        Column=list(filter(lambda x:df[x].map(lambda x: len(str(x))).max()>100, df))
    newdf[Column] = preprocess(newdf[Column]) #clean text up a bit
    wordlist=parralelproc([Type],newdf[Column],createWordList)#create our wordlist
    totals=defaultdict(int)
    for (word,score) in wordlist[Type]:
        totals[word]=totals.get(word,0)+score
    return 'Output: {}'.format(sorted(totals.items(), key=lambda t: t[1], reverse=True)[:10]) 

def run(df): 

    TextFields=list(filter(lambda x:df[x].map(lambda x: len(str(x))).max()>100, df))+["*"]
    GroupFields=['None']+list(filter(lambda x:len(df[x].unique())<15,df))
    Texts=[{'label': key, 'value': key} for key in TextFields]
    Groups=[{'label': key, 'value': key} for key in GroupFields]
    Wordtypes=[{'label': key, 'value': key} for key in wordtypes]
    OutPut=[    
        html.P('Text entries to process:'),
        dcc.Dropdown(id='wordranktext-select',options=Texts,style={'width': '100\%'}),
        html.P('Grouped By'),
        dcc.Dropdown(id='wordrankgroup-select',options=Groups,style={'width': '100\%'}),
        html.P('Word Type'),
        dcc.Dropdown(id='wordrankwordtype',options=Wordtypes,style={'width': '100\%'}),    
        html.P('view most common words across by group:'),
        dcc.Dropdown(id='wordrankgroup-dropdown',style={'width': '100\%'}),
           
        html.Div(id='filterTextOut'),
        dcc.Upload(id='upload-poem-template',children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '100%', 'border':'1px'},),    

        html.P('Here\'s a poem generated with responses in this column'),
        html.Div(id='PoemOut'),
    ]

    return html.Div(OutPut)