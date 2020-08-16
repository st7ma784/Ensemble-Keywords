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
from App import app
def sentiment(df,Column=None,Group=None):
    #load gensim model
    #FOR GRP IN GROUP
    #   for item in granularity of respone
    #       plot similarity
    if Column=='*':
        TextFields=filter(lambda x:df[x].map(lambda x: len(str(x))).max()>100, df)
        df['polarity'] = df[TextFields].map(lambda text: TextBlob(text).sentiment.polarity)
    else:
        df['polarity'] = df[Column].map(lambda text: TextBlob(text).sentiment.polarity)
    Traces={}
    if Group != 'None':
        Groups=df[Group].unique()

        for grp in Groups:
            r=random.randint(0,255)
            g=random.randint(0,255)
            b=random.randint(0,255)
            Traces[grp]=go.Box(
                y=df.loc[df[Group] == grp]['polarity'],
                name = grp,
                marker = dict(
                    color = ''.join(['rgb(',str(r),", ", str(g), ", ", str(b),' )'])
                )
            )
    Traces["all"]=go.Box(
                y=df['polarity'],
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
    return fig

def run(df): 

    TextFields=list(filter(lambda x:df[x].map(lambda x: len(str(x))).max()>100, df))+["*"]
    GroupFields=list(filter(lambda x:len(df[x].unique())<15,df))+['None']
    Texts=[{'label': key, 'value': key} for key in TextFields]
    Groups=[{'label': key, 'value': key} for key in GroupFields]
    OutPut=[    
        html.P('Text entries to process:'),
        dcc.Dropdown(id='sentimenttext-select',options=Texts,style={'width': '100\%'}),
        html.P('Grouped By'),
        dcc.Dropdown(id='sentimentgroup-select',options=Groups,style={'width': '100\%'}),
        dcc.Graph('sentimentgraph', config={'displayModeBar': False}),
    ]
    #need to take input string
    #need to pull in granularity [paragraph, sentence, word]
    return html.Div(OutPut)