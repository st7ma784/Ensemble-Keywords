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
import plotly.graph_objects as graph_objects
import os
from gensim.matutils import jaccard
DEBUG=bool(os.environ['DEBUG'])
import networkx as nx
docmodel=Doc2Vec.load("/app/UTILS/model/sentsmodel.bin")
def visualizedistance(df,threshold,Column=None):
    df['VectorList']=df[Column].map(lambda text: docmodel.infer_vector(text.split))
    G = nx.Graph()
    map(lambda i,vector: G.add_node(i),enumerate(df['VectorList']))
    texts=df[Column].unique()
    for (i1, i2) in itertools.combinations(range(len(texts)), 2):
        bow1, bow2 = texts[i1], texts[i2]
        distance = jaccard(bow1, bow2)   #HERES THE DISTANCE METRIC
        G.add_edge(i1, i2, weight=1/distance)

    pos = nx.spring_layout(G)

    
    elarge=[(u,v,d) for (u,v,d) in G.edges(data=True) if d['weight'] > threshold]
    esmall=[(u,v,d) for (u,v,d) in G.edges(data=True) if d['weight'] <= threshold]

    #convert our Nx model into plotly 
    edge_x = []
    edge_y = []
    for edge in esmall:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    weak_edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
        )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
               
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text', # show most similar
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',                  #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    data = [weak_edge_trace,strong_edge_trace, node_trace]
    layout=go.Layout(
                title='Distance between responses',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
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
        dcc.Dropdown(id='Distancestext-select',options=Texts,style={'width': '100\%'}),
        html.P('Grouped By'),
        dcc.Slider(id='distance-slider',min=0,max=4,step=0.05,value=1.25,),
        dcc.Graph('distancegraph', config={'displayModeBar': False}),
    ]

    return html.Div(OutPut)