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
from flask import Flask, render_template, send_file
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from functools import partial
from itertools import repeat
import importlib
from UTILS.utils import *
DEBUG=True

app = dash.Dash(__name__)


def main():
    global wordtypes

    server = app.server
    sid=SentimentIntensityAnalyzer()
    #Filename=os.path.join("DATA","DataoftheHeart_LakeDistrictsurvey.csv")
    #df = pd.read_csv(Filename) # open file
    #keys=list(df.keys())
    Funcs=filter(lambda x: x.endswith(".py"), [file for file in os.listdir("./APPS")])
    Funcs=map(lambda x: x[:-3],Funcs)
    #TextFields=[10,12,13,14,15,17,19,20,21,27]
    #GroupFields=['Age group','Postcode / Zip','Country','Gender']
    app.layout = html.Div([    
        dcc.Upload(id='upload-data',children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '100%', 'border':'1px'},),    
        html.Div(id='output-data-upload'),
        html.P('Graph Types:'),
        dcc.Dropdown(id='Function-select', options=[{'label': function, 'value': function} for function in Funcs],style={'width': '100\%'}),
        html.Div(id='FuncOutput'),
        html.Div(id='intermediate-value', style={'display': 'none'})
    ])
    

    # Hidden div inside the app that stores the dataset
    #<<<<<< ---------------- GENERIC CALLBACKS ------------------->>>>>>>>>>>
    

    @app.callback([Output('output-data-upload', 'children'),Output('intermediate-value', 'children'),],[Input('upload-data', 'contents')],[State('upload-data', 'filename'),State('upload-data', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = parse_contents(list_of_contents, list_of_names, list_of_dates)
        df=extractdf(list_of_contents, list_of_names)
        df=df.to_json(date_format='iso', orient='split')
        return children,df

    @app.callback(Output('FuncOutput', 'children'),[Input('Function-select', 'value'),Input('intermediate-value', 'children')])
    def GenerateHTML(Function,jsondf):
        function_string = '.'.join(['APPS',Function,"run"])
        mod_name, func_name = function_string.rsplit('.',1)
        mod = importlib.import_module(mod_name)
        df = pd.read_json(jsondf, orient='split')

        func = getattr(mod, func_name)
        return func(df)


    #<<<<<< ---------------- WORDRANK CALLBACKS ------------------->>>>>>>>>>>

    #give back options for chosen group
    @app.callback(Output('wordrankgroup-dropdown', 'options'),[Input('wordrankgroup-select', 'value'),Input('intermediate-value', 'children')])
    def update_date_dropdown(name,jsondf):
        options= ["All"]
        if name !="None":
            df = pd.read_json(jsondf, orient='split')
            newdf = df[~df[name].isnull()]
            options+=list(newdf[name].unique())
        return [{'label': i, 'value': i} for i in options]
   

    #give back a poem based on out picked group
    @app.callback(Output('PoemOut', 'children'),[Input('wordranktext-select','value'),Input('intermediate-value', 'children'),Input('upload-poem-template', 'contents')],[State('upload-poem-template', 'filename')])
    def updatepoem(Column,jsondf,list_of_contents, list_of_names):
        poem=readtextfile(list_of_contents, list_of_names)
        newdf = pd.read_json(jsondf, orient='split')
        mod = importlib.import_module("APPS.wordrank")
        wordrank = getattr(mod, "wordrankpoem")
        return wordrank(newdf,Column,poem)

    #find preferred word type of given subgroup of group
    @app.callback(Output('filterTextOut', component_property='children'),[Input('wordranktext-select','value'),Input('wordrankwordtype','value'),Input('wordrankgroup-dropdown', 'value'),Input('wordrankgroup-select', 'value'),Input('intermediate-value', 'children')])
    def update_keywordsbygroup(Column,Type,name,group,jsondf):
        df = pd.read_json(jsondf, orient='split')
        newdf = df[~df[Column].isnull()] #filter out empty rows
        newdf=newdf[newdf[group]==name]
        newdf[Column] = preprocess(newdf[Column]) #clean text up a bit
        mod = importlib.import_module("APPS.wordrank")
        wordrankpersubgroup = getattr(mod, "wordrankpersubgroup")
        return wordrankpersubgroup(Column,Type,name,group,df)

        #return [{'label': i, 'value': i} for i in newdf[name].unique()]


    #<<<<<< ---------------- POLARITY CALLBACKS ------------------->>>>>>>>>>>


    @app.callback(Output('polaritygraph', 'figure'),[Input('polaritytext-select','value'),Input('polaritygroup-select','value'),Input('intermediate-value', 'children')])
    def drawpolaritygraph(Column,Group,jsondf):
        df = pd.read_json(jsondf, orient='split')
        if Column!="*":
            newdf = df[~df[Column].isnull()] #filter out empty rows
        else:
            newdf=df
        mod = importlib.import_module("APPS.polarity")
        polarity = getattr(mod, "polarity")

        return polarity(newdf,Column,Group)
    

    #<<<<<< ---------------- SENTIMENT CALLBACKS ------------------->>>>>>>>>>>


    @app.callback(Output('sentimentgraph', 'figure'),[Input('sentimenttextarea-button','n_clicks')],[State('sentimenttext-select','value'),State('sentimentgroup-select','value'),State('intermediate-value', 'children'),State('sentimentgranularity-select','value'),State('sentimentext-entry', 'value')])
    def drawpolaritygraph(_,Column,Group,jsondf,Granularity,Text):
        df = pd.read_json(jsondf, orient='split')
        mod = importlib.import_module("APPS.sentimentfinder")
        sentiment = getattr(mod, "sentiment")

        return sentiment(df,Column,Group,Granularity,Text)



    #<<<<<< ---------------- GRAPH RESPONSES CALLBACK ----------------->>>>>>>>>>>    

    @app.callback(Output('distancegraph', 'figure'),[Input('Distancetext-select','value'),Input('distance-slider','value'),Input('intermediate-value', 'children')])
    def drawpolaritygraph(Column,threshold,jsondf):
        df = pd.read_json(jsondf, orient='split')
        mod = importlib.import_module("APPS.visualizedistance")
        sentiment = getattr(mod, "visualizedistance")
        return visualizedistance(df,Column,threshold)
    
    #<<<<<< ---------------- RUN  ------------------->>>>>>>>>>>



    app.run_server(host='0.0.0.0',debug=DEBUG, port=8050)


if __name__=="__main__":
    main()
    