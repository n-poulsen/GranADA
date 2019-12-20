import numpy as np
import pandas as pd
import plotly
import os
import json
import requests
import plotly.graph_objects as go

def plot_line(df, title, x_axis, y_axis, topn=10, field='Item', to_html=False):
    years = [str(year) for year in range(1993, 2017)] 
    fig = go.Figure()
    items = df[field]
    for i in range(0, topn):
        fig.add_trace(go.Scatter(x=years, y=df.iloc[i][years], mode='lines', name=items[i], line = dict(width=4)))
    fig.update_layout(title=title,
                       xaxis_title=x_axis,
                       yaxis_title=y_axis,
                      width=800,
                     height=500)
    if to_html:
        plotly.offline.plot(fig, filename=''.join(title.split()) +'.html')
    else:
        plotly.offline.plot(fig)