import numpy as np
import pandas as pd
import plotly
import os
import json
import requests
import plotly.graph_objects as go

def correct_names(df, name, correct, wrong):
    """Function that returns the dataframe with the correct names for the countries.
    Name indicates the name of the column for which you have to change it"""
    for i in range(len(correct)):
        df.loc[df[name] == wrong[i], name] = correct[i]
    
    return df

def plot_imports():
    
    plotly.offline.init_notebook_mode(connected=True)
    plotly.offline.init_notebook_mode()
    
    correct = ['Bahamas, The', 'Bolivia', 'Brunei',  'China',  "Cote d'Ivoire", 
          'Congo, Democratic Republic of the', 'Congo, Republic of the', 'Czech Republic',
          'Gambia, The', 'Hong Kong', 'Iran', 'Korea, South', 'Laos',
          'Macau', 'Micronesia, Federated States of', 'Moldova', 'Russia',
          'Syria', 'Taiwan', 'Tanzania', 'United States',
          'Venezuela', 'Vietnam', 'Virgin Islands']

    wrong = ['Bahamas', 'Bolivia (Plurinational State of)', 'Brunei Darussalam', 'China, mainland', "Côte d'Ivoire", 
             'Democratic Republic of the Congo', 'Congo', 'Czechia',
            'Gambia', 'China, Hong Kong SAR', 'Iran (Islamic Republic of)', 'Republic of Korea', "Lao People's Democratic Republic",
            'China, Macao SAR', 'Micronesia (Federated States of)', 'Republic of Moldova', 'Russian Federation',
            'Syrian Arab Republic', 'China, Taiwan Province of', 'United Republic of Tanzania',  'United States of America',
            'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'United States Virgin Islands']
    
    years = ['1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', 
        '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', 
        '2015', '2016'] 

    df_countries = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv') 
    df_trade = pd.read_pickle("data/df_trade_merged.pkl")
    df_trade = correct_names(df_trade, 'Importer', correct, wrong)
    df_imp = df_trade.drop(['Exporter', 'Item'], axis = 1).groupby('Importer').sum().reset_index()
    data = df_countries.merge(df_imp, left_on = 'COUNTRY', right_on = 'Importer', how = 'left').drop(['Importer'], axis = 1)
    data = data.replace(np.nan, 0)

    dataplot = []
    for i in years:
        dic = [dict(type='choropleth',
                 locations = data['COUNTRY'].astype(str),
                 z=data[i].astype(float),
                 colorscale='Blues',
                 colorbar_title = 'Imports',
                 locationmode='country names')]
        dataplot.append(dic[0].copy())

    steps = []
    for i in range(len(dataplot)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(dataplot)],
                    label='Year {}'.format(i + 1993))
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=0,
                    pad={"t": 1},
                    steps=steps)]    
    layout = dict(geo=dict(scope='world',
                  projection={'type': 'equirectangular'},
                  showframe=False),
                  margin={"r":0,"t":0,"l":0,"b":0},
                  sliders=sliders)
    fig = dict(data=dataplot, 
           layout=layout)
    plotly.offline.iplot(fig)
    a = 1
    return a

def plot_exports():
    
    plotly.offline.init_notebook_mode(connected=True)
    plotly.offline.init_notebook_mode()
    
    correct = ['Bahamas, The', 'Bolivia', 'Brunei',  'China',  "Cote d'Ivoire", 
          'Congo, Democratic Republic of the', 'Congo, Republic of the', 'Czech Republic',
          'Gambia, The', 'Hong Kong', 'Iran', 'Korea, South', 'Laos',
          'Macau', 'Micronesia, Federated States of', 'Moldova', 'Russia',
          'Syria', 'Taiwan', 'Tanzania', 'United States',
          'Venezuela', 'Vietnam', 'Virgin Islands']

    wrong = ['Bahamas', 'Bolivia (Plurinational State of)', 'Brunei Darussalam', 'China, mainland', "Côte d'Ivoire", 
             'Democratic Republic of the Congo', 'Congo', 'Czechia',
            'Gambia', 'China, Hong Kong SAR', 'Iran (Islamic Republic of)', 'Republic of Korea', "Lao People's Democratic Republic",
            'China, Macao SAR', 'Micronesia (Federated States of)', 'Republic of Moldova', 'Russian Federation',
            'Syrian Arab Republic', 'China, Taiwan Province of', 'United Republic of Tanzania',  'United States of America',
            'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'United States Virgin Islands']
    
    years = ['1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', 
        '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', 
        '2015', '2016'] 

    df_countries = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv') 
    df_trade = pd.read_pickle("data/df_trade_merged.pkl")
    df_trade = correct_names(df_trade, 'Exporter', correct, wrong)
    df_exp = df_trade.drop(['Importer', 'Item'], axis = 1).groupby('Exporter').sum().reset_index()
    data = df_countries.merge(df_exp, left_on = 'COUNTRY', right_on = 'Exporter', how = 'left').drop(['Exporter'], axis = 1)
    data = data.replace(np.nan, 0)

    dataplot = []
    for i in years:
        dic = [dict(type='choropleth',
                 locations = data['COUNTRY'].astype(str),
                 z=data[i].astype(float),
                 colorscale='Blues',
                 colorbar_title = 'Imports',
                 locationmode='country names')]
        dataplot.append(dic[0].copy())

    steps = []
    for i in range(len(dataplot)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(dataplot)],
                    label='Year {}'.format(i + 1993))
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=0,
                    pad={"t": 1},
                    steps=steps)]    
    layout = dict(geo=dict(scope='world',
                  projection={'type': 'equirectangular'},
                  showframe=False),
                  margin={"r":0,"t":0,"l":0,"b":0},
                  sliders=sliders)
    fig = dict(data=dataplot, 
           layout=layout)
    plotly.offline.iplot(fig)
    

def plot_globaltrade():
    correct = ['Bahamas, The', 'Bolivia', 'Brunei',  'China',  "Cote d'Ivoire", 
          'Congo, Democratic Republic of the', 'Congo, Republic of the', 'Czech Republic',
          'Gambia, The', 'Hong Kong', 'Iran', 'Korea, South', 'Laos',
          'Macau', 'Micronesia, Federated States of', 'Moldova', 'Russia',
          'Syria', 'Taiwan', 'Tanzania', 'United States',
          'Venezuela', 'Vietnam', 'Virgin Islands']

    wrong = ['Bahamas', 'Bolivia (Plurinational State of)', 'Brunei Darussalam', 'China, mainland', "Côte d'Ivoire", 
             'Democratic Republic of the Congo', 'Congo', 'Czechia',
            'Gambia', 'China, Hong Kong SAR', 'Iran (Islamic Republic of)', 'Republic of Korea', "Lao People's Democratic Republic",
            'China, Macao SAR', 'Micronesia (Federated States of)', 'Republic of Moldova', 'Russian Federation',
            'Syrian Arab Republic', 'China, Taiwan Province of', 'United Republic of Tanzania',  'United States of America',
            'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'United States Virgin Islands']
    
    correct2 = ['Bolivia', 'Brunei', 'Cape Verde', 'Czech Republic', 'Federated States of Micronesia',
           'Guinea Bissau', 'Iran', 'Laos', 'Moldova', 'Republic of Congo', 
           'Republic of Serbia', 'Russia', 'South Korea', 'Syria', 'Taiwan',
           'The Bahamas', 'Venezuela', 'Vietnam', 'China']
    wrong2 = ['Bolivia (Plurinational State of)', 'Brunei Darussalam', 'Cabo Verde', 'Czechia',  'Micronesia (Federated States of)',
         'Guinea-Bissau', 'Iran (Islamic Republic of)', "Lao People's Democratic Republic", 'Republic of Moldova', 'Congo',
         'Serbia', 'Russian Federation', 'Republic of Korea', 'Syrian Arab Republic', 'China, Taiwan Province of',
         'Bahamas', 'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'China, mainland'] 
    
    df_countries = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv') 
    df_coordinates = pd.read_csv('country_centroids.csv')
    df_coordinates = df_coordinates.loc[df_coordinates['homepart'] == 1]
    
    plotly.offline.init_notebook_mode(connected=True)
    plotly.offline.init_notebook_mode()
    
    df_prod = pd.read_pickle("data/df_prod.pkl")
    df_prod = correct_names(df_prod, 'Area', correct, wrong)
    df_prod = df_prod.loc[df_prod['Element'] == 'Production']
    prod = df_prod.groupby(['Area']).sum().reset_index()
    
    
    df_trade = pd.read_pickle('data/df_trade_merged.pkl')
    df_trade = correct_names(df_trade, 'Importer', correct2, wrong2)
    df_trade = correct_names(df_trade, 'Exporter', correct2, wrong2)
    
    trade = df_trade.groupby(['Importer', 'Exporter']).sum().reset_index()
    
    data = trade.merge(df_coordinates[['sovereignt', 'Longitude', 'Latitude']], left_on = 'Importer', right_on = 'sovereignt', how = 'inner').drop(['sovereignt'], axis = 1)\
    .rename(columns = {'Longitude':'Reporter Long', 'Latitude':'Reporter Lat'})
    data = data.merge(df_coordinates[['sovereignt', 'Longitude', 'Latitude']], left_on = 'Exporter', right_on = 'sovereignt', how = 'inner').drop(['sovereignt'], axis = 1)\
        .rename(columns = {'Longitude':'Partner Long', 'Latitude':'Partner Lat'})
    data.head(2)
    
    years = [['Y1993', '1993'], ['Y1994', '1994'], ['Y1995','1995'], ['Y1996','1996'], ['Y1997','1997'],
        ['Y1998','1998'], ['Y1999','1999'], ['Y2000','2000'], ['Y2001','2001'], ['Y2002','2002'],
        ['Y2003','2003'], ['Y2004','2004'], ['Y2005','2005'], ['Y2006','2006'], ['Y2007','2007'],
        ['Y2008','2008'], ['Y2009','2009'], ['Y2010','2010'], ['Y2011','2011'], ['Y2012','2012'],
        ['Y2013','2013'], ['Y2014','2014'], ['Y2015','2015'], ['Y2016','2016']] 

    dataplot = []
    for i in years:
        # Production data
        dic = [dict(type='choropleth',
                 locations = prod['Area'].astype(str),
                 z=prod[i[0]].astype(float),
                 colorscale='Blues',
                 colorbar_title = 'Production',
                 locationmode='country names')]
        dataplot.append(dic[0].copy())

        data = data.sort_values([i[1]], axis = 0, ascending = False).reset_index(drop = True)

        # Trace data
        for j in range(100):
            dic = [dict(type='scattergeo',
                locationmode = 'country names',
                lon = [data['Reporter Long'][j], data['Partner Long'][j]],
                lat = [data['Reporter Lat'][j], data['Partner Lat'][j]],
                mode = 'lines',
                line = dict(width = 1,color = 'red'),
                opacity = float(data[i[1]][j]) / float(data[i[1]].max()))]
            dataplot.append(dic[0].copy())
            
    steps = []
    num = 100 + 1
    for i in range(len(years)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(dataplot)],
                    label='Year {}'.format(i + 1993))
        for j in range(num):
            step['args'][1][i*num+j] = True
        steps.append(step)
        
    
    sliders = [dict(active=0,
                pad={"t": 1},
                steps=steps)]    
    layout = dict(geo=dict(scope='world',
                  projection={'type': 'equirectangular'},
                  showcountries=True,
                  showframe=False),
                  margin={"r":0,"t":0,"l":0,"b":0},
                  showlegend = False,
                  sliders=sliders)
    fig = dict(data=dataplot, 
               layout=layout)

    plotly.offline.iplot(fig)