import numpy as np
import pandas as pd
import plotly
import os
import json
import requests
import plotly.graph_objects as go

import numpy as np
import pandas as pd


def remove_f_years(df, year_min=[1961,1993], year_max=[2017,2020]):
    
    """
    Cleans data matrix by FAO-code columns, and FAO-flag columns. 
    Additionaly removes undesired years from matrices.
    
    :param df: pd.DataFrame The dataframe containing the import, export, production, or value data.
    :param year_min: range of minimum years to be removed [1981,1993], 1993 excluded.
    :param year_min: range of maximum years to be removed [2017,2020], 2020 excluded.
    :return: pd.DataFrame containing trimmed down data.
    """
    
    #Flags NaN mean "official data". Flag M means missing value. [NaN,NaN] in [Y#,Y#F] means zero.
    #Note: for "production value" dataset, Flags NaN is not explicitely reported as the "official data"
    for year in range(year_min[1],year_max[0]):
        yi="Y"+str(year)
        yf="Y"+str(year)+"F"
        df.loc[df[yi].isna() & df[yf].isna(), [yi]] = 0.0
    
    #Keep human readable columns not containign "Code" and "Y&F"
    df = df.drop(columns=[label for label in df.columns if 'Y' and 'F' in label])
    df = df.drop(columns=[label for label in df.columns if 'Code' in label])
    
    #Remove undesired years
    
    yr_list_min = ["Y"+str(year) for year in range(year_min[0],year_min[1])]
    yr_list_max = ["Y"+str(year) for year in range(year_max[0],year_max[1])]
    df = df.drop(columns=[year for year in df.columns if year in yr_list_min])
    df = df.drop(columns=[year for year in df.columns if year in yr_list_max])
    
    return df

def yearly_trade(df, item, unit='tonnes'):
    """
    Computes the total import and export quantities of a product for the whole df, for each year.
    
    :param df: pd.DataFrame The dataframe containing the import and export data. Must contain 'Item', 'Unit',
                'Reporter Countries', 'Partner Countries' and 'Element' columns.
    :param item: string The item for which we want the data.
    :param unit: string Either 'tonnes' or '1000 US$', depending if trade value or quantity is needed.
    :param reporter: bool If true, what the country reported. If false, what other countries reported.
    :return: pd.DataFrame Contains 'Import [unit]' and 'Export [unit]' columns, and a row for each year.
    """
    df = df[df['Item'] == item]
    df = df[df['Unit'] == unit]
    df = df[[f'Y{label}' for label in range(1993, 2017)] + ['Element']]
    df = pd.melt(df, id_vars=[f'Y{label}' for label in range(1993, 2017)],\
                 value_vars=['Element'], value_name='year').groupby('year').sum().transpose()
    if unit == 'tonnes':
        label = "Quantity"
    else:
        label = "Value"
    df = df.rename(columns={f"Export {label}": f"Exports", f"Import {label}": f"Imports"})
    return df


def yearly_trade_by_country(df, country, item, unit='tonnes', reporter=True):
    """
    Computes the total import and export quantities of a product for a given country, for each year.
    
    :param df: pd.DataFrame The dataframe containing the import and export data. Must contain 'Item', 'Unit',
                'Reporter Countries', 'Partner Countries' and 'Element' columns.
    :param country: string The country for which we want the data. 
    :param item: string The item for which we want the data.
    :param unit: string Either 'tonnes' or '1000 US$', depending if trade value or quantity is needed.
    :param reporter: bool If true, what the country reported. If false, what other countries reported.
    :return: pd.DataFrame Contains 'Import [unit]' and 'Export [unit]' columns, and a row for each year.
    """
    if reporter:
        df_country = df[df['Reporter Countries']== country]
    else:
        df_country = df[df['Partner Countries'] == country]
    df_country = yearly_trade(df_country, item, unit=unit)
    # If the country is not the reporter, the import and export columns are switched
    if not reporter:
        df_country = df_country.rename(columns={f"Exports": f"Imports", f"Imports": f"Exports"})
        df_country = df_country.iloc[:,[1, 0]]
    return df_country
    

def clean_trade_quantities(df):
    """
    Merges the trade matrix using the max method.
    
    :param df: the trade matrix.
    :return: the merged matrices.
    """
    df_importer = df[df['Element'] == 'Import Quantity']
    df_exporter = df[df['Element'] == 'Export Quantity']
    # Trades as reported by the importing country
    df_importer = df_importer.rename(columns={
        'Reporter Countries': 'Importer',
        'Partner Countries': 'Exporter',
    }).drop(['Element', 'Unit'], axis=1)

    # Trades as reported by the exporting country
    df_exporter = df_exporter.rename(columns={
        'Reporter Countries': 'Exporter',
        'Partner Countries': 'Importer',
    }).drop(['Element', 'Unit'], axis=1)

    # Rename the columns
    cols_i = [name for name in df_importer.columns]
    cols_e = [name for name in df_exporter.columns]
    cols_i[3:] = [col + 'I' for col in cols_i[3:]]
    cols_e[3:] = [col + 'E' for col in cols_e[3:]]
    df_importer.columns = cols_i
    df_exporter.columns = cols_e

    # Merge the dataframes
    df_trades_m = df_importer.merge(df_exporter, how='outer', on=['Importer', 'Exporter', 'Item']).fillna(0)

    # Take the max of both types of reporter
    df_clean_trades = df_trades_m[['Importer', 'Exporter', 'Item']]
    df_importer_reports = df_trades_m[['Y' + str(year) + 'I' for year in range(1993, 2017)]]
    df_importer_reports.columns = [str(year) for year in range(1993, 2017)]
    df_exporter_reports = df_trades_m[['Y' + str(year) + 'E' for year in range(1993, 2017)]]
    df_exporter_reports.columns = [str(year) for year in range(1993, 2017)]
    max_values = df_importer_reports.where(df_importer_reports > df_exporter_reports, df_exporter_reports)
    df_clean_trades[[str(year) for year in range(1993, 2017)]] = max_values
    
    return df_clean_trades

def TWarp(df, field, tags, normalize=True):
    
    """
    Performs time warping computation for a speccific set of time series, and returns mean and standard deviation.
    
    :param df: pd.DataFrame, The dataframe containing the import, export, production, or value data.
    :param field: string, specific field to look at like area, item, exporter, importer.
    :param tags: set, to look at i.e. ["China","Brazil"]. "All", computes for all. Integer, takes first N from df.
    :return: pd.DataFrame containing mean time warping distance, standard deviation, and number of elements computed.
    """
    
    if tags == "All":
        tags = df[field].unique()
    elif type(tags)==np.int:
        tags = df[field].unique()[0:tags]
        
    d_out = {}
    
    for tag in tags:
        #Select specific data based on field and tag
        df_test = df[df[field] == tag]
        
        #Remove any all zero data
        df_test = df_test[df_test.loc[:,"1993":].sum(axis=1)!=0]
        
        #For compatibility with dtw
        df_test = df_test.reset_index()
        
        #Extract data
        series = df_test.loc[:,"1993":"2016"].to_numpy(dtype=np.double)
        
        #Verify if there is enough data
        if len(series)<=1:
            d_out[tag] = [np.NaN,np.NaN,len(series)-1]
            continue
        
        #Normalize data
        if normalize:
            series = [scipy.stats.zscore(series[i]) for i in range(len(series))]
        
        #Compute time warping
        ds = dtw.distance_matrix_fast(series)
        
        #Replace inf with NAN
        ds[ds==np.inf] = np.nan
        
        #Compute and store statistical information
        d_out[tag] =  [np.nanmean(ds),np.nanstd(ds),(ds.shape[0]-1)]
        
    df_out = pd.DataFrame.from_dict(d_out,orient='index',columns=["mean","std","size"])
    return df_out


def correct_names(df, name, correct, wrong):
    """Function that returns the dataframe with the correct names for the countries.
    Name indicates the name of the column for which you have to change it"""
    for i in range(len(correct)):
        df.loc[df[name] == wrong[i], name] = correct[i]
    
    return df


def plot_imports(cs):
    
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
                 colorscale=cs,
                 colorbar_title = 'Imports (Tonnes)',
                 zmax = data['2016'].max(),
                 zmin = data['1993'].min(),
                 locationmode='country names')]
        dataplot.append(dic[0].copy())

    steps = []
    for i in range(len(dataplot)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(dataplot)],
                    label='Year {}'.format(i + 1993),
                   )
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


def plot_exports(cs):
    
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
                 colorscale=cs,
                 colorbar_title = 'Exports (Tonnes)',
                 zmax = data['2016'].max(),
                 zmin = data['1993'].min(),
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


def plot_globaltrade(Europe, cs, thick, th_max, th_min, th_max2, th_min2, op_per):
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
    
    correct3 =  ['EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU',
          'EU','EU','EU','EU','EU','EU','EU','EU','EU','EU', 'EU']
    wrong3 = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',  'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland',
     'Portugal', 'Romania', 'Slovakia',  'Slovenia', 'Spain', 'Sweden', 'United Kingdom', 'Cyprus']
    
    df_countries = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv') 
    df_coordinates = pd.read_csv('country_centroids.csv')
    df_coordinates = df_coordinates.loc[df_coordinates['homepart'] == 1]
    df_coordinates.loc[df_coordinates['sovereignt'] == 'United States of America', ['Longitude']] = -102.42
    df_coordinates.loc[df_coordinates['sovereignt'] == 'United States of America', ['Latitude']] = 38.75
    if Europe == True:
        df_coordinates = df_coordinates.append({'sovereignt': 'EU', 'Longitude': 9, 'Latitude': 53}, ignore_index = True)
    
    plotly.offline.init_notebook_mode(connected=True)
    plotly.offline.init_notebook_mode()
    
    df_prod = pd.read_pickle("data/df_prod.pkl")
    df_prod = correct_names(df_prod, 'Area', correct, wrong)
    df_prod = df_prod.loc[df_prod['Element'] == 'Production']
    prod = df_prod.groupby(['Area']).sum().reset_index()
    
    
    df_trade = pd.read_pickle('data/df_trade_merged.pkl')
    df_trade = correct_names(df_trade, 'Importer', correct2, wrong2)
    df_trade = correct_names(df_trade, 'Exporter', correct2, wrong2)
    
    if Europe == True:
        df_trade = correct_names(df_trade, 'Importer', correct3, wrong3)
        df_trade = correct_names(df_trade, 'Exporter', correct3, wrong3)
    
    
    trade = df_trade.groupby(['Importer', 'Exporter']).sum().reset_index()
    
    ind = [i for i in range (len(trade))  if trade['Importer'][i] == trade['Exporter'][i]]
    trade = trade.drop(ind, axis = 0)
    
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
    
    years2 = ['1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', 
        '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', 
        '2015', '2016']
    
    dataplot = []
    for i in years:
        # Production data
        dic = [dict(type='choropleth',
                 locations = prod['Area'].astype(str),
                 hoverinfo = "text",
                 hovertext = prod['Area'].astype(str) + ": " + prod[i[0]].astype(str) + " Tonnes",
                 z=prod[i[0]].astype(float),
                 colorscale=cs,
                 colorbar_title = 'Production (Tonnes)',
                 zmax = prod['Y2016'].max(),
                 zmin = prod['Y1993'].min(),
                 locationmode='country names')]
        dataplot.append(dic[0].copy())

        data = data.sort_values([i[1]], axis = 0, ascending = False).reset_index(drop = True)

        # Trace data
        for j in range(100):
            
            # Markers
            total_imp = float(data[data["Importer"] == data["Importer"][j]][i[1]].sum())
            max_imp = float(data[years2].sum().sum())
            out_th = (total_imp/max_imp)*(th_max-th_min) + th_min
            
            # Lines
            total_tr = data[i[1]][j]
            out_th_line = (total_tr/max_imp)*(th_max2-th_min2) + th_min2
            
            # Opac
            out_opacity = (float(data[i[1]][j]) / float(data[i[1]].max()))*op_per + (1-op_per)
            
            dic = [dict(type='scattergeo',
                locationmode = 'country names',
                lon = [data['Reporter Long'][j], data['Partner Long'][j]],
                lat = [data['Reporter Lat'][j], data['Partner Lat'][j]],
                hoverinfo = "skip",
                mode = 'lines',
                visible = False,
                line = dict(width = out_th_line,color = 'red'),
                opacity = out_opacity)]
            dataplot.append(dic[0].copy())
            
            dic = [dict(type='scattergeo',
                locationmode = 'country names',
                lon = [data['Reporter Long'][j]],
                lat = [data['Reporter Lat'][j]],
                hoverinfo = "text",
                text = 'Import: ' + str(round(total_imp)) + ' Tonnes',
                mode = 'markers',
                visible = False,
                marker = dict(size = out_th, color = 'red'),
                opacity = float(1))]
            dataplot.append(dic[0].copy())

            
    steps = []
    #num = 50 + 1
    num = 200
    for i in range(len(years)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(dataplot)],
                    label='Year {}'.format(i + 1993))
        for j in range(num):
            step['args'][1][i*num+j] = True
            step['args'][1][i*num+j+1] = True
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

    
def production_trade(product, Europe, cs, thick, th_max, th_min, th_max2, th_min2, op_per):
    
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
    
    correct3 =  ['EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU','EU',
          'EU','EU','EU','EU','EU','EU','EU','EU','EU','EU', 'EU']
    wrong3 = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',  'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland',
     'Portugal', 'Romania', 'Slovakia',  'Slovenia', 'Spain', 'Sweden', 'United Kingdom', 'Cyprus']
    
    
    df_countries = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv') 
    df_coordinates = pd.read_csv('country_centroids.csv')
    df_coordinates = df_coordinates.loc[df_coordinates['homepart'] == 1]
    df_coordinates.loc[df_coordinates['sovereignt'] == 'United States of America', ['Longitude']] = -102.42
    df_coordinates.loc[df_coordinates['sovereignt'] == 'United States of America', ['Latitude']] = 38.75
    if Europe == True:
        df_coordinates = df_coordinates.append({'sovereignt': 'EU', 'Longitude': 9, 'Latitude': 53}, ignore_index = True)
    
    
    df_prod = pd.read_pickle("data/df_prod.pkl")
    df_prod = correct_names(df_prod, 'Area', correct, wrong)
    prod = df_prod.loc[df_prod['Item'] == product]
    prod = prod.loc[prod['Element'] == 'Production']
    
    df_trade = pd.read_pickle('data/df_trade_merged.pkl')
    df_trade = correct_names(df_trade, 'Importer', correct2, wrong2)
    df_trade = correct_names(df_trade, 'Exporter', correct2, wrong2)
    trade = df_trade.loc[df_trade['Item'] == product]
    
    trade = trade.reset_index()
    
    if Europe == True:
        trade = correct_names(trade, 'Importer', correct3, wrong3)
        trade = correct_names(trade, 'Exporter', correct3, wrong3)
        
    ind = [i for i in range (len(trade))  if trade['Importer'][i] == trade['Exporter'][i]]
    trade = trade.drop(ind, axis = 0)
    
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
    
    years2 = ['1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', 
        '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', 
        '2015', '2016']

    dataplot = []
    for i in years:
        # Production data
        dic = [dict(type='choropleth',
                 locations = prod['Area'].astype(str),
                 z=prod[i[0]].astype(float),
                 hoverinfo = "text",
                 hovertext = prod['Area'].astype(str) + ": " + prod[i[0]].astype(str) + " Tonnes",
                 colorscale=cs,
                 colorbar_title = 'Production (Tonnes)',
                 zmax = prod['Y2016'].max(),
                 zmin = prod['Y1993'].min(),
                 locationmode='country names')]
        dataplot.append(dic[0].copy())

        data = data.sort_values([i[1]], axis = 0, ascending = False).reset_index(drop = True)

        # Trace data
        for j in range(50):
            
            # Markers
            total_imp = float(data[data["Importer"] == data["Importer"][j]][i[1]].sum())
            max_imp = float(data[years2].sum().sum())
            out_th = (total_imp/max_imp)*(th_max-th_min) + th_min
            
            # Lines
            total_tr = data[i[1]][j]
            out_th_line = (total_tr/max_imp)*(th_max2-th_min2) + th_min2
            
            # Opac
            out_opacity = (float(data[i[1]][j]) / float(data[i[1]].max()))*op_per + (1-op_per)
            
            
            dic = [dict(type='scattergeo',
                locationmode = 'country names',
                lon = [data['Reporter Long'][j], data['Partner Long'][j]],
                lat = [data['Reporter Lat'][j], data['Partner Lat'][j]],
                hoverinfo = "skip",
                mode = 'lines',
                visible = False,
                line = dict(width = out_th_line,color = 'red'),
                opacity = out_opacity)]
            dataplot.append(dic[0].copy())
            
            dic = [dict(type='scattergeo',
                locationmode = 'country names',
                lon = [data['Reporter Long'][j]],
                lat = [data['Reporter Lat'][j]],
                hoverinfo = "text",
                text = 'Import: ' + str(round(total_imp)) + ' Tonnes',
                mode = 'markers',
                visible = False,
                marker = dict(size = out_th, color = 'red'),
                opacity = float(1))]
            dataplot.append(dic[0].copy())

            
    steps = []
    #num = 50 + 1
    num = 100
    for i in range(len(years)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(dataplot)],
                    label='Year {}'.format(i + 1993))
        for j in range(num):
            step['args'][1][i*num+j] = True
            step['args'][1][i*num+j+1] = True
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

    
    
def production_price(product, cs):
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
    
    df_countries = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv') 
    

    df_prod = pd.read_pickle("data/df_prod.pkl")
    df_prod = correct_names(df_prod, 'Area', correct, wrong)
    prod_bananas = df_prod.loc[df_prod['Item'] == product]
    prod_bananas = prod_bananas.loc[prod_bananas['Element'] == 'Production'][['Area', 'Y2016']]
    
    df_imp = pd.read_pickle("data/df_trade_merged.pkl")
    df_imp = correct_names(df_imp, 'Importer', correct, wrong)
    imp_bananas = df_imp.loc[df_imp['Item'] == product]
    imp_bananas = pd.DataFrame(imp_bananas.groupby('Importer')['2016'].sum()).reset_index()
    
    df_imp = pd.read_pickle("data/df_imp.pkl")
    df_imp = correct_names(df_imp, 'Reporter Countries', correct, wrong)
    imp_usd = df_imp.loc[df_imp['Item'] ==  product]
    imp_usd = pd.DataFrame(imp_usd.groupby('Reporter Countries')['Y2016'].sum()).reset_index()
    
    df_exp = pd.read_pickle("data/df_trade_merged.pkl")
    df_exp = correct_names(df_exp, 'Exporter', correct, wrong)
    exp_bananas = df_exp.loc[df_exp['Item'] == product]
    exp_bananas = pd.DataFrame(exp_bananas.groupby('Exporter')['2016'].sum()).reset_index()
    
    df_exp = pd.read_pickle("data/df_exp.pkl")
    df_exp = correct_names(df_exp, 'Reporter Countries', correct, wrong)
    exp_usd = df_exp.loc[df_exp['Item'] == product]
    exp_usd = pd.DataFrame(exp_usd.groupby('Reporter Countries')['Y2016'].sum()).reset_index()
     
    plotly.offline.init_notebook_mode(connected=True)
    plotly.offline.init_notebook_mode()
    
    data = df_countries.merge(prod_bananas, left_on = 'COUNTRY', right_on = 'Area', how = 'left').drop(['Area'], axis = 1)\
    .rename(columns = {'Y2016':'Production'})\
    .merge(imp_bananas, left_on = 'COUNTRY', right_on = 'Importer', how = 'left')\
    .drop(['Importer'], axis = 1).rename(columns = {'2016':'Imports tonnes'})\
    .merge(exp_bananas, left_on = 'COUNTRY', right_on = 'Exporter', how = 'left')\
    .drop(['Exporter'], axis = 1).rename(columns = {'2016':'Exports tonnes'})\
    .merge(imp_usd, left_on = 'COUNTRY', right_on = 'Reporter Countries', how = 'left')\
    .drop(['Reporter Countries'], axis = 1).rename(columns = {'Y2016':'Imports usd/tonnes'})\
    .merge(exp_usd, left_on = 'COUNTRY', right_on = 'Reporter Countries', how = 'left')\
    .drop(['Reporter Countries'], axis = 1).rename(columns = {'Y2016':'Exports usd/tonnes'})
    
    data = data.replace(np.nan, 0)
    
    info = ['Production', 'Imports tonnes', 'Exports tonnes', 'Imports usd/tonnes', 'Exports usd/tonnes']
    dataplot = []
    for i in info:
        dic = [dict(type='choropleth',
                 locations = data['COUNTRY'].astype(str),
                 z=data[i].astype(float),
                 colorscale=cs,
                 colorbar_title = i,
                 locationmode='country names')]
        dataplot.append(dic[0].copy())
        
    buttons = []
    for i in range(len(info)):
        but = dict(method='restyle',
                    args=['visible', [False] * len(info)],
                    label=info[i])
        but['args'][1][i] = True
        buttons.append(but)
        
    menus=list([dict(
            x=-0.05,
            y=1,
            yanchor='top',
            buttons= buttons)])
    
    layout = dict(geo=dict(scope='world',
              projection={'type': 'equirectangular'},
              showframe=False),
              margin={"r":0,"t":0,"l":0,"b":0},
              mapbox=dict(zoom = 1.5),
              updatemenus = menus)
    
    fig = dict(data=dataplot, 
           layout=layout)
    plotly.offline.iplot(fig)

    
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