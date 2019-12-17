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