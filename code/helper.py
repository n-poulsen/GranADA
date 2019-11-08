import numpy as np
import pandas as pd


def remove_f_years(df):
    df = df.drop(columns=[label for label in df.columns if 'Y' and 'F' in label])
    df = df.drop(columns=[label for label in df.columns if 'Code' in label])
    df = df.replace(np.NaN,0.0)
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
    df = df[[f'Y{label}' for label in range(1986, 2017)] + ['Element']]
    df = pd.melt(df, id_vars=[f'Y{label}' for label in range(1986, 2017)],\
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
    