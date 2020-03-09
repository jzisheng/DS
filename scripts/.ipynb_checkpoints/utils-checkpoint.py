import math
import numpy as np
import pandas as pd
"""
Applying transformations to time
"""
def calculateUser(df):
    """
    Iterates through dataframe and performs calculation
    by each user
    """
    customerIds = df.customerId.unique()
    result = []
    for customerId in customerIds:
        mask = df.customerId == customerId
        encodedDf = encodeTimes(df[mask])
        # calculate the difference in hours between transactions
        encodedDf['hourDelta']=(encodedDf['transactionDateTime'].diff().astype('timedelta64[h]'))
        calculateByEncodedTime(encodedDf)
        calculateByEncodedTime(encodedDf,col='transactionAmount',fstr='Mean',f=np.mean)
        calculateByEncodedTime(encodedDf,col='transactionAmount',fstr='Std',f=np.std)
        result.append(encodedDf)
        pass
    return result


def calculateByEncodedTime(df,col='transactionAmount',fstr='Median',f=np.median):
    """
    Given a dataframe, iterate by column and calculate the median by the aggregated
    encoded time
    for each encoded time, calculate f(median) of col(transactionAmount)
    input:
        df with columns : accountNumber | transactionAmount | encodedTime |
                                0             10000                1
    """
    encodedTimes = np.unique(df.encodedTime)
    for enc in encodedTimes:
        mask = (df.encodedTime == enc)
        df.loc[mask,str(col+fstr)] = f(df[mask][col])
        pass

def encodeTimes(df):
    """
    Takes as input df, returns a df with
    the transactionDateTime encoded
    representing the time of day
    input:
        df from dataset
    returns:
        df[encodedTime] = 0 , 1 , 2
    """
    df = df.sort_values('transactionDateTime')
    df['encodedTime'] = df.apply(encodeTimeRow,axis=1)
    return df

def encodeTimeRow(row,ps = [0,6,11,17,23,25]):
    """
    Takes as input hour times, and percentiles
    which represents the quantiles
    default ps were selected based on dataset
    input:
        times: [5, 6, 10 ,11 ,12, ..., N]
        ps: [5, 10, 16] 5 AM, 10 AM, 4 PM - inf
    """
    rowHour = row['transactionDateTime'].hour
    for idx,pmin in enumerate(range(len(ps)-1)):
        pmin,pmax = ps[idx],ps[idx+1]
        if (rowHour > pmin and
            rowHour < pmax):
            return idx
        pass
    return (len(ps))
