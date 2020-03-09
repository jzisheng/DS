import math
import numpy as np
import pandas as pd

"""
Methods for part 2
Analyzing transactions which are duplicate Txs or reverse
charge backs
"""

"""
Method for predicting duplicates for a given user. The motivation
behind this is to first identify identical transaction amounts
and if matchin tx amonut, check if the merchant name is the same.

@param df for all accounts
@param accountNumber for which account want to flag duplicates
"""
def predictDuplicateByUser(df,accountNumber):
    mask = df.accountNumber == accountNumber
    oneAccountTxs = df[mask].sort_values(['transactionDateTime'])
    ##### Use pandas dataframe to detect duplicated transactions and merchants
    duplicatedDf = oneAccountTxs.duplicated(subset=['transactionAmount','merchantName'],keep=False)
    duplicateTxs = oneAccountTxs[duplicatedDf]
    # calculate time difference for suspected duplicated txs
    merchants = np.unique(duplicateTxs['merchantName'])
    
    results = []
    chargeBackResults = []
    for merchant in merchants:
        mask = (duplicateTxs['merchantName']==merchant)
        byMerchantDf = duplicateTxs[mask]
        duplicateFlags = predictDuplicates(byMerchantDf)
        #chargebackFlags = predictChargebacks(byMerchantDf)
        results.append(byMerchantDf[duplicateFlags])
        #results.append(byMerchantDf[chargebackFlags])
        pass
    if(len(results)>1):
        return results

"""
Method for predicting duplicates

@param df for one user accounts interactions at one merchant
@returns mask flagging what are potential duplicate Txs
"""
def predictDuplicates(df):
    df.sort_values('transactionDateTime')
    delta = df['transactionDateTime'].diff()
    return ((delta.dt.seconds < 5000) & (df.transactionType=='PURCHASE'))


def predictChargebacks(df):
    df.sort_values('transactionDateTime')
    mask = (df.transactionType == 'REVERSAL')
    cols = ['transactionDateTime','transactionAmount',
            'currentBalance','transactionType','merchantName']
    pass
                                   
"""
Methods for Part 3
Analyzing model
"""

"""
Applying transformations to time
"""
from sklearn.utils import resample

def upsampleMinority(X,colStr='isFraud'):
    # separate minority and majority classes
    not_fraud = X[X[colStr]==0]
    fraud = X[X[colStr]==1]
    # upsample minority
    fraud_upsampled = resample(fraud,
                               replace=True, # sample with replacement
                               n_samples=len(not_fraud), # match number in majority class
                               random_state=27) # reproducible results
    return pd.concat([not_fraud,fraud_upsampled])

cols = ['transactionAmount']
all_cols = ['transactionAmount','transactionDateTime','isFraud','merchantCategoryCode']

from sklearn.preprocessing import normalize, LabelEncoder, OneHotEncoder

"""
Method that encodes a column
@param df This is the dataframe to be encoded
@param col This is a string that is the column to be encoded
"""
def encodeColumn(df,col='encodedTime'):
    cat_types = np.unique(df[col])
    cat_df = pd.DataFrame(cat_types, columns=['cat_types'])
    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    cat_df['cat_cat'] = labelencoder.fit_transform(cat_df['cat_types'])
    # creating instance of one-hot-encoder
    enc = OneHotEncoder(handle_unknown='ignore')
    # passing bridge-types-cat column (label encoded values of bridge_types)
    enc_df = pd.DataFrame(enc.fit_transform(cat_df[['cat_cat']]).toarray())
    # Create new column names for OHE
    newColNames = []
    for idx,oheCol in enumerate(cat_types):
        newColNames.append("{}_{}".format(oheCol,col))
    enc_df.columns = newColNames
    # merge with main df and enc_df
    transactions_df = df.drop(columns=[col])
    df = transactions_df.join(enc_df).fillna(0)
    return df

def calculateUser(df,normalizeRows=False):
    """
    Iterates through dataframe and performs calculation
    by each user
    """
    customerIds = df.customerId.unique()
    result = []
    for customerId in customerIds:
        mask = df.customerId == customerId
        customerDf = df[mask]
        customerDf = customerDf[all_cols]
        customerDf['transactionAmount'] = normalize(customerDf['transactionAmount']
                                                    .values.reshape(-1,1),axis=0)
        encodedDf = encodeTimes(customerDf)
        # calculate the difference in hours between transactions
        encodedDf['hourDelta']=(encodedDf['transactionDateTime']
                                .diff().astype('timedelta64[h]'))
        encodedDf['hourDelta'] = normalize(encodedDf['hourDelta']
                                           .fillna(0).values.reshape(-1,1),axis=0)
        calculateByEncodedTime(encodedDf)
        calculateByEncodedTime(encodedDf,col='transactionAmount',fstr='Mean',f=np.mean)
        calculateByEncodedTime(encodedDf,col='transactionAmount',fstr='Std',f=np.std)
        result.append(encodedDf)
        pass
    return pd.concat(result).fillna(0)


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

