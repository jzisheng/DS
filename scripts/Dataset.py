import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset  # For custom datasets

class DfDataset(Dataset):
    def __init__(self, df,transform2d=True):
        """
        Args:
            df (pd.DataFrame) dataframe to iterate over
        """
        self.transform2d = transform2d
        self.data = df
        self.labels = np.asarray(self.data['isFraud']).astype('float')
        self.cols = ['transactionAmount', 'hourDelta',
                     'transactionAmountMedian', 'transactionAmountMean',
                     'transactionAmountStd', '0_encodedTime', '1_encodedTime',
                     '2_encodedTime', '3_encodedTime', '6_encodedTime',
                     'auto_merchantCategoryCode', 'entertainment_merchantCategoryCode',
                     'fastfood_merchantCategoryCode', 'food_merchantCategoryCode',
                     'food_delivery_merchantCategoryCode', 'fuel_merchantCategoryCode',
                     'gym_merchantCategoryCode', 'health_merchantCategoryCode',
                     'mobileapps_merchantCategoryCode', 'online_retail_merchantCategoryCode',
                     'personal care_merchantCategoryCode', 'rideshare_merchantCategoryCode']
        self.data = self.data[self.cols]

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index])
        dfRow = self.data.iloc[index]
        dfX = (dfRow.values)
        if self.transform2d:
            dfRow = np.array([np.pad(dfX,(0,3),'constant',
                                     constant_values=(0)).reshape(5,5)])
        else:
            dfRow = np.array([dfX])
        data_as_tensor = torch.tensor(dfRow).float()
        return (data_as_tensor, label)
        

    def __len__(self):
        return len(self.data.index)




class JSONDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.reader = pd.read_json('transactions.txt',lines=True,chunksize=70000)
        self.data = next(self.reader).fillna(0)
        self.to_tensor = transforms.ToTensor()
        self.labels = np.asarray(self.data['isFraud']).astype('float')
        self.cols = ['availableMoney','creditLimit',
                     'transactionAmount','posEntryMode'] 

    def __getitem__(self, index):
        try:
            label = torch.tensor(self.labels[index])

            dfRow = self.data.iloc[index][self.cols]
            
            data_as_tensor = torch.tensor(dfRow)
            
            return (data_as_tensor, label)
        except TypeError:
            label = torch.tensor(self.labels[index])
            dfRow = self.data.iloc[index][self.cols]
            dfRow.posEntryMode = 0
            
            data = np.asarray(dfRow).reshape(2,2).astype('float')
            data_as_tensor = torch.tensor(data)
            
            return (data_as_tensor, label)            
        

    def __len__(self):
        return len(self.data.index)

