import pandas as pd 
import os
import numpy as np


class preprocessor:
    def __init__(self, dataPath: str) -> None:
        """Class initialiser

        Args:
            dataPath (str): Path for data files 
        """
        if dataPath[-1] == '/':
            self.dataPath = dataPath
        else:
            self.dataPath = dataPath + '/'

        #Stores data from csv
        self.df = []

        #Stores targets for all stocks
        self.targetsDict = {}

    def dataLoader(self) -> None:
        """Loads data from a given directory
        """
        assert os.path.isdir(self.dataPath), f'{self.dataPath} does not exist'
        dataFiles = os.listdir(self.dataPath)
        for file in dataFiles:
            self.df.append(pd.read_csv(self.dataPath+file))
        self.df = pd.concat(self.df)
        self.df = self.df.pivot(index=['date','time'],columns='ID')

    def missingVal(self) -> None:
        """Fills in missing values.
        Considering we have no meta knowledge (e.g. stocks/exchange/timezone), we use the following logic for handling the missing values.
            1. If all stocks have missing values at a given moment, then we assume there is a trading break/halt. In this case we drop the row.
               This still holds information, however it can be represented by the time column. For example, after break - higher vol.
            2. If only partial stocks have missing values (for some columns) then we forward fill
        """
        self.df['missingCount'] = self.df.isna().sum(axis=1)
        self.df = self.df[~(self.df.missingCount == 400)]
        self.df.drop('missingCount', axis=1, inplace=True)
        self.df = self.df.ffill()
    
    def run(self) -> None:
        self.dataLoader()
        self.missingVal()

if __name__ == "__main__":
    a = preprocessor(dataPath='./data/')
    a.dataLoader()
    a.missingVal()
    a.outliers()
    a.createTargets()
    print(a.targetsDict)