from typing import Dict
import pandas as pd
from preprocess import preprocessor
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
class features:
    def __init__(self, data: pd.DataFrame, outputPath: str, trainFlag: bool) -> None:

        #Stores global features such as covariance matrix etc, prevents for recalculations.
        #Store as a dict, with dataframes inside as values and keys as feature name
        self.df = data
        self.outputPath = outputPath
        self.targetsDict = {}
        self.globalFeatures = {}
        self.trainFlag = trainFlag
        #Save percentiles from 
        self.percentiles = None
        pass

    def createGlobalFeatures(self) -> None:
        """Creates global features and persists them to save time
        """
        #Create covariances
        def covariance(tmp, idx, window):
            return tmp.iloc[idx-window:idx].cov()

        cov50Dict = {}
        window = 50
        print('Creating global features: Covariances...')
        for i in tqdm(range(window, self.df.shape[0]+1)):
            tmp2 = self.df['last'].iloc[i-window:i]
            cov50Dict[tmp2.index.max()] = tmp2.cov().reset_index(drop=True)

        self.globalFeatures['covariance'] = cov50Dict

        #Create bid/ask spreads for all stocks
        print('Creating global features: Bid ask ratios')
        bidAskRatioDf = self.df['bid']/self.df['ask']
        bidAskRatioDf.columns = [f'bidAskRatio_{i}' for i in range(1,51)]
        self.globalFeatures['bidAskRatio'] = bidAskRatioDf


    def outliers(self,df: pd.DataFrame) -> pd.DataFrame:
        """Outliers are identified when the spread is unusually large (greater than 0.99 percentile). 
        We remove these rows. Alternative will be to forward fill or shrink to median.
        """
        keepCols = df.columns
        df['spread'] = (df.ask-df.bid)
        percentile = df.spread.quantile(0.95)
        df = df[df.spread < percentile]
        #add to self.percentiles for use in test data
        self.percentiles = percentile
        return df[keepCols]

    def createFeatures(self) -> None:
        stockIds = [i for i in range(1, 51)]
        #std of returns
        print('Creating features...')
        for stock in tqdm(stockIds):
            df = self.df.xs(stock, level=1, axis=1)

            #Only remove outliers during train as it uses forward looking.
            if self.trainFlag == True:
                df = self.outliers(df)
            else:
                #load spread 95 percentile and filter test using train data (to avoid forward looking bias)
                percentile = []
                with open(f"./features/{stock}.txt", 'r') as f:
                    for line in f:
                        percentile.append(float(line.rstrip()))
                df['spread'] = df.ask-df.bid
                df = df[~(df.spread > percentile[0])]
                df = df.drop('spread',axis=1)

            #calculate mid
            df['mid'] = (df.ask+df.bid)/2
            
            #Calculate historical returns over different horizons
            df['ret'] = (df['mid']/df['mid'].shift(1) - 1)
            df['ret5'] = (df['mid']/df['mid'].shift(5) - 1)
            df['ret15'] = (df['mid']/df['mid'].shift(15) - 1)
            df['ret30'] = (df['mid']/df['mid'].shift(30) - 1)

            #Calculate vol 5-100 timesteps
            df['vol_5'] = df.ret.rolling(window=5).std()
            df['vol_10'] = df.ret.rolling(window=10).std()
            df['vol_20'] = df.ret.rolling(window=20).std()
            df['vol_50'] = df.ret.rolling(window=50).std()
            df['vol_100'] = df.ret.rolling(window=100).std()

            #Calculate rolling mean 5-100 timesteps of returns
            df['maRet_5'] = df['ret'].rolling(window=5).mean()
            df['maRet_10'] = df['ret'].rolling(window=10).mean()
            df['maRet_20'] = df['ret'].rolling(window=20).mean()
            df['maRet_50'] = df['ret'].rolling(window=50).mean()
            df['maRet_100'] = df['ret'].rolling(window=100).mean()

            #Calculate ewma of ret
            df['ewma_1'] = df['ret'].ewm(com=0.1).mean()
            df['ewma_2'] = df['ret'].ewm(com=0.2).mean()
            df['ewma_3'] = df['ret'].ewm(com=0.3).mean()
            df['ewma_4'] = df['ret'].ewm(com=0.4).mean()
            df['ewma_5'] = df['ret'].ewm(com=0.5).mean()
            df['ewma_6'] = df['ret'].ewm(com=0.6).mean()
            df['ewma_7'] = df['ret'].ewm(com=0.7).mean()
            df['ewma_8'] = df['ret'].ewm(com=0.8).mean()
            df['ewma_9'] = df['ret'].ewm(com=0.9).mean()

            #Add volume related features.
            df['volumeChng'] = df['volume']/df['volume'].shift(1)
            df['volume5Mean'] = df.volumeChng.rolling(5).mean()
            df['volume10Mean'] = df.volumeChng.rolling(10).mean()
            df['volume50Mean'] = df.volumeChng.rolling(50).mean()
            df['volume5Std'] = df.volumeChng.rolling(5).std()
            df['volume10Std'] = df.volumeChng.rolling(10).std()
            df['volume50Std'] = df.volumeChng.rolling(50).std()
          
            #Add covariances of stocks globalFeatures['covariance']
            keys = [key for key in self.globalFeatures['covariance'].keys()]
            stockCov = [pd.DataFrame(self.globalFeatures['covariance'].get(key)[stock]).T 
                                        for key in keys]
            stockCov = pd.concat(stockCov)
            stockCov.columns = [f'cov50_{i}' for i in range(1,51)]
            stockCov.index = keys
            stockCov.index = pd.MultiIndex.from_tuples(stockCov.index)
            stockCov.index.names=['date','time']
            df = pd.merge(df, stockCov, left_index=True, right_index=True)

            #Add globalFeatures['bidAskRatio']
            df = pd.merge(df, self.globalFeatures['bidAskRatio'], left_index=True, right_index=True)

            #Create targets, 5/15/30 minute forward looking returns - using 'mid'.  We use target 1/2 as intermediary, and our actual target is 'target' (30 minutes).
            #We do this to create robustness in our models.
            #Each target is shifted by -1. This is such that at time T, we predict 30step forward return from T=1. This is done for 2 reason, to mimic reality as we require 1 time
            #step to enter the trade + avoid forward looking bias
            df['target'] = ((df['mid']/df['mid'].shift(-30) - 1).shift(-1) > 0)*1
            df['target1'] = ((df['mid']/df['mid'].shift(-5) - 1).shift(-1) > 0)*1
            df['target2'] = ((df['mid']/df['mid'].shift(-15) - 1).shift(-1) > 0)*1

            #Dropna (due to rolling windows) - should only affect day 1
            df.dropna(inplace=True)

            #Combine features df with targets
            df.to_csv(f'{self.outputPath}/{stock}.csv')

            if self.trainFlag == True:
                #save percentile for outlier filtering
                with open(f'{self.outputPath}/{stock}.txt', 'w') as f:
                    f.write(f'{self.percentiles}')

    def run(self):
        """Generates the features for a given stock
        """
        self.createGlobalFeatures()
        self.createFeatures()
        

if __name__ == "__main__":
    a = preprocessor(dataPath='./trainData/')
    a.dataLoader()
    a.missingVal()

    b = features(a.df, './features', trainFlag=True)
    b.run()
    #b.createTargets()

    #b.createFeatures()
 