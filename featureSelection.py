import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
from itertools import compress
from tqdm import tqdm
import random
import os
from typing import List, Tuple

class select:
    def __init__(self, inputPath: str, outputPath: str, splits: List[Tuple[int]]) -> None:
        """Initialise class

        Args:
            inputPath (str): input dir for the features files
            outputPath (str): output dir to save the selected features. The format is saved in as follows {stockId}_{trainFoldIdx}_{num_features}
            splits (List[Tuple[int]]): Train/Validation splits.  We only use train here
        """

        self.inputPath = inputPath
        self.outputPath = outputPath
        self.splits = splits

    def extract_features(self, data: pd.DataFrame, num_features: int, features: List[str]) -> None:
        # Use lightgbm, with max_depth 1 to get feature importances. We use num_leaves=2 to avoid overfitting 
        # due to lack of data
        clf = LGBMClassifier(num_leaves=2, max_depth=1, random_state=2020)
        selector = RFE(clf, n_features_to_select=num_features, step=1)
        selector = selector.fit(data[features], data.target)
        return list(compress(features, selector.support_))

    def run(self):
        stockIds = [i for i in range(1,51)]
        #Saves the selected features in the following txt format {stock_id}_{trainFoldIDX}_{num_features}
        for stock in tqdm(stockIds):
            df = pd.read_csv(f'./features/{stock}.csv')
            filterCols = ['date','time','last','open','high','low','close','volume','bid','ask','target1','target2','target']
            featureCols = [i for i in df.columns if i not in filterCols]
            #Group size = 5,10 and 15
            for split in self.splits:
                #index of the train period
                foldIdx = max(split[0])
                df2 = df[df.date.isin(split[0])]
                feature_group = {}
                for group_size in [5,10,15]:
                    tmp = []
                    #we run a sample of 20 for each group_size such that reasonable amount of feature variations are considered.
                    for i in range(0,20):
                        #set random seed for reproducibility.
                        random.seed(i*10)
                        #select random 30 features from our full feature universe (this is chosen randomly)
                        features_filtered = self.extract_features(df2, group_size, random.sample(featureCols,30))
                        tmp.append(features_filtered)

                    tmp = [list(x) for x in set(tuple(np.sort(x)) for x in tmp)]
                    tmp = pd.DataFrame(np.transpose(tmp))
                    tmp.columns = [f'group_{i}' for i in range(0, tmp.shape[1])]
                    tmp['num_params'] = group_size
                    feature_group[group_size] = tmp

                    #save selected features
                    for key in feature_group.keys():
                        tmp = feature_group[key]
                        groupCols = tmp.columns[tmp.columns.str.contains('group_')]
                        tmp = tmp[groupCols].unstack().reset_index()
                        keepFeatures = tmp.groupby(0).count().sort_values('level_0', ascending=False).head(key).index
                        with open(f'{self.outputPath}/{stock}_{foldIdx}_{key}.txt', 'w') as f:
                            for line in keepFeatures:
                                f.write(f'{line}\n')


if __name__ == "__main__":
    splits = [([i for i in range(1,17)], [i for i in range(18,21)]), 
                ([i for i in range(21,37)], [i for i in range(38,41)]), 
                ([i for i in range(41,57)],[i for i in range(58,61)])]
    a = select(outputPath='./selectedFeatures',inputPath='./features', splits=splits)
    a.run()