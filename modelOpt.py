from typing import Dict, List, Tuple
import pandas as pd
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt import hp
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os

class HPOpt(object):
    def __init__(self, 
                df: pd.DataFrame, 
                splits: List[Tuple[int]], 
                stock_id: int, 
                target: str, 
                featureGroup: int) -> None:

        self.df = df
        self.stockId = stock_id
        self.splits = splits
        self.target = target 
        self.featureGroup = featureGroup
        
        lgb_clf_params = {
        'learning_rate':    hp.uniform('learning_rate',0.001, 0.2),
        'max_depth':        hp.choice('max_depth',        np.arange(7, 100, 1, dtype=int)),
        'max_bin':          hp.choice('max_bin',          np.arange(7, 200, 1, dtype=int)),
        'num_leaves':       hp.choice('num_leaves',       np.arange(2, 100, 1, dtype=int)),
        'scale_pos_weight': hp.uniform('scale_pos_weight',0.01, 20),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.01, 0.9, 0.01)),
        'n_estimators':     100,
        'num_boost_round': 4000,
        'verbose':          -1,
        'random_state':     1,
                        }
        lgb_fit_params = {
            'eval_metric': 'logloss',
            'early_stopping_rounds': 200,
            'verbose': False
        }

        self.lgb_para = dict()
        self.lgb_para['clf_params'] = lgb_clf_params
        self.lgb_para['fit_params'] = lgb_fit_params

    def process(self, fn_name, space, trials, algo, max_evals):
        try:
            fn = getattr(self, fn_name)
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def lgb_clf(self, para):
        try:
            clf = LGBMClassifier(**para['clf_params'])
        except:
            print("An exception occurred") 
        return self.train(clf, para)
    
    def train(self, clf, para):
        tmp = []
        models = []
        for train, val in self.splits:
            train_days = train
            val_days = val
            featureCols = []
            with open(f'./selectedFeatures/{self.stockId}_{max(train_days)}_{self.featureGroup}.txt') as f:
                for line in f:
                    featureCols.append(line.strip())
            #Fit the classifier model
            train = lgb.Dataset(self.df[self.df['date'].isin(train_days)][featureCols], label=self.df[self.df['date'].isin(train_days)][self.target])
            val = lgb.Dataset(self.df[self.df['date'].isin(val_days)][featureCols], label=self.df[self.df['date'].isin(val_days)][self.target])
            clf = lgb.train(para['clf_params'], train, num_boost_round=4000, valid_sets=[val], early_stopping_rounds=200, verbose_eval=False)

            models.append({'model':clf, 'info':f'{max(train_days)}_{self.featureGroup}'})
            pred = (clf.predict(self.df[self.df['date'].isin(val_days)][featureCols]) > 0.5)*1
            loss = accuracy_score(self.df[self.df['date'].isin(val_days)][self.target], pred)
            tmp.append(loss)
        tmp = np.mean(tmp)
        return {'loss': 1-tmp, 'status': STATUS_OK, 'model':models}

    def getBestModelfromTrials(self, trials):
        valid_trial_list = [trial for trial in trials
                                if STATUS_OK == trial['result']['status']]
        losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
        index_having_minumum_loss = np.argmin(losses)
        best_trial_obj = valid_trial_list[index_having_minumum_loss]
        return best_trial_obj['result']['model']

if __name__ == "__main__":
    """
    df = pd.read_csv('./features/1.csv').dropna()
    rmvCols = ['date','time','last','open','high','low','close','volume','bid','ask']
    featureCols = [i for i in df.columns if i not in rmvCols]
    df['target'] = (df.ret > 0)*1
    #df.rename(columns={'ret':'target'}, inplace=True)
    # Hyper param training
    

    obj = HPOpt(featureCols, df, 'target')
    trials = Trials()
    lgb_opt = obj.process(fn_name='lgb_clf', space=obj.lgb_para, trials=trials, algo=tpe.suggest, max_evals=2)
    #Gets best models from trials and save
    models = obj.getBestModelfromTrials(trials)
    for i,model in enumerate(models):
        model.booster_.save_model(f'model_{i}.txt', num_iteration=model.booster_.best_iteration)
    print('asd')"""

    splits = [([i for i in range(1,17)], [i for i in range(18,21)]), 
                ([i for i in range(21,37)], [i for i in range(38,41)]), 
                ([i for i in range(41,57)],[i for i in range(58,61)])]
    stockIds = [i for i in range(1,51)]
    featureGroups = [5,10,15]
    targets = ['target','target1','target2']
    for stock in tqdm(stockIds):
        df = pd.read_csv(f'./features/{stock}.csv')
        for group in featureGroups:
            for target in targets:
                trials = Trials()
                obj = HPOpt(df=df, splits=splits, stock_id=stock, target=target,featureGroup=group)
                obj.process(fn_name='lgb_clf', space=obj.lgb_para, trials=trials, algo=tpe.suggest, max_evals=10)

                models = obj.getBestModelfromTrials(trials)

                path = f'./models/{stock}/'
                if not os.path.isdir(path):
                    os.makedirs(path)

                for model in models:
                    model['model'].save_model(path+f'{target}_'+model['info']+'.txt', 
                                            num_iteration=model['model'].best_iteration)
    pass
