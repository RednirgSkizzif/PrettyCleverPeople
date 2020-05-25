import pandas as pd 
import numpy as np
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import matplotlib.pyplot as plt
import pdb


class data_utils(object):
    """docstring for data_utils"""
    def __init__(self,target_symbol,feature_shape,target_shape):
        super(data_utils, self).__init__()

        self.target = target_symbol
        self.feature_transformer = QuantileTransformer(n_quantiles=100,
                                            output_distribution=feature_shape,
                                                    copy=True)

        self.target_transformer = QuantileTransformer(n_quantiles=100,
                                           output_distribution=target_shape,
                                                    copy=True)
        # self.feature_transformer = MinMaxScaler()

        # self.target_transformer = MinMaxScaler()

    def create_features(self,df,
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        test_start_date,
        test_end_date,
        back=30,
        forward=30,
        trend_parameter=-1,
        take_log=False):

        X = df.drop(columns=[col for col in df.columns])
        no_repeat = []
        for col_1 in df:
            for col_2 in df:
                if col_1 == col_2:
                    continue
                no_repeat.append(col_1)
                if col_2 in no_repeat:
                    continue
                if trend_parameter == -1:
                    if take_log:
                        X[col_1.split('_')[0]+'-'+col_2.split('_')[0]] =\
                         (np.log(df[col_1])- np.log(df[col_2])) - (np.log(df[col_1])-np.log(df[col_2])).mean()
                    else:
                        X[col_1.split('_')[0]+'-'+col_2.split('_')[0]] =\
                         (df[col_1]-df[col_2]) - (df[col_1]-df[col_2]).mean()
                else:
                    if take_log:
                        X[col_1.split('_')[0]+'-'+col_2.split('_')[0]] =\
                         (np.log(df[col_1])- np.log(df[col_2])) - (np.log(df[col_1])-np.log(df[col_2])).rolling(trend_parameter).mean()
                    else:
                        X[col_1.split('_')[0]+'-'+col_2.split('_')[0]] =\
                         (df[col_1]-df[col_2]) - (df[col_1]-df[col_2]).rolling(trend_parameter).mean()
                
        self.X_train = X[train_start_date:train_end_date]
        self.X_train_dates = X[train_start_date:train_end_date].index
        self.X_val = X[val_start_date:val_end_date]
        self.X_val_dates = X[val_start_date:val_end_date].index
        self.X_test = X[test_start_date:test_end_date]
        self.X_test_dates = X[test_start_date:test_end_date].index

        return self.X_train, self.X_train_dates, self.X_val, self.X_val_dates, self.X_test, self.X_test_dates

    def make_targets(self,df,
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        test_start_date,
        test_end_date,
        forward=30,
        take_log=False):
    
        if forward == 1:
            if take_log:
                Y = -1.0*(np.log(df[self.target]) - np.log(df[self.target]).shift(-1*forward))
            else:
                Y = -1.0*(df[self.target] - df[self.target].shift(-1*forward))
        else:
            if take_log:
                Y = -1.0*(np.log(df[self.target]) - np.log(df[self.target]).rolling(forward).mean().shift(-1*forward))
            else:  
                Y = -1.0*(df[self.target] - df[self.target].rolling(forward).mean().shift(-1*forward))
    

        self.Y_train = Y[train_start_date:train_end_date]
        self.Y_train_dates = Y[train_start_date:train_end_date].index
        self.Y_val = Y[val_start_date:val_end_date]
        self.Y_val_dates = Y[val_start_date:val_end_date].index
        self.Y_test = Y[test_start_date:test_end_date]
        self.Y_test_dates = Y[test_start_date:test_end_date].index

        return self.Y_train, self.Y_train_dates, self.Y_val,self.Y_val_dates, self.Y_test, self.Y_test_dates


