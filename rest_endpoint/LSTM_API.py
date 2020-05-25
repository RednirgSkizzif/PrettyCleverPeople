import pandas as pd
from keras.models import load_model
import keras.backend as K
from sklearn.preprocessing import QuantileTransformer
from keychain import quandl_key
from DataManager import DataManager
from datetime import datetime, date
import numpy as np
import pytz
import pdb
import backtest.DataUtils as dataU
import importlib
# from api_config import ApiConfig

def loss_wrapper(beta):
    _beta = beta
    def customLoss(y_true,y_pred):
        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)

        return K.mean(K.square(y_pred - y_true), axis=-1) + K.maximum(0.0,_beta*K.mean(-(y_pred-0.5) * (y_true-0.5), axis=-1))

    return customLoss

def agnostic_metric(y_true,y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    return K.mean(K.square(0.0 - y_true), axis=-1)

class LSTM_API(object):
	"""docstring for WheatModel:
	This class will represent a data-driven model to forcast the value of US wheat price. 
	It should be used by the flask webserver."""
	def __init__(self, config_file, data):
		super(LSTM_API, self).__init__()
		model_config = importlib.import_module(config_file.replace('/','.').replace('.py',''))

		#Master configuration file. All parameters should exist here.
		self.config = model_config.ApiConfig()

		#This will be the pre-trained model
		print(self.config.model_hdf)
		self.model = load_model( self.config.model_hdf,
		 custom_objects={'customLoss': loss_wrapper(0.0),'agnostic_metric':agnostic_metric})

		self.train_start_date = datetime(self.config.train_year,
		 self.config.train_month,
		 self.config.train_day, 0, 0, 0, 0, None)

		self.train_end_date = datetime(self.config.val_start_year,
		 self.config.val_start_month,
		 self.config.val_start_day, 0, 0, 0, 0, None)

		self.test_start_date = datetime(self.config.val_end_year,
		 self.config.val_end_month,
		 self.config.val_end_day, 0, 0, 0, 0, None)

		self.raw_data = self.get_data_from_csv(data)

		#Create data utility to pre-process input data
		self.data_utility = dataU.data_utils(self.config.target_name,
			self.config.feature_transform_shape,
			self.config.target_transform_shape)

        # self.target_transformer = MinMaxScaler()
		self.X_train, self.X_train_dates, self.X_val, self.X_val_dates,self.X_test, self.X_test_dates = \
		self.data_utility.create_features(self.raw_data,
			self.train_start_date,
			self.train_end_date,
			self.test_start_date,
			self.raw_data.index[-1],
			self.test_start_date,
			self.raw_data.index[-1],
			back=self.config.look_back,
			forward=self.config.look_forward,
			trend_parameter=self.config.trend_parameter)

		self.Y_train, self.Y_train_dates, self.Y_val,self.Y_val_dates, self.Y_test, self.Y_test_dates = \
		self.data_utility.make_targets(self.raw_data,
			self.train_start_date,
			self.train_end_date,
			self.test_start_date,
			self.raw_data.index[-1],
			self.test_start_date,
			self.raw_data.index[-1],
			forward=self.config.look_forward)


		self.data_utility.feature_transformer.fit( self.X_train )
		self.data_utility.target_transformer.fit( self.Y_train.values.reshape(-1,1) )

		self.X_test_transformed = self.data_utility.feature_transformer.transform(self.X_test)

		# pdb.set_trace()
		self.predictions = self.make_predictions()

		#This should be the final product that the server needs to know about
		self.lookup_table = self.fill_lookup_table()


	def get_name(self):
		return self.config.model_name


	def get_data_from_csv(self,path):
		raw_data = pd.read_csv(path,index_col="Date")
		raw_data.index = pd.to_datetime(raw_data.index)
		# raw_data.index =  raw_data.Date.apply(datetime.strptime,args=('%Y-%m-%d',))
		# del raw_data["Date"]
		return raw_data[self.train_start_date:]



	def make_predictions(self):
		KPI = np.full((len(self.X_test_transformed),1),np.nan)
		for x in range(self.config.look_back,len(self.X_test_transformed)):
		    
		    feature = self.X_test_transformed[x-self.config.look_back:x]
		    KPI[x] =  self.model.predict( feature[np.newaxis,...] )

		KPI_flat = KPI[:,0]

		self.untransformed_predictions = KPI_flat.reshape(-1,1) 
		return self.data_utility.target_transformer.inverse_transform( KPI_flat.reshape(-1,1) )


	def update_data(self):
		self.Manager = DataManager(quandl_key,
			key_order=self.config.key_order)
		return 0


	def fill_lookup_table(self):

		self.raw_data["Target Momentum"] = \
		self.raw_data[self.config.target_name].rolling(self.config.momentum_parameter).mean()\
		- self.raw_data[self.config.target_name]
		#self.raw_data['Target Change'] = self.raw_data[self.config.target_name].shift(1)\
		# - self.raw_data[self.config.target_name]

		temp = pd.DataFrame({'AI Indicator':self.predictions[:,0]},index=self.X_test_dates)
		#temp['Date'] = self.X_test_dates.map(str)
		temp['Future Change'] = -(self.raw_data[self.config.target_name] \
		-self.raw_data[self.config.target_name].shift(-1*self.config.look_forward).rolling(self.config.look_forward).mean())[temp.index]
		temp['Residual'] = (temp['Future Change'] - temp['AI Indicator'])[temp.index]
		temp['SMA(%s)' % self.config.momentum_parameter] = self.raw_data["Target Momentum"][temp.index]
		temp['Price'] = self.raw_data[self.config.target_name][temp.index]


		return temp




