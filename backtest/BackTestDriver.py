# from collections import OrderedDict
import pytz
import datetime 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from zipline.api import symbol as zipline_symbol
# from zipline.api import order, record, fetch_csv, order_percent, order_target_percent, set_benchmark
# from zipline.algorithm import TradingAlgorithm
# from zipline.utils.factory import create_simulation_parameters
# from zipline.finance import commission
# import pyfolio as pf
# try:
#     from backtest.backtest_engine import getTempFile, outputPath
# except:    
#     from backtest_engine import getTempFile, outputPath

import seaborn as sns
from six import viewkeys



#Import the regressors for the model
from sklearn.preprocessing import QuantileTransformer
from keras.models import save_model
import numpy as np
import pdb
from time import time
import shutil
try:
    import backtest.PyAlgoBackTest as PyAlgoBackTest
except:
    import PyAlgoBackTest

class BackTestDriver(object):
	"""docstring for BackTestDriver"""
	def __init__(self, df, params, data_utility):
		super(BackTestDriver, self).__init__()
		self.df = df
		self.params = params

		try:
			self.signalsPath = "backtest/temp/"+str(time())
			shutil.os.makedirs(self.signalsPath)
		except:
			print("Failed to make directory for backtesting.")
			quit()

		self.DataUtility = data_utility
		self.model = None# Starts of without a model
		self.corr = 0.
		self.performance = None
		# self.prediction_matrix = None#This will ultimatly be the product of the NN


		self.Y_train, self.Y_train_dates, self.Y_val,self.Y_val_dates,self.Y_test,self.Y_test_dates = self.DataUtility.make_targets(df,
																		params['train_start'],
																		params['train_end'],
																		params['val_start'],
																		params['val_end'],
																		params['test_start'],
																		params['test_end'],
																		forward=params['look_forward'])
		self.X_train, self.X_train_dates, self.X_val, self.X_val_dates, self.X_test, self.X_test_dates = self.DataUtility.create_features(df,
																		params['train_start'],
																		params['train_end'],
																		params['val_start'],
																		params['val_end'], 
																		params['test_start'],
																		params['test_end'],
																		back=params['look_back'],
																		forward=params['look_forward'])	

		self.X_train_trans = self.DataUtility.feature_transformer.fit_transform(self.X_train)
		self.Y_train_trans = self.DataUtility.target_transformer.fit_transform(self.Y_train.values.reshape(-1,1))
		self.X_val_trans = self.DataUtility.feature_transformer.transform(self.X_val)
		self.Y_val_trans = self.DataUtility.target_transformer.transform(self.Y_val.values.reshape(-1,1))
		self.X_test_trans = self.DataUtility.feature_transformer.transform(self.X_test)
		self.Y_test_trans = self.DataUtility.target_transformer.transform(self.Y_test.values.reshape(-1,1))


		#Default value overwritten when a model is trained
		#This is used to decide when to buy or sell a security during backtest
		self.prediction_std = 0.0



	def evaluate_model(self,X):
		"""Docstring:
		Method for evaluating the neural network
		X: The transformed feature matrix. It is passed ambiguosly here because
		the class does not know/care if is train/val/test. 
		"""
		KPI = np.full((len(X),1),np.nan)
		for i in range(self.params['look_back'],len(X)):
		    
		    feature = X[i-self.params['look_back']:i]
		    KPI[i] =  self.model.predict( feature[np.newaxis,...] )

		KPI_flat = KPI[:,0]

		untransformed_predictions = KPI_flat.reshape(-1,1) 
		return untransformed_predictions#self.DataUtility.target_transformer.inverse_transform( untransformed_predictions )

	def build_predictions(self,X,Y,Y_dates):
		pmat = pd.DataFrame(self.df.loc[Y_dates],index=Y_dates)
		pmat['targets'] = Y
		pmat['rnn_score'] = self.evaluate_model(X)
		pmat['unmapped_targets'] = self.DataUtility.target_transformer.inverse_transform(pmat['targets'].values.reshape(-1,1))
		pmat['unmapped_score'] = self.DataUtility.target_transformer.inverse_transform(pmat['rnn_score'].values.reshape(-1,1))
		return pmat.dropna(axis=0)

	def make_residual_hists(self,table,epoch,dataset):
		#Take the top right corner of the correlation matrix
		self.corr = np.corrcoef(table['unmapped_score'],table['unmapped_targets'])[0][1]
		plt.clf()
		sns.kdeplot( table['unmapped_targets'], shade=True , alpha=0.2, label='Change (%s days)'%self.params['look_forward'])
		sns.kdeplot( (table['unmapped_targets'] - table['unmapped_score']) , shade=True , alpha=0.5, label='Residual (%s days)'%self.params['look_forward'])
		plt.xlabel('Average change over next %s days '%self.params['look_forward'])
		plt.legend()
		plt.savefig(self.params['master_hash']+'/Residuals_'+dataset+'_{}_{:.4f}.png'.format(epoch,self.corr)  )
		plt.clf()
		sns.scatterplot(x='unmapped_targets',y='unmapped_score',data=table)
		plt.savefig(self.params['master_hash']+'/Scatter_'+dataset+'_{}_{:.4f}.png'.format(epoch,self.corr) )
		print(np.corrcoef(table['unmapped_score'],table['unmapped_targets']))

	def make_performance_plots(self,perf,epoch,dataset):
		plt.clf()
		fig, axs = plt.subplots(3)
		fig.set_size_inches(16, 22, forward=True)
		axs[0].plot([1+x for x in perf['returnAnalyzer'].getCumulativeReturns()],
		 label='Portfolio Value',linewidth=4)
		axs[0].set_ylabel('Portfolio Value', color='b',fontsize=18)
		axs[0].tick_params(labelsize=16)
		axs[0].legend(prop={'size': 6})
		axs[0].set_ylim([0.8,1.8])
		# axs02 = axs[0].twinx() 
		axs[1].tick_params(labelsize=16)
		axs[1].plot( self.df[self.params['target_symbol']] , label=self.params['target_symbol'], color='r')
		axs[1].set_ylabel(self.params['target_symbol'], color='r',fontsize=18)
		axs[1].legend(loc='lower right',prop={'size': 6})
		axs[1].set_xlim([self.sim_start,self.sim_end])
		axs[1].set_ylim([min(self.df[self.params['target_symbol']])*1.2,0.7*max(self.df[self.params['target_symbol']])])

		# axs02.set_ylim([self.backTestPanel[self.params['target_symbol']].close.iloc[0]*0.9,
		# 	self.backTestPanel[self.params['target_symbol']].close.iloc[0]*2.1])

		axs[2].plot( self.prediction_matrix['unmapped_score'], label='RNN Score')
		axs[2].set_ylabel('RNN Score', color='b',fontsize=18)
		axs[2].tick_params(labelsize=16)
		# axs12 = axs[1].twinx() 
		# axs12.tick_params(labelsize=16)
		# axs12.plot( long/short , label='Position Value', color='r')
		# axs12.set_ylabel('Position Value', color='r',fontsize=18)
		# axs12.legend(loc='lower right',prop={'size': 6})
		axs[2].legend(prop={'size': 6})
		axs[2].set_xlim([self.sim_start,self.sim_end])

		plt.savefig(self.params['master_hash']+'/Backtest_'+dataset+'_epoch-{}_return-{:.4f}.png'.format(epoch,perf['annualReturns']))
		plt.cla()
		plt.clf()
		plt.close(fig)
		del fig
		del axs

	def make_backtest_panel(self,predictions):
		"""
		This the the original backtest panel required by the zipline framework.
		It should be depreciated and replaced by the pyalgotrade framework.
		"""
		dataframes = []
		panel_dictionary = {}
		for symbol in self.params['symbols']:
			rnn_score = predictions["unmapped_score"] 
			temp = rnn_score.to_frame()
			temp.columns = ["unmapped_score"]
			temp["target_MA20"] = (predictions[self.params['target_symbol']].rolling(self.params['backtest_momentum']).mean() \
			- predictions[self.params['target_symbol']] )# Renormalize to gaussian using training std
			#df["VolumeScore"] = zScore( testSet[symbol]['volume'] ,10)
			temp["Symbol"]   = symbol
			temp["close"] = predictions[symbol]
			temp["date"]     = temp.index#.map(dateToStringF)
			dataframes.append(temp)
			panel_dictionary[symbol] = temp
		result = pd.concat(dataframes)

		result.to_csv(self.signalsPath, index=False)
		return pd.Panel(panel_dictionary)

	def make_backtest_csv_files(self,predictions):
		"""
		This file will produce a csv file in the temp/ dir for the target variable as well as 
		the prediction matrix.
		Input: predictions <pandas.DataFrame>
		Output: 0 upon successful completion

		A seperate csv file will be created for the rnn_score and the target.
		The High,Low,Settle,etc. are all set to the "Settle" value for now.
		""" 

		#String to log date on temprorary csv files
		self.today_string = str(datetime.datetime.today().date())

		#start with predictions
		rnn_score = predictions["unmapped_score"] 


		#The backtest framework requires these columns to operate
		backtest_values = {"Date Time": rnn_score.index.values,
							"Open":rnn_score.values,
							"High":rnn_score.values,
							"Low":rnn_score.values,
							"Close":rnn_score.values,
							"Settle":rnn_score.values,
							"Volume":1000000000}

		rnn_score = pd.DataFrame(backtest_values)
		rnn_score.set_index("Date Time",inplace=True)

		#After the format is constructed write the predictions to csv
		rnn_score.to_csv(\
			self.signalsPath+"/rnn_score_"+self.today_string+".csv",date_format="%Y-%m-%d %H:%M:%S")


		#The backtest framework needs the target symbol to perform backtest on
		target_df = self.df[self.params['target_symbol']].loc[rnn_score.index]

		#Refill the table values with the target df
		backtest_values = {"Date Time": target_df.index.values,
							"Open":target_df.values,
							"High":target_df.values,
							"Low":target_df.values,
							"Close":target_df.values,
							"Settle":target_df.values,
							"Volume":1000000000}

		target_df = pd.DataFrame(backtest_values)
		target_df.set_index("Date Time",inplace=True)


		#Export the target values to seperate csv file for backtesting
		target_df.to_csv(\
			self.signalsPath+"/"+self.params['target_symbol']+"_"+self.today_string+".csv",date_format="%Y-%m-%d %H:%M:%S")
		return 0



	def val_backtest(self,model,epoch,loss,trade_parameter):
		self.model = model
		self.trade_parameter = trade_parameter

		self.prediction_matrix = self.build_predictions(self.X_val_trans, self.Y_val_trans,self.Y_val_dates)
		self.make_residual_hists(self.prediction_matrix,epoch,'Val')
		# self.backTestPanel = self.make_backtest_panel(self.prediction_matrix)
		self.make_backtest_csv_files(self.prediction_matrix)
		print(self.prediction_matrix[['unmapped_score','unmapped_targets']].tail(55))

		# self.SignalThreshold = self.backTestPanel[ self.params['target_symbol'] ]['unmapped_score'].std()*self.trade_parameter# sigma threshold for signal to open position
		# self.AveThreshold = self.params['backtest_momentum_trade']
		# self.close_parameter = self.SignalThreshold

		save_model(self.model, self.params['master_hash']+'/model_epoch-{}_corr-{:.4f}_loss-{:.6f}.hdf5'.format(epoch,self.corr, loss))
		self.performance = self.do_val_simulation()
		# print("Backtest Performance: " + str(self.performance.returns.sum()))
		self.make_performance_plots(self.performance, epoch,'Val')
		# self.save_correlation_toDB(self.performance)

		return self.performance, self.corr, self.prediction_matrix, self.prediction_std


	def do_val_simulation(self):
		"""
		This function is in charge of calling the PyAlgoTrade backetst framework and returning the results.
		"""
		self.sim_start = datetime.datetime(int(self.params['val_start'].split('-')[0]),
		 int(self.params['val_start'].split('-')[1]), int(self.params['val_start'].split('-')[2]),
		  0, 0, 0, 0, None)
		self.sim_end = datetime.datetime(int(self.params['val_end'].split('-')[0]),
				 int(self.params['val_end'].split('-')[1]), int(self.params['val_end'].split('-')[2]),
		  0, 0, 0, 0, None)

		#In order to make a good backtest we use the std of the training prediction
		train_set_scores = self.evaluate_model( self.X_train_trans  )
		#The training set suffers from nans in the beginning of the dataset.
		#It must be reshaped because the transformer requires this
		train_set_scores = train_set_scores[~np.isnan(train_set_scores)].reshape(-1,1)
		self.prediction_std = self.DataUtility.target_transformer.inverse_transform( train_set_scores).std(ddof=1)

		#For now I will leave the std as zero, but should try other methods
		output_dic = PyAlgoBackTest.algo_test(self.params['target_symbol'],
			self.signalsPath+"/"+self.params['target_symbol']+"_"+self.today_string+".csv",
			"rnn_score",
			self.signalsPath+"/rnn_score_"+self.today_string+".csv",
			model_prediction_std=self.prediction_std)


		return output_dic
