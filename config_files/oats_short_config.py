
class ApiConfig(object):
	"""ApiConfig: Configuration object that will handle the data pipeline 
	feeding the API."""
	def __init__(self):
		super(ApiConfig, self).__init__()

		self.model_name = "oats_back30_forward15"
		self.model_hdf = "/home/goldenboard/new_backtest/forecast-models/oats_models/blah-model_1030_0.3472.hdf5"

		self.train_year = 2011
		self.train_month = 1
		self.train_day = 1

		self.val_start_year = 2017
		self.val_start_month = 1
		self.val_start_day = 1

		self.val_end_year = 2019
		self.val_end_month = 1
		self.val_end_day = 1

		self.look_forward = 15 
		self.momentum_parameter = 2
		self.look_back = 30

		self.feature_transform_shape = 'uniform'
		self.target_transform_shape = 'uniform'

		self.target_name = 'oats_settle'
		
		self.datasets = { "wheat" : "CHRIS/CME_W1",
					"rici" : "RICI/RICI",
					"oats" : "CHRIS/CME_O1",
					"canola" : "CHRIS/ICE_RS1",
					"corn" : "CHRIS/CME_C1",
					"copper" : "CHRIS/CME_HG1",
					"oil" : "CHRIS/ICE_G1" } 

		self.key_order = ['corn','copper','canola','rici','oil','wheat','oats']

		self.symbols = ['rici_settle'
			         ,'canola_settle'
			         ,'oats_settle'
			         ,'wheat_settle'
			         ,'corn_settle'
			         ,'copper_settle'
			         ,'oil_settle']



		#This is the distance in the past to take the residulas of input features.
		#-1 is used for complete training set.
		self.trend_parameter = -1
