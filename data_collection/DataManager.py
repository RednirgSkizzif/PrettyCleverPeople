import pandas as pd
import numpy as np
import time
import sys
import pdb
from datetime import datetime, date

def quandl_request(name,token):
    return quandl.get(name, authtoken=token)

class DataManager(object):
	"""docstring for DataManager:
	Class to control the data collection and management for the Wheat forecast model."""
	def __init__(self, token, path="data",key_order=None,primary_variable=None):
		super(DataManager, self).__init__()
		self.token = token
		self.path = path
		self.dataframe = None
		self.frames = {}

		if primary_variable != None:
			self.primary_variable = primary_variable
		else:
			self.primary_variable = "Settle"
		if key_order != None:
			self.col_order = [ label+"_settle" for label in key_order ]
		else:
			self.col_order = None



	def pull_datasets(self,datasets):

		for key, sets in datasets.items():
		    Try = True
		    count = 0
		    print("Trying " + sets)
		    while Try:
		        count += 1 
		        # pdb.set_trace()
		        try:
		            self.frames[key] =  quandl_request(sets, self.token) 
		            # print( self.frames[key] )
		            # print( self.frames[key].columns )
		            # pdb.set_trace()
		            if self.primary_variable in self.frames[key].columns:
		            	print(self.primary_variable + " variable present in " + key)
		            	self.frames[key] = pd.DataFrame( self.frames[key][ self.primary_variable ],index=self.frames[key].index )
		            	self.frames[key].rename(columns = { self.primary_variable :key+"_settle"}, inplace = True)
		            	# self.frames[key].index
		            	# self.frames[key].columns = [key+"_settle"]
		            elif "Value" in self.frames[key].columns:
		            	print(" Value variable present in " + key)
		            	self.frames[key] = pd.DataFrame( self.frames[key]["Value"],index=self.frames[key].index )
		            	self.frames[key].rename(columns = {"Value":key+"_settle"}, inplace = True)

		            elif "Bullish" in self.frames[key].columns:
		            	print(" Bullish variable present in " + key)
		            	self.frames[key] = pd.DataFrame( self.frames[key]["Bullish"] - self.frames[key]["Bearish"], index=self.frames[key].index )
		            	# self.frames[key].rename(columns = {"Bullish":key+"_settle"}, inplace = True)
		            	self.frames[key].columns = ["Sentiment_settle"]

		            else:
		            	print("ERROR:: Missing variable")
		            Try=False
		            print("Success")
		            time.sleep(0.5)
		        except:
		            print( "Failure try again for the %s time " % count )
		            time.sleep(1)
		            if count > 8: break
		print("Size of : " + str( sys.getsizeof(self.frames) ) )
		return self.frames

	def make_dataframe(self):
		self.dataframe = pd.concat( [ x for x in  self.frames.values() ] ,axis=1)
		# self.dataframe = self.dataframe[np.isfinite(self.dataframe)]
		self.dataframe = self.dataframe.interpolate(method="time")
		if self.col_order  != None:
			self.dataframe = self.dataframe[ self.col_order ]
		return self.dataframe

	def save_as_csv(self,name):
		self.dataframe.to_csv(self.path+"/"+name)
		return 1




