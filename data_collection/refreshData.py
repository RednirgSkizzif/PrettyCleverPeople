from keychain import quandl_key, tiingo_key
from DataManager import DataManager
from datetime import date
import pandas as pd
import pdb

datasets = { "wheat" : "CHRIS/CME_W3",
			 # "cotton" : "CHRIS/ICE_CT1",
			# "CHRIS/CME_FC2",
			# "CHRIS/ASX_WM5",
			# "FRED/WPU0131",
			# "FRED/IY11192",
			# "FRED/IQ00000",
			# "RICI/RICIA",
			"rici" : "RICI/RICI",
			# "rica" : "RICI/RICIA",
			# "CFTC/001626_FO_ALL",
			# "CFTC/001602_FO_CHG",
			# "soybeans" : "CHRIS/ICE_ISM2",
			# "positions" : "CFTC/001612_FO_ALL_CR"
			# "investerSent" : "AAII/AAII_SENTIMENT",
			# "cattle" : "CHRIS/CME_LC2",
			"oats" : "CHRIS/CME_O2",
			"canola" : "CHRIS/ICE_RS5",
			"corn" : "CHRIS/CME_C5",
			"copper" : "CHRIS/CME_HG3",
			"oil" : "CHRIS/ICE_G3" } 

datasets = { "wheat" : "CHRIS/CME_W1",
			 # "cotton" : "CHRIS/ICE_CT1",
			# "CHRIS/CME_FC2",
			# "CHRIS/ASX_WM5",
			# "FRED/WPU0131",
			# "FRED/IY11192",
			# "FRED/IQ00000",
			# "dollar" : "CHRIS/ICE_DX1",
			"rici" : "RICI/RICI",
			# "bloomberg" : "CHRIS/CME_AW1",
			# "CFTC/001626_FO_ALL",
			# "CFTC/001602_FO_CHG",
			# "soybeans" : "CHRIS/ICE_ISM1",
			# "positions" : "CFTC/001612_FO_ALL_CR"
			# "investerSent" : "AAII/AAII_SENTIMENT",
			# "cattle" : "CHRIS/CME_LC2",
			"oats" : "CHRIS/CME_O1",
			"canola" : "CHRIS/ICE_RS1",
			"corn" : "CHRIS/CME_C1",
			"copper" : "CHRIS/CME_HG1",
			"oil" : "CHRIS/ICE_G1" } 


key_order = ['corn','copper','canola','rici','oil','wheat','oats']#,'dollar','bloomberg']

Manager = DataManager(quandl_key,key_order=key_order)
Manager.pull_datasets(datasets)
Manager.make_dataframe()
Manager.save_as_csv("raw_data_1monthFuture"+str(date.today())+".csv")

# pdb.set_trace()

#Benchmark
def tget(ticker,start="2008-01-01",end="2020-01-01",output="json"):
	try:
		DF = pd.read_json("https://api.tiingo.com/tiingo/daily/"+ticker+"/prices?startDate="+start+"&endDate="+end+"&token="+tiingo_key)
		DF.index = DF.date
		DF.to_csv( "data/benchmark_data_"+str(date.today())+".csv" )#, orient='records', lines=True)
		print("SUCCESS")
		return 0
	except Exception as e: 
		print("Error in request")
		print(e)
		return 1

tget("spy",end=str(date.today()))



