import quandl
import pandas as pd
from keychain import quandl_key
import datetime
import numpy as np
import datetime 
import time
import pdb
from backtest.utils import padding


def check_data(df):

      #The dataframe should adhere to this format
       #Date Time,Open,High,Low,Close,Volume
       #2013-01-01 13:59:00,13.51001,13.56,13.51,13.56,273.88014126
      for i in range(0,len(df)):
            xm1 = df.iloc[i-1]
            x = df.iloc[i]
            try:
                  xp1 = df.iloc[i+1]
            except:
                  xp1 = df.iloc[i]

            #Check the 'Open' First
            if np.isnan( x['Open']):
                  print("NAN in Open: " + str(i))
                  try:
                        x['Open'] = x1['Close']
                  except:
                        try:
                              x['Open'] = x['Settle']
                              x['High'] = x['Settle']
                              x['Low'] = x['Settle']
                        except:
                              print("Open column has faulty values")
                              print("Row: " + str(i))
                              print(x)
                              raise Exception
            #Check the 'Close'
            if  np.isnan( x['Close']):
                  print("NAN in Close: " + str(i))
                  try:
                        x['Close'] = xp1['Open']
                  except:
                        try:
                              x['Close'] = x['Settle']
                              x['High'] = x['Settle']
                              x['Low'] = x['Settle']
                        except:
                              print("Close column has faulty values")
                              print("Row: "+str(i))
                              print(x)
                              raise Exception

            #Check the 'Close'
            if np.isnan( x['Low']):
                  print("NAN in Low: " + str(i))
                  try:
                        x['Low'] = min(x['Open'],x['Close'])
                  except:
                        try:
                              x['Low'] = x['Settle']
                        except:
                              print("Low column has faulty values")
                              print("Row: "+str(i))
                              print(x)
                              raise Exception

            #Check the 'Close'
            if np.isnan( x['High']):
                  print("NAN in High: " + str(i))
                  try:
                        x['High'] = max(x['Open'],x['Close'])
                  except:
                        try:
                              x['High'] = x['Settle']
                        except:
                              print("High column has faulty values")
                              print("Row: "+i)
                              print(x)
                              raise Exception

      #Check low > high
            if x['Low'] > x['High']:
                  print("Low>High: " + str(i))
                  try:
                        x['Low'] = x['Settle']
                        x['High'] = x['Settle']
                        x['Open'] = x['Settle']
                        x['Close'] = x['Settle']
                  except:
                        try:
                              x['Low'] = xm1['Settle']
                              x['High'] = xm1['Settle']
                              x['Open'] = xm1['Settle']
                              x['Close'] = xm1['Settle']
                        except:
                              print("Error in Low > High logic")
                              raise Exception

      #Check low > high
            if x['Low'] > x['Open'] or x['Low'] >x['Close']:
                  print("Low > Open|Close: " + str(i))
                  try:
                        x['Low'] = min( [ x['Open'], x['Close'] ] )
                  except:
                        try:
                              x['Low'] = x1['Settle']
                              x['High'] = x1['Settle']
                              x['Open'] = x1['Settle']
                              x['Close'] = x1['Settle']
                        except:
                              try:
                                    x['Low'] = xm1['Settle']
                                    x['High'] = xm1['Settle']
                                    x['Open'] = xm1['Settle']
                                    x['Close'] = xm1['Settle']
                              except:
                                    print("Error in Low > Open|Close logic")
                                    raise Exception

      #Check low > high
            if x['High'] < x['Open'] or x['High'] < x['Close']:
                  print("High< Open|Close: " + str(i))
                  try:
                        x['High'] = max( [ x['Open'], x['Close'] ] )
                  except:
                        try:
                              x['Low'] = x1['Settle']
                              x['High'] = x1['Settle']
                              x['Open'] = x1['Settle']
                              x['Close'] = x1['Settle']
                        except:
                              try:
                                    x['Low'] = xm1['Settle']
                                    x['High'] = xm1['Settle']
                                    x['Open'] = xm1['Settle']
                                    x['Close'] = xm1['Settle']
                              except:
                                    print("Error in High < Open|Close logic")
                                    raise Exception

            df.iloc[i] = x
            #}End Loop.... Fucking python

      return df


while(True):

      #Create a master index to fill in all dates uniformly 
      index_master = pd.date_range(start="1999-01-01",
                       end=datetime.date.today(),
                       freq='D').strftime("%Y-%m-%d %H:%M:%S")
      index_master = index_master.to_frame()

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


      training_df = pd.DataFrame()

      for key,item in datasets.items():

            print("Working on : "+ key)
            try:
                  x = quandl.get(item, authtoken=quandl_key)
                  print("Successfully downloaded : " + key)
            except LimitExceededError as error:
                  print("Failure to download : " + key)
                  print(error)


            if "Close" not in x.columns and "Open" in x.columns and "Last" not in x.columns:
                  x['Close'] = x['Settle']

            #In the case of indexes there is only one value per day.
            if "Open" not in x.columns and "Value"  in x.columns :
                  print("Replacing all fields with VALUE column")
                  x['Open'] = x['Value']
                  x['Close'] = x['Value']
                  x['High'] = x['Value']
                  x['Low'] = x['Value']
                  x['Settle'] = x['Value']
                  x['Volume'] = 0.0


            #Some datasets use the term last in place of close
            if "Last" in x.columns and "Close" not in x.columns:
                  print("Replacing LAST with CLOSE")
                  x.rename(columns={'Last':'Close'},inplace=True)

            x.index.rename('Date Time',inplace=True)

            #Redundant data
            if "Change" in x.columns:
                  print("Deleting CHANGE column")
                  del x['Change']


            #Index master
            index_master = pd.date_range(start="1999-01-01",
                           end=datetime.date.today(),
                           freq='D')
            index_master = index_master.to_frame()
            #Remove weekends. Still going to have issues with holidays.
            index_master = index_master.loc[ index_master.index.dayofweek < 5 ]

            #Set the raw data index to proper format as used by later steps
            x.index = pd.to_datetime(x.index)


            x = check_data(x)


            x = pd.concat([index_master,x],axis=1)

            del x[0]
            # pdb.set_trace()
            x = padding(x)
            x = check_data(x)


            # x = x.interpolate(method="pad",axis=1)
            x.index.rename('Date Time',inplace=True)
            x = x.loc["1999-01-01":datetime.date.today()]

            x.to_csv( "backtest/data/"+key+"_"+str(datetime.date.today())+".csv",
                  date_format="%Y-%m-%d %H:%M:%S")
            print("Finished : " + key)

            training_df[key+"_settle"] = x['Settle']
            training_df.index = x.index



      training_df.to_csv('data/raw_data_for_training_'+str(datetime.date.today())+".csv",
            date_format="%Y-%m-%d %H:%M:%S")

            #Sleep for a day
      print("Begin slepeping for 1 day....")
      time.sleep(60*60*24)


# Manager = DataManager(quandl_key,key_order=key_order)
# Manager.pull_datasets(datasets)
# Manager.make_dataframe()
# Manager.save_as_csv("raw_data_1monthFuture"+str(date.today())+".csv")

# pdb.set_trace()