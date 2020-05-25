from backtest.properties import *

import pandas as pd
import time
import os
import numpy as np

dataPathStocks = "{0}/data/usaStocks".format(projectLocalRootPath)
dataPathStocksMetadata = "{0}/metadata".format(dataPathStocks)
outputPath = "{0}/output".format(projectLocalRootPath)

K=1000 #kilo
M=K*K # Million
B=M*K # Billion


def padding(df):
      
      for i in range(1,len(df)):
            for col in df.columns:
                  if np.isnan( df.iloc[i][col]):
                        # print("Replacing " + str(df.iloc[i][col]) +\
                        #  " with " + str(df.iloc[i-1][col]))
                        df.iloc[i][col] = df.iloc[i-1][col]

      return df

def createRecursiveFoldersIfAbsent(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def getNanoSeconds():
    return int(time.time() * M)

def getTempFile(rootFolder):
    return "{0}/temp{1}.file".format(rootFolder,getNanoSeconds())


# receive name of variable obj : "obj"
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

