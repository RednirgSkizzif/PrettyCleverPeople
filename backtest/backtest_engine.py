try:
    from backtest.properties import *
except:
    from properties import *
import time 

K=1000 #kilo
M=K*K # Million
B=M*K # Billion


def getNanoSeconds():
    return int(time.time() * M)

def getTempFile(rootFolder):
    return "{0}/temp{1}.file".format(rootFolder,getNanoSeconds())

outputPath = "{0}/output".format(projectLocalRootPath)
