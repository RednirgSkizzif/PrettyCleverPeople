#! /home/goldenboard/miniconda3/bin/python
import pymongo
from pprint import pprint
import pdb 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np

db = pymongo.MongoClient('localhost', 27017).gridpoints

rcParams['figure.figsize'] = (20.0, 10.0)
rcParams['font.size'] = 24

print("----------")
print("----------")
print("model_parameters keys: " + str(db.model_parameters.find_one().keys()))
print("----------")
print("backtests keys: " + str(db.backtests.find_one().keys()))
print("----------")
print("----------")

def plot_backtest(epoch,corr,perf,val_loss,train_loss,master_hash,sharp,sortino):
    fig, axs = plt.subplots(3)
    fig.set_size_inches(16, 20, forward=True)
    axs[0].set_title(master_hash)
    axs[0].plot(epoch,corr , label='Correlation',linewidth=4)
    # axs[0].plot(epoch, corr, label='Sharpe',linewidth=4)
    axs[0].set_ylabel('Correlation coefficient', color='b',fontsize=22)
    axs[0].tick_params(labelsize=16)
    axs02 = axs[0].twinx() 
    axs02.tick_params(labelsize=16)
    # axs02.plot(epoch,perf, label='Backtest performance (2YRS)', color='r')
    axs02.plot(epoch,perf, label='Annual performance ', color='r')
    axs02.set_ylabel("Return", color='r',fontsize=22)
    axs02.legend(loc='lower right',prop={'size': 16})

    axs[0].legend(prop={'size': 6})
    # axs[0].set_xlim([self.sim_start,self.sim_end])
    # axs[0].set_ylim([90000,320000])
    axs[1].plot( epoch,val_loss, label='Val loss')
    # axs[1].plot( epoch,train_loss-np.mean(val_loss), label='Train loss')
    axs[1].set_ylabel('Loss',fontsize=22)
    axs[1].tick_params(labelsize=16)
    axs[1].legend(prop={'size': 16})
    axs[1].set_yscale("log")

    axs[2].plot(epoch,sharp , label='Sharpe',linewidth=4)
    axs[2].plot(epoch,sortino, label='Sortino', color='r')
    axs[2].set_ylabel('Ratio',fontsize=22)
    axs[2].tick_params(labelsize=16)

    axs[2].legend(prop={'size': 16})
    plt.savefig('plots/'+master_hash+'.png')
    plt.clf()
    del fig
    del axs
    del axs02

    # axs[1].set_xlim([self.sim_start,self.sim_end])

for gp in db.model_parameters.find():
    print(" ".join([str(gp['look_forward']),str(gp['look_back']),gp['target_symbol'] ]) )
    epoch = []
    correlation = []
    backtest_results = []
    val_loss = []
    train_loss = []
    sharp = []
    sortino = []
    print("Gridpoint ID: " + str(gp['_id']))
    backtests = db.backtests.find( {'gridpoint_id':gp['_id']  } )
    

    if backtests.count() > 5:
        print("Backtests present : " + str(backtests.count()))
    else:
        print("Backtests absent : DELETING")
        db.model_parameters.delete_one({'_id':gp['_id'] })
        continue
    for backtest in backtests:
        epoch.append(backtest['epoch'])
        correlation.append(backtest['target_correlation'])
        backtest_results.append(backtest['performance'])
        val_loss.append(backtest['logs']['val_loss'])
        train_loss.append(backtest['logs']['loss'])
        try:
            sharp.append(backtest['sharpe_ratio'])
        except:
            sharp.append(backtest['sharp_ratio'])
        sortino.append(backtest['sortino_ratio'])
        # try:
        #     sharp.append(backtest['sharpe_ratio'])
        #     sortino.append(backtest['sortino_ratio'])
        #     print("worked!")
        # except:
        #     sharp.append(1)
        #     sortino.append(1)
    val_loss[0] = val_loss[1]
    train_loss[0] = train_loss[1]



    plot_backtest(epoch,
        correlation,
        backtest_results,
        val_loss,
        train_loss,
        gp['master_hash'],
        sharp,sortino)
    # plot_backtest(epoch,correlation,backtest_results,val_loss,train_loss,gp['master_hash'])
    # pdb.set_trace()

