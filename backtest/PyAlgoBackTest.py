from pyalgotrade import strategy
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, sortino
from pyalgotrade.utils import stats
from pyalgotrade.barfeed import csvfeed
import pyalgotrade
import numpy as np
import pdb

# class MyStrategy(strategy.BacktestingStrategy):
#     def __init__(self, feed, instrument):
#         super(MyStrategy, self).__init__(feed)
#         self.__instrument = instrument
#         self.__rnn_std = 0
    
#     def onBars(self, bars):
#         bar = bars[self.__instrument]
        
#         owned_shares = self.getBroker().getShares(self.__instrument)
#         cash = self.getBroker().getCash()

#         equity = owned_shares*bar.getClose()
#         current_price = bars[self.__instrument].getPrice()

        
#         # print(bar.getDateTime())
#         # print("cash = " + str(cash))
#         # print("shares = " + str(owned_shares))
#         # print("equity = " + str(equity))
#         # print("cash+equity = " + str(equity+cash))
#         # print("---------------------------")

#         score = bars["rnn_score"].getClose() 

          
#         if score > self.__rnn_std and owned_shares > 0 :
#             pass
        
#         elif score > self.__rnn_std and owned_shares <= 0:
            
#             #We want to turn 95% of our cash to shares.
#             quantity = int( round( ( ((cash) * 0.95)/ current_price) - 0.5 ) )
#             print("BUY "+str(quantity) +" shares "+self.__instrument +" @ "+\
#              str(bar.getClose())+" for total of : "+str(current_price*quantity))
#             print(bar.getDateTime())
#             print("cash = " + str(cash))
#             print("shares = " + str(owned_shares))
#             print("equity = " + str(equity))
#             print("cash+equity = " + str(equity+cash))
#             print("---------------------------")

#             self.enterLong(self.__instrument, quantity)

#         elif score < self.__rnn_std and owned_shares < 0 :
#             pass
        
#         elif score < self.__rnn_std and owned_shares >= 0:            
#             #We want to sell shares equal to the cash reserves of our account. 
#             quantity = int( round( ( ( (cash+equity) * 0.95)/ current_price) - 0.5 ) ) + owned_shares
            
#             print("SELL "+str(quantity) +" shares "+self.__instrument +" @ "+\
#              str(bar.getClose())+" for total of : "+str(current_price*quantity))
#             print(bar.getDateTime())
#             print("cash = " + str(cash))
#             print("shares = " + str(owned_shares))
#             print("equity = " + str(equity))
#             print("cash+equity = " + str(equity+cash))
#             print("---------------------------")
#             self.enterShort(self.__instrument, quantity)


class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument,std):
        super(MyStrategy, self).__init__(feed)
        self.__instrument = instrument
        self.__rnn_std = std
    
    def onBars(self, bars):
        bar = bars[self.__instrument]
        
        owned_shares = self.getBroker().getShares(self.__instrument)
        cash = self.getBroker().getCash()

        equity = owned_shares*bar.getClose()
        current_price = bars[self.__instrument].getPrice()

        
        # print(bar.getDateTime())
        # print("cash = " + str(cash))
        # print("shares = " + str(owned_shares))
        # print("equity = " + str(equity))
        # print("cash+equity = " + str(equity+cash))
        # print("---------------------------")

        score = bars["rnn_score"].getClose() 

        if score < self.__rnn_std and score > -1*self.__rnn_std and owned_shares != 0:
            print("CLOSING positions: " + str(owned_shares))
            print(bar.getDateTime())
            print("cash = " + str(cash))
            print("shares = " + str(owned_shares))
            print("equity = " + str(equity))
            print("cash+equity = " + str(equity+cash))
            print("---------------------------")

            self.marketOrder(self.__instrument,-1*owned_shares)

        elif score > self.__rnn_std and owned_shares > 0 :
            pass
        
        elif score > self.__rnn_std and owned_shares <= 0:
            
            #We want to turn 95% of our cash to shares.
            quantity = int( round( ( ((cash) * 0.95)/ current_price) - 0.5 ) )
            print("BUY "+str(quantity) +" shares "+self.__instrument +" @ "+\
             str(bar.getClose())+" for total of : "+str(current_price*quantity))
            print(bar.getDateTime())
            print("cash = " + str(cash))
            print("shares = " + str(owned_shares))
            print("equity = " + str(equity))
            print("cash+equity = " + str(equity+cash))
            print("---------------------------")

            self.enterLong(self.__instrument, quantity)

        elif score < -1*self.__rnn_std and owned_shares < 0 :
            pass
        
        elif score < -1*self.__rnn_std and owned_shares >= 0:            
            #We want to sell shares equal to the cash reserves of our account. 
            quantity = int( round( ( ( (cash+equity) * 0.95)/ current_price) - 0.5 ) ) + owned_shares
            
            print("SELL "+str(quantity) +" shares "+self.__instrument +" @ "+\
             str(bar.getClose())+" for total of : "+str(current_price*quantity))
            print(bar.getDateTime())
            print("cash = " + str(cash))
            print("shares = " + str(owned_shares))
            print("equity = " + str(equity))
            print("cash+equity = " + str(equity+cash))
            print("---------------------------")
            self.enterShort(self.__instrument, quantity)


 
def algo_test(target_name,target_csv,model_name,model_csv,model_prediction_std=0.0):
    # Load the bar feed from the CSV file
    feed = csvfeed.GenericBarFeed(pyalgotrade.bar.Frequency.DAY)
    feed.addBarsFromCSV(target_name, target_csv)
    feed.addBarsFromCSV(model_name, model_csv)

    # Evaluate the strategy with the feed's bars.
    myStrategy = MyStrategy(feed, target_name,model_prediction_std)
    # myStrategy = MyBasicStrategy(feed, target_name)


    # Attach analyzers.
    retAnalyzer = returns.Returns()
    myStrategy.attachAnalyzer(retAnalyzer)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    myStrategy.attachAnalyzer(sharpeRatioAnalyzer)
    drawDownAnalyzer = drawdown.DrawDown()
    myStrategy.attachAnalyzer(drawDownAnalyzer)
    sortinoRatioAnalyzer = sortino.SortinoRatio()
    myStrategy.attachAnalyzer(sortinoRatioAnalyzer)

    # Run the strategy
    myStrategy.run()

    # Print the results.
    print("Final portfolio value: $%.2f" % myStrategy.getResult())
    print("Total return: %.2f %%" % (retAnalyzer.getCumulativeReturns()[-1] * 100))
    print("Average daily return: %.2f %%" % (stats.mean(retAnalyzer.getReturns()) * 100))
    print("Std. dev. daily return: %.4f" % (stats.stddev(retAnalyzer.getReturns())))
    print("Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0)))
    print("Sortino ratio: %.2f" % (sortinoRatioAnalyzer.getSortinoRatio(0)))

    returnsPerDay = stats.mean(retAnalyzer.getReturns())

    # print("shares list : ")
    # print(myStrategy.getShares())
    output = {'sharpe':sharpeRatioAnalyzer.getSharpeRatio(0),
              'sortino': sortinoRatioAnalyzer.getSortinoRatio(0),
              'returnAnalyzer': retAnalyzer,
              'annualReturns': returnsPerDay*252 }

    return output



