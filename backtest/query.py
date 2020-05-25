import pandas as pd

from utils import *

#print(B)
#print(projectLocalRootPath)

#rootToMetadataFolder = "{0}/nasdaq_metadata/".format(dataPathStocksMetadata)

# format of Dataframe with compny's metadata
# 'exchange', 'Symbol', 'Name', 'LastSale', 'MarketCap', 'ADR TSO', 'IPOyear', 'Sector', 'Industry', 'Summary Quote'
#meta = pd.read_csv(rootToMetadataFolder+"usListed.csv",sep="|")
#meta.head(50)

#energy = meta[(meta.Sector=="Energy") & (meta.Industry=="Oil & Gas Production")].sort_values('MarketCap', ascending=False)
#energy["MC"]=energy.MarketCap/B
#energy.head(50)

# construction of Symbol-Exchange dictionary - aware some double-exchange mentioned/listed companies - 11 cases
#symbolsByFreq = meta.Symbol.value_counts() # Symb-freq : Series
#multipleOccurenceSymbols = list(symbolsByFreq[symbolsByFreq>1].index.values) #
#uniqueMetaPart = meta[~meta.Symbol.isin(multipleOccurenceSymbols)]
#multipleMetaPart = meta[meta.Symbol.isin(multipleOccurenceSymbols)].sort_values(by='exchange').groupby('Symbol').first()
#uniqueMeta=pd.concat([uniqueMetaPart,multipleMetaPart])
#debug(uniqueMeta)
#symbolToExchangeDict = pd.Series(index=uniqueMeta.Symbol.values,data=uniqueMeta.exchange.values).to_dict()


def getDailyPricesFilePath(symbol):
    exchange=symbolToExchangeDict[symbol]
    path="{dataPathStocks}/prices/EOD/tiingo/{0}/{1}/{2}.csv".format(exchange,symbol[0],symbol,dataPathStocks=dataPathStocks)
    return path

def getDailyFundamentalsFilePath(symbol):
    exchange=symbolToExchangeDict[symbol]
    path="{dataPathStocks}/fundamentals/tiingo/{0}/{1}.json".format(symbol[0],symbol,dataPathStocks=dataPathStocks)
    return path
#print(getDailyPricesFilePath('OXY'))

# return prices as Panel[SymbolString] - each value contains Dataframe
#  index - "date" column
#  columns = list from "columns" parameter
# tiingo columns format: 'date,close,high,low,open,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume,divCash,splitFactor'
def getPrices(pathTemplate,symbols,startDate,endDate,columns):
    data = {}
    for symb in symbols:
        path=getDailyPricesFilePath(symb)
        #path = pathTemplate % symb
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df.date)
        df=df[(df.date>=pd.to_datetime(startDate)) & (df.date<=pd.to_datetime(endDate))]
        df = df.set_index('date')
        data[symb] = df[columns]
        #debug(data[symb],broad=True,lines=3)
    pn = pd.Panel(data)#.to_frame()
    #debug(pn)
    #print(pn.describe)
    return pn

def getFundamentals(symbols,startDate,endDate,columns=None):
    data = {}
    for symb in symbols:
        path=getDailyFundamentalsFilePath(symb)
        df = pd.read_json(path)
        df['year'] = pd.to_datetime(df.start_date)
        df=df[ ( ( df.year >= pd.to_datetime(startDate) ) & (df.year <= pd.to_datetime(endDate) ) ) ]
        df = df.set_index('year')
        data[symb] = df
        #debug(data[symb],broad=True,lines=3)
    pn = pd.Panel(data)
    return pn
