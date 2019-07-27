# -*- coding: utf-8 -*-
#please watch https://docs.poloniex.com/#currency-pair-ids
#Please search by the keyword that is "USDT_BTC".

import poloniex
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv

def main():
    CoinNameList=['USDT_ETH','USDT_BTC','USDT_LTC','USDT_XMR']
    for CoinName in CoinNameList:
        date, data = getDataPoloniex(CoinName)

        with open(CoinName+'.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
            for loop in range(len(date)):
                writer.writerow([date[loop],data[loop]])  # list（1次元配列）の場合

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(date, data)
    ax.set_title("ETH Price")
    ax.set_xlabel("Day")
    ax.set_ylabel("BTC Price[$]")
    plt.grid(fig)
    plt.show(fig)
    '''
def getDataPoloniex(CoinName):
    polo = poloniex.Poloniex()
    polo.timeout = 2
    date_return = []
    data_return = []
    current_time_obj = datetime.datetime(2019,2,20,0,0,0,tzinfo=datetime.timezone.utc)#タイムゾーンをUTCにした。
    current_time = current_time_obj.timestamp()
    for loop in reversed(range(0,300)):
        one_term=14400*10
        start_time = (loop+1)*one_term
        end_time = loop*one_term
        chartUSDT_BTC = polo.returnChartData(CoinName, period=300, start=current_time - start_time, end=current_time - end_time)
        print(chartUSDT_BTC)

        tmpDate = [chartUSDT_BTC[i]['date'] for i in range(len(chartUSDT_BTC))]
        print([datetime.datetime.fromtimestamp(tmpDate[i]) for i in range(len(tmpDate))])
        date_return.extend([datetime.datetime.fromtimestamp(tmpDate[i],tz=datetime.timezone.utc) for i in range(len(tmpDate))])
        data_return.extend([float(chartUSDT_BTC[i]['open']) for i in range(len(chartUSDT_BTC))])
        time.sleep(15)
    return date_return ,data_return

if __name__ == "__main__":
    main()