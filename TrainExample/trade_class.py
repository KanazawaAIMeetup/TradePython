#coding: utf-8
'''
MITライセンス　このプログラムについては、改変・再配布可能です
著作者： Tomohiro Ueno (kanazawaaimeetup@gmail.com)
'''

import numpy as np
import poloniex
import datetime
import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt


class TradeClass(object):
    def __init__(self):
        self.trade_history = []
        self.price_history = []
        self.transaction_fee = 0.0001 #取引手数料。この場合0.01%としたが、自由に変えて良い。

    def ReadPoloniexCSV(self):
        import csv
        history_data=[]
        with open('../../../DATA/USDT_BTC_LATEST.csv', 'r') as f:
            reader=csv.reader(f,delimiter=',')
            next(reader)
            for row in reader:
                history_data.append(float(row[1]))
                #print(float(row[1]))
            return history_data

    def ReadBitflyerJson(self):
        import csv
        history_data=[]
        import csv
        with open(os.environ['HOME']+'/bitcoin/bitflyerJPY_convert.csv', 'r') as f:
            reader=csv.reader(f,delimiter=',')
            next(reader)  # ヘッダーを読み飛ばしたい時
            for row in reader:
                history_data.append(float(row[1]))
            return history_data

    def GetDataPoloniex(self):
        polo = poloniex.Poloniex()
        polo.timeout = 10
        chartUSDT_BTC = polo.returnChartData('USDT_ETH', period=300, start=time.time() - 1440*60 * 500, end=time.time())#1440(min)*60(sec)=DAY
        tmpDate = [chartUSDT_BTC[i]['date'] for i in range(len(chartUSDT_BTC))]
        date = [datetime.datetime.fromtimestamp(tmpDate[i]) for i in range(len(tmpDate))]
        data = [float(chartUSDT_BTC[i]['open']) for i in range(len(chartUSDT_BTC))]
        return date ,data

    def PercentageLabel(self,Xtrain,yTrain):
        X=[]
        Y=[]
        for i in range(0,len(yTrain)):
            original=Xtrain[i][-1]
            X.append([float(val/original) for val in Xtrain[i]])
            Y.append(float(float(yTrain[i]/Xtrain[i][-1])-1)*100*100)#%*100
        return X,Y

    def TestPercentageLabel(self,Xtrain):
        X=[]
        for i in range(0,len(Xtrain)):
            original = Xtrain[-1]
            X.append([float(val/original) for val in Xtrain])
        return X

    def buy_simple(self,money, ethereum, total_money, current_price):
        first_money, first_ethereum, first_total_money = money, ethereum, total_money
        spend = money * 0.1#資産全体の１割を
        money -= spend * (1+self.transaction_fee)
        if money <= 0.0:
            return first_money,first_ethereum,first_total_money

        ethereum += float(spend / current_price)
        total_money = money + ethereum * current_price

        return money, ethereum, total_money

    def sell_simple(self,money, ethereum, total_money, current_price):
            first_money, first_ethereum, first_total_money = money, ethereum, total_money
            spend = ethereum * 0.1
            ethereum -= spend * (1+self.transaction_fee)
            if ethereum <= 0.0:
                return first_money,first_ethereum,first_total_money

            money += float(spend * current_price)
            total_money = money + float(ethereum * current_price)

            return money, ethereum, total_money
    def pass_simple(self,money,ethereum,total_money,current_price):
        total_money = money + float(ethereum * current_price)
        return money,ethereum,total_money

    def SellAndCalcAmoutUsingPrediction(self,pred,money, ethereum, total_money, current_price):
        first_money, first_ethereum, first_total_money = money, ethereum, total_money
        spend = ethereum * 0.5 * (abs(pred)*0.1)
        ethereum -= spend * (1.0+self.transaction_fee)#取引手数料も含めていくら分購入したかを計算。
        if ethereum < 0.0 or abs(pred) < 0.5:##資産がマイナスになる or 予測に自信がない場合
            #何もしない
            return first_money,first_ethereum,first_total_money

        money += float(spend * current_price)#仮想通貨を売却した分、法定通貨が増える。
        total_money = money + float(ethereum * current_price)#仮想通貨と法定通貨両方を合計した資産を計算

        return money, ethereum, total_money
    
    def BuyAndCalcAmoutUsingPrediction(self,pred,money, ethereum, total_money, current_price):
        first_money, first_ethereum, first_total_money = money, ethereum, total_money
        spend = money * (abs(pred)*0.05)#資産全体の何割を一回の取引に使うか abs(pred)にすると便利
        money -= spend * (1.0+self.transaction_fee)#取引手数料も含めていくら分購入したかを計算。
        if money < 0.0 or abs(pred) < 0.5:#資産がマイナスになる or 予測に自信がない場合
            #何もしない
            return first_money,first_ethereum,first_total_money

        ethereum += float(spend / current_price)#法定通貨を消費した分、仮想通貨が増える。
        total_money = money + ethereum * current_price#仮想通貨と法定通貨両方を合計した資産を計算

        return money, ethereum, total_money

    def PassUsingPrediction(self, pred, money, ethereum, total_money, current_price):
        first_money, first_ethereum, first_total_money = money, ethereum, total_money
        return first_money,first_ethereum,first_total_money

    # 配列の長さに気をつける。
    #実験結果：何割の資産を取引に使うかについて、0.01%だけだと＋30ドル 0.1%*pred(予測値によって取引量を変える)で+200ドル
    def simulate_trade(self,price, X_test, model):
        money = 300
        ethereum = 0.01
        total_money = money + np.float64(price[0] * ethereum)
        first_total_money = total_money

        for i in range(0, len(price)):
            print(i)
            current_price = price[i]
            prediction = model.predict(X_test[i])
            pred = prediction[0]
            if pred > 0:
                print("buy")
                money, ethereum, total_money = self.BuyAndCalcAmoutUsingPrediction(pred,money, ethereum, total_money, current_price)
                print("money"+str(money))
            elif pred <= 0:
                print("sell")
                money, ethereum, total_money = self.SellAndCalcAmoutUsingPrediction(pred,money, ethereum, total_money, current_price)
                print("money"+str(money))
        print("FIRST"+str(first_total_money))
        print("FINAL" + str(total_money))
        return total_money

    def update_trading_view(self, current_price, action):
        self.price_history.append(current_price)
        self.trade_history.append(action)

    def reset_trading_view(self):
        self.price_history=[]
        self.trade_history=[]

    def draw_trading_view(self):
        data, date = np.array(self.price_history), np.array([idx for idx in range(0, len(self.price_history))])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(date, data)#,marker='o'
        ax.plot()

        for num in range(0,len(self.price_history)):
            if self.trade_history[num] == 0:
                plt.scatter(date[num], data[num], marker="^", color="green")
            elif self.trade_history[num] == 1:
                plt.scatter(date[num],data[num], marker="v", color="red")

        ax.set_title("Cripto Price")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price[$]")
        plt.grid(fig)
        print("===Show Figure===")
        plt.show(fig)
        
        self.price_history=[]
        self.trade_history=[]
