#coding: utf-8
'''
MITライセンス　このプログラムについては、改変・再配布可能です
著作者： Tomohiro Ueno (kanazawaaimeetup@gmail.com)

Usage: ddqn-multiple-inputディレクトリから実行する。
python main.py
注意：評価する場合は、正解データのリークが起きないようにする。Train,Validation,Testの分割方法に気をつける
このプログラムは、リークを厳密に回避していません！
実行を早くするため、test_term=120000 epochs=1 となっていますが、実際に評価する場合はtest_term=20000、 epocsも20などに直して下さい。
詳しくは、このソースコード内を「TODO」で検索して下さい。
'''
import sys, os
sys.path.append("..")
sys.path.append("../..")
sys.path.append(os.getcwd())
sys.path.append(os.pardir)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import random
import sys,os,copy,traceback
from sklearn.utils import shuffle
from trade_class import TradeClass
from sklearn import preprocessing
ss = preprocessing.StandardScaler()

print(os.path.basename(__file__))

tradecl=TradeClass()
price_data = tradecl.ReadPoloniexCSV()
np.set_printoptions(threshold=np.inf)
print("price_data idx 0-10"+str(price_data[0:10]))
print("price_data idx last 10"+str(price_data[-1]))

input_price_len=400
input_discrete_value_size=3
total_input_size = input_price_len+input_discrete_value_size
n_actions=3

#obs_size = input_len+n_actions#shape#env.observation_space.shape[0]
#データを標準化して、ディープラーニングで学習しやすくする。
def standarization(x, axis = None):
    x = np.array(x)
    x2 = x - x.mean()
    xstd  = np.std(x2, axis=axis, keepdims=True)
    zscore = x2/(3*xstd)
    return zscore.tolist()

test_term=180000#TODO 20000に直す。
valid_term = 4000
X_train = []
y_train = []
X_valid = []
y_valid = []
for idx in range(input_price_len, len(price_data)-test_term-valid_term):
    X_train.append(standarization(price_data[idx - input_price_len:idx]))#idx番目の価格がトレーニングデータに入らないのが重要。教師がデータに含まれてしまう。
    y_train.append(price_data[idx])
    if np.array(X_train[-1]).shape != (400,):
        print(np.array(X_train[-1]).shape)

for idx in range(len(price_data)-test_term-valid_term,len(price_data)-test_term):
    X_valid.append(standarization(price_data[idx - input_price_len:idx]))#idx番目の価格がトレーニングデータに入らないのが重要。教師がデータに含まれてしまう。
    y_valid.append(price_data[idx])

X_test = []
y_test = []
for idx in range(len(price_data)-test_term,len(price_data)):
    X_test.append(standarization(price_data[idx - input_price_len:idx]))
    y_test.append(price_data[idx])

X_train, y_train = shuffle(X_train, y_train, random_state=1234)

#取引を何もしなくても価格の変化に応じて資産が増減するようにする
def reset_info():
    money = 300
    before_money = money
    cripto = 0.01
    total_money = money + np.float64(y_train[0] * cripto)
    first_total_money = total_money
    pass_count = 0
    buy_sell_count = 0
    pass_renzoku_count = 0
    before_action = 2 

    return money,before_money,cripto,total_money,first_total_money,pass_count,buy_sell_count,pass_renzoku_count, before_action

def action_if(action,buy_sell_count,pass_count,money,cripto,total_money,current_price):
    #buy_simple, sell_simple, pass_simple関数は一階層上のtrade_class.py参照。
    ratio = 0.5
    if action == 0:
        #Buy
        buy_sell_count += 1
        money, cripto, total_money = tradecl.buy_using_ratio(money, cripto, total_money, current_price, ratio)
    elif action == 1:
        #Sell
        buy_sell_count -= 1
        money, cripto, total_money = tradecl.sell_using_ratio(money, cripto, total_money, current_price, ratio)
    elif action == 2:
        #PASS
        money, cripto, total_money = tradecl.pass_using_ratio(money, cripto, total_money, current_price, ratio)
        pass_count += 1

    total_money=money+cripto*current_price

    return buy_sell_count, pass_count, money, cripto, total_money

# Kerasでモデルを定義
#model = Sequential()
# 1つの学習データのStep数(今回は25)
print("Model Define")
length_of_sequence = input_price_len 
in_out_neurons = 1
n_hidden = 300

model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
optimizer = Adam(lr=0.001)
model.compile(loss="mean_absolute_error", optimizer=optimizer)

# 教師データを正規化して、スケールを合わせる
y_train_normalized = np.array(y_train) / 10000
y_test_normalized = np.array(y_test) / 10000

'''
#X_trainなど入力はLSTMを使う場合、以下のような形に変形する必要がある。
array([[1],
       [2],
       [3],
       [4]])
'''
X_train = np.array(X_train).reshape(len(X_train),input_price_len,1)
X_valid = np.array(X_valid).reshape(len(X_valid),input_price_len,1)
X_test = np.array(X_test).reshape(len(X_test),input_price_len,1)

print(X_train.shape)
print(X_train[0:3])

print("Training Starts")
model.fit(X_train, y_train_normalized,
          batch_size=64,
          validation_data=(X_valid, y_valid),
          verbose=1,
          epochs=1)

'''
reward: 強化学習のトレーニングに必要な報酬
money: 現在の総資産(スタート時は300ドルなど任意の値で初期化)
before_money: 1ステップ前の資産
cripto: 資産として保持している仮想通貨の量
total_money: 法定通貨と仮想通貨両方を合計した現在の総資産
first_total_money: スタート時の総資産　運用成績を評価するために使用
pass_count:　何回売買をせずに見送ったか。passをした合計回数を記録
#buy_sell_count: 今までの取引の中でBuyとSellにどれだけ偏りがあるかを表す数。Buyされる度に+1,Sellされる度に-1される。つまり、正数の場合はBuyばかりされていて、負数の場合はSellばかりされている。
pass_renzoku_count: 取引せずに見送るPassを何回連続で行なったか。学習の状況や取引を可視化するために作成した。
'''

#TODO モデルの保存
tradecl.reset_trading_view()#グラフの描画をリセットする
before_price = y_test[0]
before_pred = y_test[0]

'''
一つの入力データについてだけ予測したい場合
#pred_array = model.predict(np.array([input_data]))  # 教師が入力に入らないように。
#pred = pred_array.tolist()[0][0]#出力がnumpy型のa=np.array([[0.5467384]])のようになっている
'''
pred_array = model.predict(X_test)#TODO X_test[:2000]を、X_testに変更する。     
print(pred_array.shape)
print(pred_array[0:2])

money, before_money, cripto, total_money, first_total_money, pass_count, buy_sell_count,pass_renzoku_count, before_action = reset_info()
for idx in range(0, len(pred_array.tolist())-1):#TODO 配列の長さを元に戻す。
    current_price = y_test[idx]#添字間違えないように

    pred = pred_array[idx][0]

    pred_diff = pred - before_pred
    if pred_diff > 0.00005 and before_action == 2:
        #入札する
        action = 0
    elif pred_diff < -0.00005 and before_action == 2:
        #入札する
        action = 1
    elif pred_diff > 0.00005 and before_action == 1:
        #決済する
        action = 0
    elif pred_diff < 0.00005 and before_action == 0:
        #決済する
        action = 1
    else:
        #何もしない
        action = 2

    tradecl.update_trading_view(current_price, action)
    buy_sell_count, pass_count, money, cripto, total_money = \
            action_if(action,buy_sell_count,pass_count,money,cripto,total_money,current_price)
    before_money = total_money
    before_price = current_price
    before_pred = pred
    before_action = action

print("====================TEST======================")
print("START MONEY" + str(first_total_money))
print("FINAL MONEY:" + str(total_money))
print("pass_count：" + str(pass_count))
print("buy_sell_count(at the end of TEST):" + str(buy_sell_count))


#matploblibでトレードの結果をグラフで可視化
try:
    tradecl.draw_trading_view()
except:
    print(traceback.format_exc()) 
    print("tradecl.draw_trading_view FAILED!!")

