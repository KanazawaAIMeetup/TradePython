#coding: utf-8
'''
MITライセンス　このプログラムについては、改変・再配布可能です
作者： Tomohiro Ueno 
'''
import chainer
#import chainer.functions as F
#import chainer.links as L
from chainerrl.agents import a3c
import chainer.links as L
import chainer.functions as F
import os, sys
print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.pardir)
#from chainerrl.action_value import DiscreteActionValue
#from chainerrl.action_value import QuadraticActionValue
#from chainerrl.optimizers import rmsprop_async
#from chainerrl import links
#from chainerrl import policies
import chainerrl
from chainerrl_visualizer import launch_visualizer
from chainer import Variable, optimizers, Chain, cuda
import numpy as np
import random
import sys,os,copy,traceback
from trade_class import TradeClass
from trade_class import buy_simple, sell_simple, pass_simple
cp = cuda.cupy
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
    x=np.array(x)
    x2 = x - x.mean()
    xstd  = np.std(x2, axis=axis, keepdims=True)
    zscore = x2/(3*xstd)
    return zscore.tolist()

test_term=20000
X_train = []
y_train = []
for idx in range(input_price_len, len(price_data)-test_term):
    X_train.append(standarization(price_data[idx - input_price_len:idx]))#idx番目の価格がトレーニングデータに入らないのが重要。教師がデータに含まれてしまう。
    y_train.append(price_data[idx])

X_test = []
y_test = []
for idx in range(len(price_data)-test_term,len(price_data)):
    #X_train.append(np.flipud(training_set_scaled[i-60:i]))
    X_test.append(standarization(price_data[idx - input_price_len:idx]))
    y_test.append(price_data[idx])

# 時系列データを単純に全結合層に入れるモデル
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=500,input_price_len=input_price_len, n_discrete_size=input_discrete_value_size):
        super(QFunction, self).__init__(
            l_price=L.Linear(input_price_len, n_hidden_channels),#historical_dataの1バッチあたりの長さが入力なので、全体から離散値を引く
            l1=L.Linear(n_hidden_channels+n_discrete_size, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        #print(x)
        #print(x[0])
        #print(x[0][-1])
        x=cp.asarray(x,dtype=cp.float32)

        if x.shape ==  (1,total_input_size):
            h_price = F.tanh(self.l_price(x[:,:-3]))
        else:
            x_price_shape=x[:, :-3].shape
            #print("SHAPE of x[:,:-3]"+str(x_price_shape))
            h_price = F.tanh(self.l_price(x[:,:-3].reshape(x.shape[0],x_price_shape[1])))

        #print(x.shape)
        if x.shape ==  (1,total_input_size):
            h_too_many_buy = F.tanh(cp.array([[x[0][-3]]],dtype=cp.float32))
            h_too_many_sell = F.tanh(cp.array([[x[0][-2]]],dtype=cp.float32))
            h_buysell_count = F.tanh(cp.array([[x[0][-1]]],dtype=cp.float32))
        else:
            #temp=np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
            #          [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
            #          [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
            #          [1.0], [1.0]], dtype=np.float32)
            #print("dtype:"+str(temp.dtype))
            #print("SHAPE of x"+str(x.shape))
            h_too_many_buy = F.tanh(x[:,-3].reshape(x.shape[0],1))
            h_too_many_sell = F.tanh(x[:,-2].reshape(x.shape[0],1))
            h_buysell_count = F.tanh(x[:,-1].reshape(x.shape[0],1))

        concat_layer=F.concat([h_price, h_too_many_buy,h_too_many_sell,h_buysell_count], axis=1)
        h = F.tanh(self.l1(concat_layer))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))



model = QFunction(obs_size=total_input_size, n_actions=n_actions)
gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)
#opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=0.01, alpha=0.98)
opt=chainer.optimizers.Adam(eps=1e-5)
opt.setup(model)

# Set the discount factor that discounts future rewards.
gamma = 0.95

class RandomActor:
    def __init__(self):
        pass
    def random_action_func(self):
        # 所持金を最大値にしたランダムを返すだけ
        return random.randint(0,2)

ra = RandomActor()

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=ra.random_action_func)

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 7)


agent = chainerrl.agents.DoubleDQN(
    model, opt, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100)

def env_execute(action,current_price,next_price,cripto_amount,usdt_amount):

    return reward

try:
  agent.load('chainerRLAgent')
except:
    print("Agent load failed")

#取引を何もしなくても価格の変化に応じて資産が増減するようにする
def reset_info():
    reward=0
    money = 300
    before_money = money
    ethereum = 0.01
    total_money = money + np.float64(y_train[0] * ethereum)
    first_total_money = total_money
    pass_count=0
    buy_sell_count=0
    pass_renzoku_count=0

    return reward,money,before_money,ethereum,total_money,first_total_money,pass_count,buy_sell_count,pass_renzoku_count

def action_if(action,buy_sell_count,pass_count,money,ethereum,total_money,current_price):
    #buy_simple, sell_simple, pass_simple関数は一階層上のtrade_class.py参照。
    if action == 0:
        #Buy
        buy_sell_count += 1
        money, ethereum, total_money = buy_simple(money, ethereum, total_money, current_price)
    elif action == 1:
        #Sell
        buy_sell_count -= 1
        money, ethereum, total_money = sell_simple(money, ethereum, total_money, current_price)
    elif action == 2:
        #PASS
        money, ethereum, total_money = pass_simple(money, ethereum, total_money, current_price)
        pass_count += 1

    total_money=money+ethereum*current_price

    return buy_sell_count, pass_count, money, ethereum, total_money

'''
reward: 強化学習のトレーニングに必要な報酬
money: 現在の総資産(スタート時は300ドルなど任意の値で初期化)
before_money: 1ステップ前の資産
ehereum: 資産として保持している仮想通貨の量
total_money: 法定通貨と仮想通貨両方を合計した現在の総資産
first_total_money: スタート時の総資産　運用成績を評価するために使用
pass_count:　何回売買をせずに見送ったか。passをした合計回数を記録
#buy_sell_count: 今までの取引の中でBuyとSellにどれだけ偏りがあるかを表す数。Buyされる度に+1,Sellされる度に-1される。つまり、正数の場合はBuyばかりされていて、負数の場合はSellばかりされている。
pass_renzoku_count: 取引せずに見送るPassを何回連続で行なったか。学習の状況や取引を可視化するために作成した。
'''
reward, money, before_money, ethereum, total_money, first_total_money, pass_count, buy_sell_count, pass_renzoku_count = reset_info()
for episode in range(0,5):
    reward, money, before_money, ethereum, total_money, first_total_money, pass_count, buy_sell_count, pass_renzoku_count = reset_info()
    for idx in range(0, len(y_train)):#TODO
        if idx % 1000 == 0:
            print("EPISODE:"+str(episode)+"LOOP IDX:"+str(idx))
            print("BEGGINING MONEY:" + str(first_total_money))
            print("Current Total MONEY:" + str(total_money))
        current_price = y_train[idx]
        buy_sell_num_flag=[1.0,0.0,abs(buy_sell_count)] if buy_sell_count >= 1 else [0.0,1.0,abs(buy_sell_count)]
        state_data=np.array(X_train[idx]+buy_sell_num_flag,dtype='f')
        action = agent.act_and_train(state_data, reward)#idx+1が重要。
        #print(agent.get_statistics())
        tradecl.update_trading_view(current_price, action)
        reward=0

        buy_sell_count, pass_count, money, ethereum, total_money = \
            action_if(action,buy_sell_count,pass_count,money,ethereum,total_money,current_price)

        reward += 0.01 * (total_money - before_money)  # max(current_price-bought_price,0)##
        before_money = total_money

        if False:#idx % 5000 == 500:#学習状況の可視化のためのツール　普段はFalseにする。
            print("Initial MONEY:"+str(first_total_money))
            print("Current Total MONEY:" + str(total_money))
            print("ある回数の予測において、Passは"+str(pass_count)+"回")
            print("ある回数の予測において、終わった後のbuy_sell_count:" + str(buy_sell_count) + "回"+ "(買いの回数が多い)" if buy_sell_count > 0 else "(売りの回数が多い)")
            pass_count=0
            try:
                #matplotlib(GUI)でどのタイミングで売買を行なっているか可視化。
                tradecl.draw_trading_view()
            except:
                pass
            agent.save('chainerRLTradeAgent')
    #インデントこれであってる
    buy_sell_num_flag = [1.0, 0.0, abs(buy_sell_count)] if buy_sell_count >= 1 else [0.0, 1.0, abs(buy_sell_count)]
    agent.stop_episode_and_train(X_train[-1]+buy_sell_num_flag, reward, True)#エピソード（価格データの最初から最後までを辿ること）を一旦停止
    #強化学習の初期では少ない手数料で、学習後半には手数料を少しずつ増やしていく

print("Training END")
print("Passは" + str(pass_count) + "回")
print("ある回数の予測において、終わった後のbuy_sell_count" + str(buy_sell_count) + "回"+ "　買いの回数が多い" if buy_sell_count > 0 else "　売りの回数が多い")

print("Initial MONEY" + str(first_total_money))
print("FINAL MONEY:" + str(total_money))
# Save an agent to the 'agent' directory
agent.save('chainerRLAgentFinal-Dense')

'''
ACTION_MEANINGS = {
    0: 'a',
    1: 'b',
    2: 'c'
}

launch_visualizer(
    agent,  # required
    env,  # required
    ACTION_MEANINGS,  # required
    port=5002,  # optional (default: 5002)
    log_dir='log_space',  # optional (default: 'log_space')
    raw_image_input=False,  # optional (default: False)
    contains_rnn=False,  # optional (default: False)
)
'''

#print(traceback.format_exc())

reward, money, before_money, ethereum, total_money, first_total_money, pass_count, buy_sell_count, pass_renzoku_count = reset_info()
tradecl.reset_trading_view()#グラフの描画をリセットする
for idx in range(0, len(y_test)):
    current_price = y_test[idx]#添字間違えないように
    buy_sell_num_flag = [1.0, 0.0, abs(buy_sell_count)] if buy_sell_count >= 1 else [0.0, 1.0, abs(buy_sell_count)]
    state_data = np.array(X_test[idx] + buy_sell_num_flag, dtype='f')
    action = agent.act(state_data)  # 教師が入力に入らないように。
    tradecl.update_trading_view(current_price, action)
    buy_sell_count, pass_count, money, ethereum, total_money = \
            action_if(action,buy_sell_count,pass_count,money,ethereum,total_money,current_price)
    #buy_sell_count, pass_count, money, ethereum, total_money
    before_money = total_money

pass_count=0

print("==========================================")
print("Test END")
print("pass：　合計" + str(pass_count) + "回")
print("ある回数の予測において、終わった後のbuy_sell_count" + str(buy_sell_count) + "回"+ "　買いの回数が多い" if buy_sell_count > 0 else "　売りの回数が多い")
print("Initial MONEY" + str(first_total_money))
print("FINAL MONEY:" + str(total_money))
print(os.path.basename(__file__))

#matploblibでトレードの結果をグラフで可視化
try:
    tradecl.draw_trading_view()
except:
    pass
