# coding: utf-8
import chainer
# import chainer.functions as F
# import chainer.links as L
from chainerrl.agents import a3c
import chainer.links as L
import chainer.functions as F
import os, sys

print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.pardir)
# from chainerrl.action_value import DiscreteActionValue
# from chainerrl.action_value import QuadraticActionValue
# from chainerrl.optimizers import rmsprop_async
# from chainerrl import links
# from chainerrl import policies
import chainerrl
from chainerrl_visualizer import launch_visualizer
from chainer import Variable, optimizers, Chain, cuda
from chainerrl.recurrent import RecurrentChainMixin
import numpy as np
import random
import sys, os, copy, traceback
from trade_class import TradeClass

cp = cuda.cupy
from sklearn import preprocessing

ss = preprocessing.StandardScaler()

print(os.path.basename(__file__))

tradecl = TradeClass()
price_data = tradecl.ReadPoloniexCSV()
np.set_printoptions(threshold=np.inf)
print("price_data idx 0-10" + str(price_data[0:10]))
print("price_data idx last 10" + str(price_data[-1]))

'''
def getDataPoloniex():
    polo = poloniex.Poloniex()
    polo.timeout = 10
    chartUSDT_BTC = polo.returnChartData('USDT_BTC', period=300, start=time.time() - 1440 * 60 * 500, end=time.time())  # 1440(min)*60(sec)=DAY
    tmpDate = [chartUSDT_BTC[i]['date'] for i in range(len(chartUSDT_BTC))]
    date = [datetime.datetime.fromtimestamp(tmpDate[i]) for i in range(len(tmpDate))]
    data = [float(chartUSDT_BTC[i]['open']) for i in range(len(chartUSDT_BTC))]
    return date, data
'''

# time_date, price_data = getDataPoloniex()

input_price_len = 2000
input_discrete_value_size = 3
total_input_size = input_price_len + input_discrete_value_size
n_actions = 3


# obs_size = input_len+n_actions#shape#env.observation_space.shape[0]

def standarization(x, axis=None):
    x = np.array(x)
    x2 = x - x.mean()
    xstd = np.std(x2, axis=axis, keepdims=True)
    zscore = x2 / (3 * xstd)
    return zscore.tolist()


test_term = 20000
X_train = []
y_train = []
#temp_test_number_delete_necessary=50000
for idx in range(input_price_len, len(price_data) - test_term):
    # X_train.append(np.flipud(training_set_scaled[i-60:i]))
    X_train.append(standarization(price_data[idx - input_price_len:idx]))  # idx番目の価格がトレーニングデータに入らないのが重要。教師がデータに含まれてしまう。
    y_train.append(price_data[idx])

X_test = []
y_test = []
for idx in range(len(price_data) - test_term, len(price_data)):
    # X_train.append(np.flipud(training_set_scaled[i-60:i]))
    X_test.append(standarization(price_data[idx - input_price_len:idx]))
    y_test.append(price_data[idx])

class QFunction(chainer.Chain, RecurrentChainMixin):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50, input_price_len=input_price_len,
                 n_discrete_size=input_discrete_value_size):
        self.first_flag=False
        self.input_price_len=input_price_len
        self.n_hidden_channels=n_hidden_channels
        super(QFunction, self).__init__(
            l_price=L.LSTM(input_price_len,n_hidden_channels),  # historical_dataの1バッチあたりの長さが入力なので、全体から離散値を引く
            l1=L.Linear(n_hidden_channels + n_discrete_size, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))
    def reset_state(self):
        self.l_price.reset_state()
    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """

        #x = np.asarray(x, dtype=cp.float32)
        #x_price_shape = x[:, :-3].shape
        if x.shape == (1, total_input_size):
            #print(x[:, :-3].shape)
            if True:#self.first_flag:
                first_timestep_input=[]
                for i in range(0,32):
                    first_timestep_input.extend(x[:, :-3].reshape(1,self.input_price_len ))
                first_timestep_input=np.array(first_timestep_input)
                #print("first time step input"+str(first_timestep_input.shape))
                h_price=F.tanh(self.l_price(first_timestep_input))
                h_price=h_price[0].reshape(1,self.n_hidden_channels)
                #print(h_price.shape)
                self.first_flag = False
            else:
                h_price = F.tanh(self.l_price(x[:, :-3].reshape(1,self.input_price_len)))
        else:
            #print(x_price_shape)
            #print(x[:, :-3].reshape(x.shape[0], x_price_shape[1]).shape)
            #print(x[:, :-3].shape)
            # print("SHAPE of x[:,:-3]"+str(x_price_shape))
            #print("else first layer")
            #print("x_price_shape"+str(x[:,:-3].shape))
            #print("else l_price input shape:"+str(x[:, :-3].reshape(x_price_shape[0],1000,1).shape))
            h_price = F.tanh(self.l_price(x[:, :-3].reshape(32,self.input_price_len)))#(32,1,1000)はダメ。array size変えるなというエラー

        #h_price=h_price.reshape()
        #print("h_price.shape")
        #print(h_price.shape)
        # print(x.shape)
        if x.shape == (1, total_input_size):
            h_too_many_buy = F.tanh(np.array([[x[0][-3]]], dtype=np.float32))
            h_too_many_sell = F.tanh(np.array([[x[0][-2]]], dtype=np.float32))
            h_buysell_count = F.tanh(np.array([[x[0][-1]]], dtype=np.float32))
        else:
            # temp=np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
            #          [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
            #          [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
            #          [1.0], [1.0]], dtype=np.float32)
            # print("dtype:"+str(temp.dtype))
            # print("SHAPE of x"+str(x.shape))
            h_too_many_buy = F.tanh(x[:, -3].reshape(x.shape[0], 1))
            h_too_many_sell = F.tanh(x[:, -2].reshape(x.shape[0], 1))
            h_buysell_count = F.tanh(x[:, -1].reshape(x.shape[0], 1))

        #print("discrete value shape")
        #print(h_too_many_sell.shape)

        concat_layer = F.concat([h_price, h_too_many_buy, h_too_many_sell, h_buysell_count], axis=1)
        h = F.tanh(self.l1(concat_layer))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


model = QFunction(obs_size=total_input_size, n_actions=n_actions)
gpu_device = 0
cuda.get_device(gpu_device).use()
#model.to_gpu(gpu_device)
model.reset_state()
# opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=0.01, alpha=0.98)
opt = chainer.optimizers.Adam(eps=1e-5)
opt.setup(model)

# Set the discount factor that discounts future rewards.
gamma = 0.95


class RandomActor:
    def __init__(self):
        pass

    def random_action_func(self):
        # 所持金を最大値にしたランダムを返すだけ
        return random.randint(0, 2)


ra = RandomActor()

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=ra.random_action_func)

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 7)

agent = chainerrl.agents.DoubleDQN(
    model, opt, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100)

# one_percent=0.01
buy_sell_fee = 0.0  # 0.0001#0.01%の手数料#one_percent*0.000001


def buy_simple(money, ethereum, total_money, current_price):
    first_money, first_ethereum, first_total_money = money, ethereum, total_money
    spend = money * 0.5
    money -= spend * (1 + buy_sell_fee)
    if money <= 0.0:
        return first_money, first_ethereum, first_total_money

    ethereum += float(spend / current_price)
    total_money = money + ethereum * current_price

    return money, ethereum, total_money


def sell_simple(money, ethereum, total_money, current_price):
    first_money, first_ethereum, first_total_money = money, ethereum, total_money
    spend = ethereum * 0.5
    ethereum -= spend * (1 + buy_sell_fee)
    if ethereum <= 0.0:
        return first_money, first_ethereum, first_total_money

    money += float(spend * current_price)
    total_money = money + float(ethereum * current_price)

    return money, ethereum, total_money


def pass_simple(money, ethereum, total_money, current_price):
    total_money = money + float(ethereum * current_price)
    return money, ethereum, total_money


# try:
#  agent.load('chainerRLAgent')
# except:
#    print("Agent load failed")


# トレーニング
# 放っておいてtotal_priceが上がることもある。今のプログラムだと放っておいても値段変わらない
def reset_info():
    reward = 0
    money = 300
    before_money = money
    ethereum = 0.01
    total_money = money + np.float64(y_train[0] * ethereum)
    first_total_money = total_money
    pass_count = 0
    buy_sell_count = 0

    return reward, money, before_money, ethereum, total_money, first_total_money, pass_count, buy_sell_count


def action_if(action, buy_sell_count, pass_count, money, ethereum, total_money, current_price):
    if action == 0:
        # print("buy")
        buy_sell_count += 1
        money, ethereum, total_money = buy_simple(money, ethereum, total_money, current_price)
    elif action == 1:
        # print("sell")
        buy_sell_count -= 1
        money, ethereum, total_money = sell_simple(money, ethereum, total_money, current_price)
    elif action == 2:
        # print("PASS")
        money, ethereum, total_money = pass_simple(money, ethereum, total_money, current_price)
        pass_count += 1

    total_money = money + ethereum * current_price

    return buy_sell_count, pass_count, money, ethereum, total_money


def print_info_interval(first_total_money, total_money, pass_count, buy_sell_count):
    print("BEGGINING MONEY:" + str(first_total_money))
    print("Current Total MONEY:" + str(total_money))
    print("1000回中passは" + str(pass_count) + "回")
    print("1000回終わった後のbuy_sell_countは" + str(buy_sell_count) + "回")
    pass_count = 0
    try:
        tradecl.draw_trading_view()
    except:
        pass
    agent.save('chainerRLAgent-LSTM')


reward, money, before_money, ethereum, total_money, first_total_money, pass_count, buy_sell_count = reset_info()
for episode in range(0, 5):
    reward, money, before_money, ethereum, total_money, first_total_money, pass_count, buy_sell_count = reset_info()
    for idx in range(0, len(y_train)):  # TODO
        if idx % 1000 == 0:
            print("EPISODE:" + str(episode) + "LOOP IDX:" + str(idx))
            print("BEGGINING MONEY:" + str(first_total_money))
            print("Current Total MONEY:" + str(total_money))
        current_price = y_train[idx]
        buy_sell_num_flag = [1.0, 0.0, abs(buy_sell_count)] if buy_sell_count >= 1 else [0.0, 1.0, abs(buy_sell_count)]
        state_data = np.array(X_train[idx] + buy_sell_num_flag, dtype='f')
        #state_data=state_data.reshape(state_data.shape[0],1)#This code is necessary only for LSTM
        action = agent.act_and_train(state_data, reward)  # idx+1が重要。
        # print(agent.get_statistics())
        tradecl.update_trading_view(current_price, action)

        buy_sell_count, pass_count, money, ethereum, total_money = action_if(action, buy_sell_count, pass_count, money,
                                                                             ethereum, total_money, current_price)

        reward = 0
        reward += 0.01 * (total_money - before_money)  # max(current_price-bought_price,0)##

        before_money = total_money

        if False:  # idx % 5000 == 500:
            print_info_interval(first_total_money, total_money, pass_count, buy_sell_count)

    # インデントこれであってるから変更しないこと。
    buy_sell_num_flag = [1.0, 0.0, abs(buy_sell_count)] if buy_sell_count >= 1 else [0.0, 1.0, abs(buy_sell_count)]
    state_data=np.array(X_train[-1]+buy_sell_num_flag, dtype='f')
    #state_data = state_data.reshape(state_data.shape[0], 1)#This line is necessary only for LSTM
    agent.stop_episode_and_train(state_data, reward, True)
    # buy_sell_fee=buy_sell_fee*10 #TODO #強化学習の初期では少ない手数料で、少しずつ増やしていく

print("=====Training END============================")
print("START MONEY" + str(first_total_money))
print("FINAL MONEY:" + str(total_money))
print("pass(at the end of training):" + str(pass_count))
print("buy_sell_count" + str(buy_sell_count))
print("buy_sell_fee:" + str(buy_sell_fee))

# Save an agent to the 'agent' directory
agent.save('chainerRLAgentFinal-zerofee-lstm-1000inputlen')

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

# print(traceback.format_exc())

reward, money, before_money, ethereum, total_money, first_total_money, pass_count, buy_sell_count = reset_info()
tradecl.reset_trading_view()
for idx in range(0, len(y_test)):
    current_price = y_test[idx]  # 重要
    buy_sell_num_flag = [1.0, 0.0, abs(buy_sell_count)] if buy_sell_count >= 1 else [0.0, 1.0, abs(buy_sell_count)]
    state_data = np.array(X_test[idx] + buy_sell_num_flag, dtype='f')
    #state_data = state_data.reshape(state_data.shape[0], 1)
    action = agent.act(state_data)  # 教師が入力に入らないように。
    tradecl.update_trading_view(current_price, action)
    buy_sell_count, pass_count, money, ethereum, total_money = \
        action_if(action, buy_sell_count, pass_count, money, ethereum, total_money, current_price)
    # before_money = total_money

print("====================TEST======================")
print("START MONEY" + str(first_total_money))
print("FINAL MONEY:" + str(total_money))
print("pass_count：" + str(pass_count))
print("buy_sell_count(at the end of TEST):" + str(buy_sell_count))
print("buy_sell_fee:" + str(buy_sell_fee))

try:
    tradecl.draw_trading_view()
except:
    pass

print(os.path.basename(__file__))



