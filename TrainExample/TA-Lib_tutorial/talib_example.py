import numpy as np
import pandas as pd
import talib
import quandl

df = quandl.get('NIKKEI/INDEX')  # quandlの場合

close = np.array(df['Close Price'])
close[:5]

output = close.copy()
cols = ['Original']
# 単純移動平均(SMA: Simple Moving Average)
output = np.c_[output, talib.SMA(close)]
cols += ['SMA']
# 加重移動平均(WMA: Weighted Moving Average)
output = np.c_[output, talib.WMA(close)]
cols += ['WMA']
# 指数移動平均(EMA: Exponential Moving Average)
output = np.c_[output, talib.EMA(close)]
cols += ['EMA']
# ２重指数移動平均(DEMA: Double Exponential Moving Average)
output = np.c_[output, talib.DEMA(close)]
cols += ['DEMA']
# ３重指数移動平均(TEMA: Triple Exponential Moving Average)
output = np.c_[output, talib.T3(close)]
cols += ['TEMA']
# 三角移動平均(TMA: Triangular Moving Average)
output = np.c_[output, talib.TRIMA(close)]
cols += ['TMA']
# Kaufmanの適応型移動平均(KAMA: Kaufman Adaptive Moving Average)
output = np.c_[output, talib.KAMA(close)]
cols += ['KAMA']
# MESAの適応型移動平均(MAMA: MESA Adaptive Moving Average)
for arr in talib.MAMA(close):
    output = np.c_[output, arr]
cols += ['MAMA', 'FAMA']
# トレンドライン(Hilbert Transform - Instantaneous Trendline)
output = np.c_[output, talib.HT_TRENDLINE(close)]
cols += ['HT_TRENDLINE']
# ボリンジャー・バンド(Bollinger Bands)
for arr in talib.BBANDS(close):
    output = np.c_[output, arr]
cols += ['BBANDS_upperband', 'BBANDS_middleband', 'BBANDS_lowerband']
# MidPoint over period
output = np.c_[output, talib.MIDPOINT(close)]
cols += ['MIDPOINT']

# 変化率(ROC: Rate of change Percentage)
output = np.c_[output, talib.ROCP(close)]
cols += ['ROC']
# モメンタム(Momentum)
output = np.c_[output, talib.MOM(close)]
cols += ['MOM']
# RSI: Relative Strength Index
output = np.c_[output, talib.RSI(close)]
cols += ['RSI']
# MACD: Moving Average Convergence/Divergence
for arr in talib.MACD(close):
    output = np.c_[output, arr]
cols += ['MACD', 'MACD_signal', 'MACD_hist']
# APO: Absolute Price Oscillator
output = np.c_[output, talib.APO(close)]
cols += ['APO']
# PPO: Percentage Price Oscillator
output = np.c_[output, talib.PPO(close)]
cols += ['PPO']
# CMO: Chande Momentum Oscillator
output = np.c_[output, talib.CMO(close)]
cols += ['CMO']

# ヒルベルト変換 - Dominant Cycle Period
output = np.c_[output, talib.HT_DCPERIOD(close)]
cols += ['HT_DCPERIOD']
# ヒルベルト変換 - Dominant Cycle Phase
output = np.c_[output, talib.HT_DCPHASE(close)]
cols += ['HT_DCPHASE']
# ヒルベルト変換 - Phasor Components
for arr in talib.HT_PHASOR(close):
    output = np.c_[output, arr]
cols += ['HT_PHASOR_inphase', 'HT_PHASOR_quadrature']
# ヒルベルト変換 - SineWave
for arr in talib.HT_SINE(close):
    output = np.c_[output, arr]
cols += ['HT_SINE_sine', 'HT_SINE_leadsine']
# ヒルベルト変換 - Trend vs Cycle Mode
output = np.c_[output, talib.HT_TRENDMODE(close)]
cols += ['HT_TRENDMODE']
output.shape

# 60日単純移動平均
output = np.c_[output, talib.SMA(close, timeperiod=60)]
cols += ['SMA60']
# 15日ボリンジャー・バンド
for arr in talib.BBANDS(close, timeperiod=15, nbdevup=2, nbdevdn=2, matype=0):
    output = np.c_[output, arr]
cols += ['BBANDS15_upperband', 'BBANDS15_middleband', 'BBANDS15_lowerband']
# 21日RSI
output = np.c_[output, talib.RSI(close, timeperiod=21)]
cols += ['RSI21']

data = pd.DataFrame(output, index=df.index, columns=cols)
data.tail()

data.to_csv('NIKKEI_ta.csv')