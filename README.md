# PythonでAIトレードを学ぶ　(ZOOM) 資料
PythonでトレードをするAIを作ります。この会では、純粋に金融工学などで使われているアルゴリズムについて知見を深めることを目的にしています。

# License
MIT

# 注意
ソースコードや資料は、引用している部分に関しては再配布はお控えいただきたいですが、主催者(tomoueno,KatsuyukiSaegusa)が作成した資料やコードについてはMITライセンスです。煮るなり焼くなり好きに使ってください。また、再配布をせずに利用する個人利用などの目的については、自由に改変して良いです。再配布可能なコードについては、冒頭に「再配布可能」と書くようにしますので、しばしお待ちを。

## Install
Gitを使ったクローン方法
```
git clone https://github.com/KanazawaAIMeetup/TradePython/
```
Python3.5.6を使用して、ライブラリをインストールする。
```
pip install -r requirements.txt
```

Requirement already satisfied と表示されるが、import できない場合はこの記事(https://pod.hatenablog.com/entry/2019/04/10/130101)が参考になるかもしれません。

（はじめてPythonをさわる方）
AnacondaのPythonをインストール。3系の最新版をインストール。その後、Anaconda上での仮想環境を作成し、その環境にライブラリをインストールする。なお、仮想環境とは言うもののVM(Virtual Machine)とは全く関係ない。Anacondaは基本的に```pip install```と```conda install```を同じ環境内で併用してはいけないのが基本。ただし、matplotlibやopencvなど画面で表示する必要のあるライブラリは私の場合condaでインストールする場合が多い。(MacやWindowsの場合はpipでインストールしてもうまく動かない場合がある。)

全て「y」や「yes」を選択。
```
conda create -n CriptoTrade python=3.5.6
conda activate CriptoTrade
(以下同様)
pip install -r requirements_gpu.txt
```

## ディレクトリの説明


```
.
├── DATA
│   ├── USDT_BTC_LATEST.csv
│   ├── USDT_ETH_LATEST.csv
│   ├── USDT_LTC_LATEST.csv
│   └── USDT_XMR_LATEST.csv
├── GetHistoricalData
│   └── ResearchGetHistoryData.py
├── LICENSE
├── README.md
├── requirements_gpu.txt
├── TrainExample
│   ├── ChainerRLExample
│   │   ├── DDQNMultipleInput
│   │   │   ├── main_dense.py
│   │   │   ├── main_lstm.py
│   │   │   ├── README.md
│   │   │   └── requirements_chainerrl_gpu.txt
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       └── trade_class.cpython-35.pyc
│   ├── ExampleGeneral
│   │   └── keras_multiple_input.py
│   ├── __init__.py
│   ├── PredictPrice
│   │   └── SimplePricePrediction
│   │       ├── main_price_prediction.py
│   │       └── README.md
│   ├── __pycache__
│   │   └── trade_class.cpython-35.pyc
│   └── trade_class.py
└── Trash
    └── requirements_cpu_old.txt
```
- DATA/ 過去2年分程度の5分刻みの仮想通貨(BitCoin,Ethreum,LiteCoin,Monero)の価格データがあります。
- GetHistoricalData/ ヒストリカルデータをpoloniexから取得するためのスクリプトがあります。
- TrainExample/ tomouenoが学習に使用したサンプルのソースコードがあります。
- TrainExample/trade_class.py 簡易的な取引のエミュレータがあります。手数料を加味して、単純に資産の増減を表示する程度ですが、αやβなどの指数を出す機能も今後は実装したいです。

# 機能
trade_class.py 
### draw_trading_viw()
![](https://i.imgur.com/FKaqrL6.png)
取引の結果を可視化する。緑がBuyで赤がSell



## 引用文献・参考文献
- 「Poloniex - Crypto Asset Exchange」(https://poloniex.com/)
- 「ChainerRL Visualizer」(https://github.com/chainer/chainerrl-visualizer)


