# TA-Libと関係するライブラリのインストールコマンド

## Ubuntuの場合

terminalで以下のコマンドを入力する。
```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -zxvf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install
cd ../
rm -rf ta-lib-0.4.0-src.tar.gz
rm -rf ta-lib
pip install TA-Lib
```

もしくは、Install.shなど適当な名前を付け、上記のコマンドを適当な場所に保存し、

```bash
bash Install.sh
```
とコマンドを打つと自動でインストールが始まる。

## Colaboratoryの場合
以下を参照。

https://investment.abbamboo.com/trading-tools/google-colab-install-ta-lib/

## サンプルプログラムを実行する際に必要かもしれないライブラリ
追加で、以下のライブラリもインストールすると良い。

```
conda install quandl
```

## 何かに役立つかもしれないメモ
TA-Libのインストールは面倒なので、whlをダウンロードするか、brewでインストールするのが簡単らしい。
- https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib