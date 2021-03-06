# ソースコードの説明
- main.py (gpuで計算) 時系列の価格データを単純な全結合層を用いた強化学習で学習を行なったバージョン
- main_lstm.py (cpuで計算) 時系列の価格データをLSTMを用いた強化学習で学習を行なったバージョン

# main_lstm.pyの説明

buy_sell_num_flagは、買った回数と売った回数を引き算した差分の値が入っています。
https://github.com/KanazawaAIMeetup/TradePython/blob/master/TrainExample/ChainerRLExample/DDQNMultipleInput/main_lstm.py#L236
ここの行では、
[1.0, 0.0, abs(buy_sell_count)]
のような配列が代入されますが、配列の0番目、1番目には買いが多いか、売りが多いかのフラッグが、2番目には正の値で、回数が格納されています。
このような配列を特徴量に加えた理由は、取引をAIにさせる過程で、売りばかりしたり、もしくは買いばかりして取引の効率が悪くなるのを強化学習で防ごうという意図があります。（建て玉が増えすぎてしまうのを防ぐイメージ）
main_lstm.pyに関して申しますと、Chainerの仕様でchainer-rlを使おうとすると、どうしてもハードコーディングしなければならず（私の調べた範囲では）、特徴量を
state_data = np.array(X_train[idx] + buy_sell_num_flag, dtype='f')
として一つの変数にまとめた後に、再びLSTMの内部で別々に入力できるよう、取り出しています。
ネットワークは、時系列データを処理するLSTMの出力層の値と、フラッグの値を最後の全結合層でマージしています。Fuse-Netと似ている構造です。


# ソースコード作成のメモ

## 注意点（作者用）

- cupyの配列とnumpyの配列を混同してエラーが出る。to_gpu()使うと、ネットワークの定義部分でエラーが出る。
- https://www.iandprogram.net/entry/2016/06/27/182548
- testとvalidationにデータを分けるときに、シャッフルしないこと。
- rewardを計算して代入する部分のインデントに注意。
- データの標準化の関数にバグがあった。->2/28　修正。
## 実装上のTODO
- [x] データをtrainとtestに分割する
- [x] chainerrlを使って、複数の離散値も入力として受け付けることができるように改造する
- [ ] 各々の期間のヒストリカルデータ(csv)をつなぎ合わせる
- [ ] LSTM,CNN,Dilated Convolutionなどを試してみる
- [ ] ライブラリchainerのLSTMのソースコードを改造して、バッチサイズ1や32両方を与える場合でもエラーが出ないようにする。

## 実装のアイデア TODO
- [ ] 報酬の与え方を工夫したほうが良い。例えば、価格が上がるか下がるかや、資産が常に一定だとして、何％増えたか減ったかなど　今は資産が増えたか、減ったかで計算しているので不安定。
- [ ] 時系列データをある時点の価格を平均として、分散を考慮した正規化もしたほうが良い
- [ ] どのくらいの量を売買するかも強化学習で求める
## 以前に思いついたアイデア
- [ ] 金融ビッグデータ分析のコンテスト主催する。データ自体も自分で用意してくださいという趣旨でやる。
- [ ] 最終的にUSDTとビットコインの価格を比べて、取引所５つの平均の価格を取る。
- [ ] いろいろな種類の仮想通貨のデータ全てを0~1に正規化して、まとめて学習してしまう。
- [ ] 今から考えて5分後の価格の予測値と現在の価格を比べて、買うかどうか判断するんじゃなくて、今から考えて５分前から予測した現在の価格を比べる。（予測同士を比べて、価格が上昇する見込みがあったら買う）
- [ ] ビットコインのデータで学習する場合は、ずっと右肩上がりなので常に買ったほうが良いという予想が出やすい。これを回避するために、一日おきとか短い期間でデータを切り出して予測する
- [ ] 移動平均線も機械学習の入力データとして使う。
- [ ] v3つ先までの価格も予測する。
- [ ] ボリンジャーとかの曲線も使用する。
- [ ] 長期の相場の状況も反映する。（１時間足、一日足なども必要）　ネットワークで二つのモデルを作り、合体させるとか
- [ ] 板情報も使って予測する。強化学習する。
- [ ] ５分おきのデータしかなくても大丈夫。最新のデータを取ってきて、それをLSTMに食わせる。５分後までまたなくても大丈夫。最新のデータとして入力を受け付けるレイヤーを用意する。
- [ ] 今から何％動くかを分類問題として予測する。中野先生の天体予測のアイデアを使う。
- [ ] １日後の価格の平均をとって、それを予測する。
- [ ] 今の価格を1.0と置いて、過去の価格を1.2とか0.9とかで表す。
- [ ] 株価をフーリエ変換して傾向を取り出す
- [ ] 入札が消費されなかった場合に、困る。５分待っても消費されなかった場合とか
- [ ] 上がると思った時のみ現金を仮想通貨に変え、それ以外は現金で持ち続けるようなアルゴリズムにする。
- [ ] 海外の取引所のデータを入力として、日本の取引所の価格を予想する。　様々な取引所のデータを使う。
- [ ] 時間のデータを入れる。昼なのか夜なのか、休日なのか
- [ ] 期待値で計算する　上がるか下がるかを出力してみて、当たったら正解確率100パーセント（自信ある）外れたら正解率０パーセントで答えあわせする。テストデータでは0~100%で出力 買うかどうかは、(上がる確率)*(利益)+（下がる確率）*(損)で計算する。
- [ ] STOP LIMITを設定するプログラムを書く
- [ ] 持っている資産のバランスも強化学習の入力とする
- [ ] 価格の予測をし、予測結果を強化学習の入力として使用する。 (edited)
- [ ] 強化学習でどれくらいの比率と量で仮想通貨を買えばよいか、売ればよいかを学習させて、そのモデルを回帰分析で求める。（そのモデルを数式で近似)
- [ ] A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problemの論文のソースコードを使う　このプログラムは、ポートフォリオを組んで仮想通貨のトレードをするプログラム（４倍の利益を達成）
- [ ] 一つのニューラルネットで、価格の予測をする出力と、何パーセント変わるかの出力２種類作る。
- [ ] 強化学習で求めた売る、買う、ノーポジの３択で、信用取引ができるように工夫する。
- [ ] 全仮想通貨の銘柄でトレーニングしたモデルとビットコインだけでオーバーフィトさせたモデルを、ブースティングでマージする。
- [x] ARIMAのseasonal_decomposeを使用して、価格を予測する


## 実験結果
まだ作成していません。。

## main.py
- 最後の20000個のデータをテストとして使用。
- 5エポック回して学習を行う。




# How to write Markdown
- [x] sample
- [ ] sample

* Item 1
* Item 2
  * Item 2a
  * Item 2b

1. Item 1
1. Item 2
1. Item 3
   1. Item 3a
   1. Item 3b
   
http://github.com - automatic!
[GitHub](http://github.com)

I think you should use an
`<addr>` element here instead.

```javascript
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```
