# 強化学習を用いた仮想通貨の自動取引

# Install
```
conda install matplotlib
pip install poloniex
pip install chainer
pip install chainerrl
pip install cupy
```


## 新規性１

ハードフォークがある時点から何日前に発表されたか、何日後に発表されたかなどの情報を入力データとして与えることで、精度を上げる。方法は今のところ二種類思いついた。

ひとつ目の数式
```
【ハードフォーク前の場合】
100 ー（ハードフォーク当日 ー 現在）

【ハードフォーク後】
ー(100 ー (現在 ー ハードフォークがあった日))
```

ふたつ目の数式
```
２つの入力ノードを用意する。one hot encoding的なことをする。

【ハードフォーク前】
ひとつ目のノードに、100 ー　(ハードフォーク当日 ー 現在)
ふたつ目のノードに、0
【ハードフォーク後】
ひとつ目のノードに、0
ふたつ目のノードに、100 ー　(現在 ー ハードフォーク当日)

```

## 新規性２


