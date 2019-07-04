## go-nlp

Golangで自然言語処理を行う。

### 前処理

1. テキストから余計な文字列を削除
1. Stop Words("I", "like", etc)の削除
1. 言語の標準化

### Vectorize

* bag of words
* tfidf

### ベクトル間距離

* ユークリッド距離
* コサイン距離

### USAGE

```
go build
./nlp
```
