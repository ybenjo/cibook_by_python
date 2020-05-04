# CI Book by Python

## About

[安井翔太「効果検証入門～正しい比較のための因果推論／計量経済学の基礎」](https://gihyo.jp/book/2020/978-4-297-11117-5)のコードを Python で再実装したコードを格納したレポジトリです．

このコードでは，本文および[著者の R 実装](https://github.com/ghmagazine/cibook)の多くのコードを再実装していますが，以下の点は再実装していません．

- 各種図表の描画
- 4章の `causalimpact` を用いた実験
- 5章の Nonparametric RDD の実験

また，以下の点に注意が必要です．

- いくつかの値，特にセレクションバイアスを発生させたメールデータにおける数値が本文と一致しません
- `ch3_lalonde.py` における `3.4.4` の傾向スコアの推定において，本文では `re74` と `re75` の二乗の値を共変量に追加していますが，私が検証したところ傾向スコアの振る舞いが著しく悪化したため，それら二つは加えずに推定しました
- 傾向スコアマッチングには貪欲法を用いています．二部グラフの最大重みマッチングによる推定は失敗しました

## Requirements

- `sklearn`
    - 追加実験で `lightgbm`
- `pandas`
- `statsmodels`
- `tqdm`
- `pyreadr`

## data

`data/` には必要なデータを保存してください．

    wget http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
    wget https://users.nber.org/~rdehejia/data/cps_controls.dta
    wget https://users.nber.org/~rdehejia/data/cps_controls3.dta
    wget https://users.nber.org/~rdehejia/data/nsw_dw.dta

でダウンロード可能です．

`ch5_cigar.py` で用いるデータのみ， R の `Edcat` パッケージから `.RData` 形式で抽出する必要があります．

