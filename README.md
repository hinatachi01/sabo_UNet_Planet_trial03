# sabo_UNet_Planet_trial03

## 概要 / Overview
複数の斜面崩壊事例を用いてU-Netモデルを学習し、**日本全域における斜面崩壊地の分布図作成**を目的とする。<br>

| カテゴリ値 | カテゴリ名 |
|-----|-----|
| 0 | 崩壊地 |
| 1 | 非崩壊地 |
  
---
## 試行状況 / Data Used

<試行ごとの入力データ>
| 番号 | 特徴量 | 入力データ説明 | 使用モデル | 学習枚数 | 学習ログ | 結果 | 実行日 |
|-----|-----|-----|-----|-----|-----|-----|-----| 
| 01 | Planet 4bands (/10,000) | ・1万で割り正規化 | model A | train 679 <br> val 102 <br> test 2 | |  ・崩壊地の輪郭を捉えられている <br> ・閾値の調節は必要 | April 17 |
| 02 | Planet 4bands (/10,000) | 上記に加え <br> ・値0-1の確認✅ <br> ・崩壊地と教えている所が本当に崩壊地か確認 (未) (基準が難しい) | model A |train <br> val <br> test 2| | | April 18 |

**<評価>**
| 番号 | 崩壊地判断の確率閾値 | filename | precision (誤抽出) | recall (見逃し) | f1 | iou |
|-----|-----|-----|-----|-----|-----|-----|
| 01 | <0.59 | tile_gofukuya_01792_01024 |	0.9402 | 0.8490 | 0.8923 | 0.8056 |
| 〃 | 〃 | tile_gofukuya_01920_01024 |	0.8862 | 0.8757 | 0.8809 | 0.7872 |
| 02 | | | | | | |

---
## モデル

- モデルA
  - モデル構造：縮小・拡大パスを持つ簡易UNet（3層エンコーダ＋3層デコーダ）
  - 入力データ：128×128ピクセル、4バンド（RGB + NIR）
  - 出力形式：128×128ピクセル、2クラス（softmax出力）
  - マスク前処理：崩壊地＝1、非崩壊地＝0としてone-hotエンコード
  - 損失関数：categorical_crossentropy（クラス重みあり）
  - 最適化手法：Adam（学習率1e-5）
  - 重み付け：崩壊地=5.0、非崩壊地=1.0のsample_weightを使用
  - 学習設定：エポック数10、バッチサイズ16、EarlyStopping適用（patience=5）

---

## 元データ / Original Data

### 特徴量作成用
- **衛星画像**：PlanetScope（3m解像度）
- **標高データ**：基盤地図情報 数値標高モデル [GSI 地理院地図](https://service.gsi.go.jp/kiban/)

### マスク作成用
- **崩壊地判読図**：国土地理院 防災・災害対応ページより取得 [GSI 防災・災害対応](https://www.gsi.go.jp/bousai.html)

### 使用事例
| 年度 | 災害名 | 地域 |
|------|--------|------|
| H30  | 北海道胆振東部地震   | 北海道 胆振地方 |
| R01  | 台風19号豪雨         | 宮城県 五福谷川 |
| R06  | 能登半島地震         | 石川県 能登地方 |
| R06  | 能登半島豪雨         | 石川県 能登地方 |

---

## ディレクトリ構成 / Dataset Structure

sabo_UNet_Planet_trial03/

- original_data/
- tiles/
  - features/
  - masks/
  - タイル作成 (128 pixel × 128 pixel) のpythonファイル×2
  - タイル作成後に選別するpythonファイル×3
- dataset/
  - train/     
    - features/
    - masks/
  - val/               
    - features/
    - masks/
  - test/
    - features/
    - masks/
- training.py ... U-Netモデルの定義、学習、モデル出力
- validation.py ... 未学習画像によるモデルの検証
- model.h5 ... 出力されたモデル
