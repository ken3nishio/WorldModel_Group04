# FramePack 再現手順書 (Reproduction Guide)

## 0. 概要

本ドキュメントは、World Model (FramePack) の環境構築から、提案手法（Step-Adaptive CFG）を用いたベンチマーク評価の実行までの一連の手順を記述します。

## 1. Google Colab で実行する場合 (推奨)

Colab等のノートブック環境で実行する場合は、以下のセル順序で実行・セットアップしてください。

### Step 1: クローンとセットアップ
```python
# リポジトリのクローン
!git clone https://github.com/ken3nishio/WorldModel_Group04.git
%cd WorldModel_Group04
!git checkout experiment

# 依存ライブラリのインストール (重要)
!pip install -r requirements.txt
```

### Step 2: VBenchのインストール
※ VBenchを使用して評価を行う場合のみ必要です（推奨）。時間がかかります。
```python
!pip install git+https://github.com/Vchitect/VBench.git
```

### Step 3: 実験の実行
```python
# 入力画像ディレクトリの作成（必要に応じて）
!mkdir -p experiments/inputs

# ベンチマーク実行
!python experiments/run_benchmark.py
```

## 2. ローカルLinux環境で実行する場合

### 2-1. 環境構築
Python 3.10以上を推奨します。

```bash
git clone https://github.com/ken3nishio/WorldModel_Group04.git
cd WorldModel_Group04
git checkout experiment

python3 -m venv venv
source venv/bin/activate

# 必須ライブラリのインストール
pip install -r requirements.txt

# VBenchのインストール (任意)
pip install git+https://github.com/Vchitect/VBench.git
```

### 2-2. 実験の実行
```bash
python experiments/run_benchmark.py
```

## 3. データの準備と確認
（以下、元の内容と同様）
...
