# FramePack Evaluation Toolkit

## 概要
FramePackの課題である「Static Bias（変化の抑制）」と「Consistency（一貫性）」のトレードオフを定量評価するためのツールキットです。
VBenchの指標と、自作の指標（STF, GAS）を組み合わせて、プロンプトのカテゴリ（通常/消失）ごとに最適な評価を行います。

## ファイル構成
- `metrics.py`: 自作の評価指標（STF, GAS, DF）の実装。
- `run_vbench_custom.py`: 評価実行のメインスクリプト。動画とプロンプトを読み込み、スコアを計算・集計してレポートを出力します。

## 依存関係
以下のライブラリが必要です。
```bash
pip install vbench opencv-python pandas tqdm
```
※ `vbench` のインストールは時間がかかる場合があります。

## 使い方

### 1. 評価用データの準備 (Metadata)
評価したい動画が入ったディレクトリと、ファイル名・プロンプトの対応表（JSON）を用意してください。

**metadata.json の形式例**:
```json
[
  {
    "filename": "sample_001.mp4",
    "prompt": "A girl smiling in the park"
  },
  {
    "filename": "sample_002.mp4",
    "prompt": "The snowman melts and disappears"
  }
]
```

### 2. 評価の実行
以下のコマンドで評価を実行します。

```bash
python evaluation/run_vbench_custom.py \
  --video_dir /path/to/videos \
  --metadata /path/to/metadata.json \
  --output_dir evaluation/results_2024
```

### 3. 結果の確認
`evaluation/results_2024` ディレクトリに以下のファイルが生成されます。
- `evaluation_results_YYYYMMDD.csv`: 動画ごとの全スコア（Rawデータ）
- `evaluation_report_YYYYMMDD.md`: カテゴリごとの平均スコアと分析レポート

## 評価カテゴリについて
スクリプトはプロンプトのキーワード（`disappear`, `melt` 等）に基づいて、自動的にカテゴリを分類します。

- **Category A (Standard)**: 通常の動作。`Subject Consistency` が高いことが望ましい。
- **Category B (Disappearance)**: 消失・変形。`Overall Consistency` が低くても、`STF` (指示適合性) が高いことが望ましい。
