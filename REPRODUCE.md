# FramePack 再現手順書 (Reproduction Guide)

## 0. 概要

本ドキュメントは、World Model (FramePack) の環境構築から、提案手法（Step-Adaptive CFG）を用いたベンチマーク評価の実行までの一連の手順を記述します。
新規のLinux環境（GPU搭載）を想定しています。

## 1. 環境構築

### 1-1. リポジトリのクローン
```bash
git clone https://github.com/ken3nishio/WorldModel_Group04.git
cd WorldModel_Group04

# 実験用ブランチへの切り替え
git checkout experiment
```

### 1-2. Python仮想環境の作成
Python 3.10以上を推奨します。

```bash
python3 -m venv venv
source venv/bin/activate
```

### 1-3. 依存ライブラリのインストール
PyTorch, diffusers, および評価用ライブラリ（VBench, LPIPS, CLIP, Matplotlib等）をインストールします。

```bash
pip install -r requirements.txt
```

※ `requirements.txt` には `vbench`, `lpips`, `matplotlib`, `transformers` 等が含まれています。
※ VBenchのインストールには時間がかかる場合があります。

## 2. データの準備

### 2-1. 入力画像の配置
実験に使用する入力画像を配置します。
デフォルトでは、`experiments/benchmark_prompts.json` に記載されたパス（`experiments/inputs/` 以下）の画像を使用します。

```bash
mkdir -p experiments/inputs
# ここに評価対象の画像（例: 434605182-f3bc35cf-656a-4c9c-a83a-bbab24858b09.jpg）を配置してください
```

### 2-2. プロンプト設定の確認
`experiments/benchmark_prompts.json` を編集し、実行したいプロンプトと画像パス、および**ターゲットプロンプト**を指定します。

```json
[
  {
    "id": "Test_Disappearance",
    "category": "B_Disappearance",
    "prompt": "The man gradually disappears.",
    "target_prompt": "empty background, street scene only",  // 消失後の状態を記述（CLIP評価用）
    "input_image": "experiments/inputs/your_image.jpg"
  }
]
```

## 3. 実験の実行 (ベンチマーク)

以下のコマンドで、Baseline (`beta=0.0`) と Proposed (`beta=0.7`) の比較生成および自動評価を実行します。

```bash
python experiments/run_benchmark.py
```

### コマンド実行時の挙動:
1.  モデル（HunyuanVideo）のロード。
2.  `benchmark_prompts.json` の各ケースについて、2種類の動画（Baseline/Proposed）生成。
3.  生成された動画は `experiments/results/YYYYMMDD_HHMMSS/` ディレクトリに保存。
4.  **評価プロセス**の自動実行:
    - **VBench**: Subject Consistency, Dynamic Degree, Motion Smoothness 等の計測。
    - **LPIPS**: フレーム間の知覚的変化量（Dynamics）の計測。
    - **Frame-wise CLIP**: 各フレームと `target_prompt` との類似度推移を計測し、**グラフ画像**を生成。

## 4. 結果の確認

生成された `experiments/results/` 以下のディレクトリを確認します。

- **Generated Videos**: `...beta0.0...mp4` vs `...beta0.7...mp4` (動画ファイル)
- **Evaluation Report**: `eval_report_....md` (Markdownレポート)
- **Visualizations**: `visualizations/clip_trend_....png` (CLIPスコア推移グラフ)

### 評価のポイント（消失タスク）
- **Subject Consistency**: Baselineより**低くなっているか**？（＝被写体が消えた）
- **Dynamic Degree / LPIPS**: Baselineより**高くなっているか**？（＝変化が起きた）
- **CLIP Slope**: 右肩上がりのグラフになっているか？（＝ターゲット状態に近づいた）

## トラブルシューティング

- **VBench/LPIPSエラー**: ライブラリがインストールされていない場合、その指標はスキップされます。`pip install vbench lpips` を確認してください。
- **GPUメモリ不足**: `experiments/run_benchmark.py` 内の `preserved_memory_gb` 設定を調整するか、解像度/フレーム数を下げてください。
