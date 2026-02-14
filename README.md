# Adaptive Control of Temporal Inertia in Large-Scale Video Generation Models
**(Static Death Mitigation Project)**

大規模動画生成モデルにおける「Static Death（静的死）」問題を解決するための、推論時介入手法（Relaxation & Decay）の研究開発リポジトリです。本プロジェクトは、HunyuanVideo等の基盤モデルに対し、追加学習なしで「バク転」等の動的アクション生成を実現しました。

## 研究概要

### 背景: Static Deathの課題
大規模動画生成モデルは、学習データの大規模化により高い安定性と画質を獲得しましたが、副作用として過去の文脈（初期フレームの状態）に過剰に固執する「Static Death」という現象を引き起こしています。たとえば、「男性がバク転をする」と指示しても、モデルは初期の「立ち姿」を維持し続け、決して回転しようとしません。これは、モデルが安定性を重視しすぎるあまり、状態変化のためのエネルギー障壁を乗り越えられなくなっているためです。

### 提案手法: Relaxation & Temporal Unlearning
我々は、エネルギー地形モデルに基づき、モデルが初期のポテンシャルの谷から脱出できないことが原因であると突き止めました。従来の「強制的な力（Impulse/Boost）」を与える手法は、逆に映像をカオス化させる（Spinning Poi現象）ことが判明しました。
そこで本研究では、以下の3つの要素を組み合わせた「Relaxation（緩和）」戦略を提案しました：

1.  **Adaptive CFG (Relaxation)**: 初期のCFGスケールを $\beta$ の割合で減衰させ、モデルの探索自由度を高める。
    *   初期段階での拘束を解くことで、モデルが「重い腰を上げる」手助けをする。
2.  **Decay Control**: 緩和効果を $p$ の指数で減衰させ、後半の整合性を保つ。
    *   いつまでも緩めていると形が崩れるため、動き出した後は速やかに通常の拘束に戻す。
3.  **Temporal Unlearning**: 直前フレームへのAttention Key/Valueをガウシアンブラー ($\sigma_{blur}$) で時間的に拡散させる。
    *   直前の自分自身の姿（立ち姿）を意図的に「忘れさせる」ことで、新しい姿勢への上書きを容易にする。

---

## 主な成果

以下の表は、提案手法導入前後での VideoMAEスコア（アクション認識精度）の比較です。

| Method | Beta ($\beta$) | Power ($p$) | VideoMAE Score | Qualitative Result |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Pure)** | 0.0 | - | 0.073 | **Breakdancing** (Failed) |
| **Forced (Boost)** | -1.0 | 0.7 | 0.007 | **Spinning Poi** (Chaos) |
| **Ours (Best)** | **0.75** | **0.7** | **0.394** | **Somersaulting** (Success) |

推論時のパラメータ調整のみ（Training-free）で、これまで不可能だった動的な「バク転」生成に成功し、VideoMAEスコアをBaseline比で**約5.4倍**に向上させました。

---

## 環境構築

### 必要要件
*   Python 3.10+
*   PyTorch 2.0+ (CUDA enabled)
*   Diffusers, Transformers

### インストール
```bash
# Clone the repository
git clone https://github.com/ken3nishio/WorldModel_Group04.git
cd WorldModel_Group04

# Install dependencies
pip install -r requirements.txt
```

---

## 使い方

### 1. 推論実行 (Gradio Demo)
ブラウザ上でパラメータを調整しながら生成実験を行えます。

```bash
python demo_gradio_f1.py
```
*   ブラウザで `http://localhost:7860` にアクセス。
*   Prompt: "A man performs a backflip" 等を入力。
*   Advanced Settingsから `Adaptive CFG` を有効化。
*   推奨設定: `Beta=0.75`, `Power=0.7`, `CFG Min=1.0`.

### 2. バッチ実験実行
複数のパラメータ設定で自動的に生成・評価を行います。

```bash
python experiments/run_experiment_batch.py
```
結果は `experiments/results/` ディレクトリにMarkdown形式のレポートとして保存されます。

---

## ディレクトリ構造

*   `docs/`: ドキュメント群
    *   `paper/`: **JSAI投稿用論文ファイル** (`main.tex`, `experiment_results.tex`)。本研究の理論的詳細はこちらを参照。
    *   `experiments/`: 実験計画や仮説検証のメモ。
*   `experiments/`: 実験関連
    *   `run_experiment_batch.py`: バッチ実験実行スクリプト。
    *   `results/`: 実験結果（スコア、動画サムネイル、レポート）。
*   `diffusers_helper/`: モデルラッパー
    *   `k_diffusion/wrapper.py`: **Adaptive CFG (Relaxation)** の実装箇所。
    *   `pipelines/`: カスタムパイプラインの実装。
*   `evaluation/`: 評価スクリプト
    *   `evaluate_all_metrics.py`: VideoMAE, CLIP, LPIPSスコアの計算。
*   `demo_gradio_f1.py`: メインのGUIアプリケーション。

---

## Citations
本研究は以下の技術をベースにしています。

*   **HunyuanVideo**: Foundation Model used as the backbone.
*   **FramePack**: Memory optimization technique for long video generation.
*   **FreeInit / ALG**: Inspiration for inference-time intervention.
