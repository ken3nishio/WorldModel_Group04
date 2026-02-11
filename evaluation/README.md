# evaluation ディレクトリ

FramePack の改善手法（Step-Adaptive CFG / Temporal Blur 等）を定量評価するためのスクリプト群。

---

## ファイル一覧

| ファイル | 用途 | 動画生成 | 評価 | 推奨シーン |
|---|---|:---:|:---:|---|
| `evaluate_successful_case.py` | 既存動画の単体評価 | ❌ | ✅ | Gradio等で手動生成した動画をサクッと評価したい時 |
| `generate_and_evaluate.py` | 動画生成 + 評価の統合パイプライン | ✅ | ✅ | 条件を指定して生成から評価まで一気通貫で行いたい時 |
| **`compare_baseline_proposed.py`** | **Baseline vs Proposed 比較** | **✅** | **✅** | **デフォルト設定と成功条件を同時に生成→比較したい時** |
| `evaluate_disappearance.py` | 消失タスク専用のCLIP分析 | ❌ | ✅ | 複数動画の比較プロット作成時 |
| `run_vbench_custom.py` | VBench + LPIPS + CLIP の包括評価 | ❌ | ✅ | メタデータJSONを使ったバッチ評価時 |

---

## 使い方

### 1. 既存動画の評価（`evaluate_successful_case.py`）

手動（Gradio UI等）で生成した動画を事後評価するスクリプト。
`experiments/results/main.txt` に記載の成功条件がデフォルトで設定済み。

```bash
# デフォルト設定で実行（main.txtの成功条件）
python evaluation/evaluate_successful_case.py --skip_vbench

# カスタム動画を評価
python evaluation/evaluate_successful_case.py \
    --video_path path/to/video.mp4 \
    --prompt "A man walks away" \
    --object_prompt "a man" \
    --empty_prompt "empty street" \
    --skip_vbench

# CPU で実行
python evaluation/evaluate_successful_case.py --device cpu --skip_vbench --skip_lpips
```

**出力**（`evaluation/results/eval_<timestamp>/` に生成）:
- `evaluation_dashboard.png` — 4パネルの可視化ダッシュボード
- `evaluation_report.md` — Markdown形式のレポート
- `evaluation_results.json` — 生データ（JSON）

### 2. 動画生成 + 評価（`generate_and_evaluate.py`）

FramePackモデルで動画を生成し、そのまま評価まで実行するフルパイプライン。

```bash
# デフォルト設定（main.txtの成功条件）で生成＋評価
python evaluation/generate_and_evaluate.py \
    --input_image experiments/inputs/image.jpg \
    --skip_vbench

# パラメータをカスタム指定
python evaluation/generate_and_evaluate.py \
    --input_image experiments/inputs/image.jpg \
    --prompt "A man walks away and disappears" \
    --beta -0.5 \
    --seed 31337 \
    --steps 25 \
    --skip_vbench

# JSON設定ファイルから実行
python evaluation/generate_and_evaluate.py \
    --input_image experiments/inputs/image.jpg \
    --config experiments/eval_config.json

# 生成だけ（評価スキップ）
python evaluation/generate_and_evaluate.py \
    --input_image experiments/inputs/image.jpg \
    --skip_eval
```

**出力**（`evaluation/results/run_<timestamp>/` に生成）:
- `generated_beta_*.mp4` — 生成動画
- `config.json` — 実験設定
- `evaluation_dashboard.png` — 可視化
- `evaluation_report.md` — レポート
- `evaluation_results.json` — 生データ
- `pipeline.log` — 実行ログ

### 3. Baseline vs Proposed 比較（`compare_baseline_proposed.py`）★推奨

デフォルト設定（β=0）と成功条件（β=-0.5, blur=1.3）を**1コマンドで生成→評価→比較**するスクリプト。
モデルは1度だけロードして使い回すため、GPU時間を節約できる。

```bash
# 新規生成して比較（推奨）
python evaluation/compare_baseline_proposed.py \
    --input_image experiments/inputs/434605182-f3bc35cf-656a-4c9c-a83a-bbab24858b09.jpg

# 既存動画で評価だけ比較（生成スキップ）
python evaluation/compare_baseline_proposed.py \
    --baseline_video experiments/results/beta_0_dancer_performing_backflip.mp4 \
    --proposed_video experiments/results/successful_data_blur_1.3_beta_-0.5.mp4

# カスタムbeta/blurで比較
python evaluation/compare_baseline_proposed.py \
    --input_image experiments/inputs/image.jpg \
    --proposed_beta -0.7 --proposed_blur 1.5
```

**出力**（`evaluation/comparison_results/comparison_<timestamp>/` に生成）:
- `baseline/video.mp4` — ベースライン動画
- `proposed/video.mp4` — 提案手法動画
- `comparison_table.csv` — 比較表
- `comparison_chart.png` — 比較チャート
- `comparison_report.md` — Markdownレポート
- `comparison_results.json` — 生データ

### 4. 消失タスクの比較分析（`evaluate_disappearance.py`）

複数動画のObject/Empty確率推移を比較プロットする。

```bash
python evaluation/evaluate_disappearance.py \
    --video_paths video_baseline.mp4 video_proposed.mp4 \
    --labels "Baseline" "Proposed" \
    --object_prompt "a man walking" \
    --empty_prompt "empty street background" \
    --output_plot comparison.png
```

### 4. バッチ評価（`run_vbench_custom.py`）

メタデータJSONに基づき、複数動画を一括で VBench + LPIPS + CLIP 評価する。
通常は `experiments/run_benchmark.py` から自動呼び出しされる。

```bash
python evaluation/run_vbench_custom.py \
    --video_dir experiments/results/<timestamp> \
    --metadata experiments/results/<timestamp>/metadata.json \
    --output_dir evaluation/results
```

---

## 評価指標

### 使用している指標

| 指標 | 出典 | 消失タスクでの解釈 |
|---|---|---|
| **CLIP Object/Empty Prob** | OpenAI CLIP | 最終フレームで Empty > Object なら消失成功 |
| **Frame-wise CLIP Score** | OpenAI CLIP | 正のslope = ターゲット状態に近づいている |
| **LPIPS** | Zhang et al. | 高値 = フレーム間の変化が大きい（動的） |
| **VBench Dynamic Degree** | CVPR 2024 | 高値 = 動画内の動きが大きい |
| **VBench Subject Consistency** | CVPR 2024 | **消失タスクでは低い方が良い**（逆説的） |
| **VBench Motion Smoothness** | CVPR 2024 | 高値 = 時間的な滑らかさ |

### 成功基準（消失タスク）

| 基準 | 条件 |
|---|---|
| 最終フレーム消失 | Empty Prob > Object Prob |
| Object確率減少 | drop > 0.05 |
| CLIP Slope 正 | slope > 0 |
| 動的変化あり | LPIPS mean > 0.02 |

---

## 関連ファイル

- `experiments/run_benchmark.py` — 動画生成＋バッチ評価のフルベンチマーク
- `experiments/benchmark_prompts.json` — ベンチマーク用プロンプト定義
- `experiments/results/main.txt` — 成功条件の記録
- `docs/evaluation_metrics.md` — 評価指標の選定理由
- `docs/evaluation_selection.md` — 評価フレームワーク選定ドキュメント

---

## 依存関係

```
# 必須
torch
transformers
opencv-python
numpy
matplotlib
Pillow

# オプション（指定メトリクスのみ）
lpips          # LPIPS 評価に必要
vbench         # VBench 評価に必要
```

インストール:
```bash
pip install lpips
pip install vbench
```

---

## アーキテクチャ図

```
┌─────────────────────────────────────────────────────────┐
│                   実験フロー                             │
│                                                         │
│  ┌─────────────┐     ┌──────────────────────────────┐   │
│  │ Gradio UI   │────→│ evaluate_successful_case.py   │   │
│  │ (手動生成)   │     │ （評価のみ / 軽量）            │   │
│  └─────────────┘     └──────────────────────────────┘   │
│                                                         │
│  ┌─────────────┐     ┌──────────────────────────────┐   │
│  │ CLI引数/JSON │────→│ generate_and_evaluate.py      │   │
│  │ (条件指定)   │     │ （生成 + 評価 / 統合版）       │   │
│  └─────────────┘     └──────────────────────────────┘   │
│                                                         │
│  ┌─────────────┐     ┌──────────────────────────────┐   │
│  │ benchmark   │────→│ run_benchmark.py +             │   │
│  │ _prompts.json│     │ run_vbench_custom.py           │   │
│  │ (バッチ定義) │     │ （バッチ生成 + 評価）           │   │
│  └─────────────┘     └──────────────────────────────┘   │
│                                                         │
│  共通評価ライブラリ:                                      │
│  ├── evaluate_disappearance.py (CLIP消失分析)            │
│  └── run_vbench_custom.py (VBench/LPIPS/CLIP包括)        │
└─────────────────────────────────────────────────────────┘
```
