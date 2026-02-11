# 研究概要：Frequency-Adaptive Hysteresis Control (FAHC)
# FramePackにおける意味的慣性の克服

---

## 1. 一言で言うと

> **FramePack（長尺動画生成モデル）が「消失」や「大動作」のプロンプトを無視する問題を、
> デノイジング初期段階のCFGスケールを動的に制御する推論時テクニック（FAHC / Step-Adaptive CFG）
> で解決し、一貫性を維持しつつダイナミクスを向上させた。**

---

## 2. 研究の背景

### 2.1 長尺動画生成の課題
- 近年の動画生成モデル（Sora, HunyuanVideo, Wan等）は数秒の生成に限定
- **FramePack** は「階層的コンテキスト圧縮」で無限の長尺生成を実現した画期的手法
- しかし、そのメモリ保持の強力さが**副作用**を生んでいる

### 2.2 問題定義：「意味的慣性（Semantic Inertia）」
物理学の慣性のアナロジー：
- **質量（＝過去の記憶量）が大きいほど、運動状態（生成映像）の変化に大きな力が必要**
- 具体的現象：
  - 「人が消える」と指示しても、消えない（被写体恒常性バイアス）
  - 「バク転する」と指示しても、その場で震えるだけ（静的バイアス）
  - シーン遷移時にゴースト（前シーンの残像）が発生

### 2.3 根本原因（構造分析）
FramePackの3つのメカニズムが「変化」を阻害：

| メカニズム | 設計意図 | 副作用 |
|---|---|---|
| 階層的コンテキスト圧縮 | 計算効率の確保 | 低周波の「存在情報」が圧縮後も残存 |
| アンチドリフティング | エラー蓄積の防止 | 意図的な急変化を「エラー」として抑制 |
| KVキャッシュ / Attention Sink | 長期一貫性の維持 | 過去の状態に対するAttentionが支配的 |

加えて、基盤モデル（HunyuanVideo）の**学習データバイアス**（静的シーンが多い）と**条件画像の漏洩**（1フレーム目への過剰適合）が問題を増幅。

---

## 3. 先行研究

3つの主要アプローチを調査：

| 手法 | 論文 | アプローチ | FramePackへの適用 |
|---|---|---|---|
| **ALG** (Adaptive Low-Pass Guidance) | 2025 | 初期段階で条件画像にローパスフィルタ → Dynamic Degree +36% | **直接適用可能** → 本研究の着想元 |
| **MotionRAG** | 2024-2025 | 外部ビデオDBから動作パターンを検索・転写 | 有望だが実装コスト大 |
| **Rolling Forcing / SVI** | 2024-2025 | 同時デノイジング＋エラー再利用学習 | 再学習が必要でコスト大 |

**本研究の着想**：ALGの「初期段階で条件の影響を弱める」というアイデアを、CFGスケールの動的調整として再解釈。

---

## 4. 提案手法：Step-Adaptive CFG / FAHC

### 4.1 核心アイデア
拡散モデルの性質を利用：
- **初期段階**（σ≈1）：全体構図（低周波）を決定 → CFGを**弱く**して構造変化を許容
- **終盤段階**（σ≈0）：詳細（高周波）を描画 → CFGを**強く**してディテールを維持

### 4.2 数式

$$\text{cfg}(\sigma) = \text{cfg}_{min} + (\text{cfg}_{base} - \text{cfg}_{min}) \times (1 - \beta \times \sigma)$$

| パラメータ | 意味 | 推奨値 |
|---|---|---|
| cfg_base | 元のCFGスケール | 10.0 |
| cfg_min | 初期段階の最小CFG | 1.0 |
| **β (beta)** | **制御の核心パラメータ** | **-0.5（消失タスク）** |
| σ | ノイズレベル (0〜1) | 自動計算 |

### 4.3 Negative Beta の発見
初期実装では β>0（初期CFG低下）を想定 → **消失タスクでは効果なし**

**原因**：初期段階でCFGを下げると、入力画像への依存がむしろ**強まる**

**解決**：β<0（**Negative Beta / Initial Boost**）
- 初期段階でCFGを**上げる** → プロンプトの力で入力画像の構造を**破壊**
- 終盤でCFGを下げる → アーティファクトを抑制

これは**当初の仮説（ALG的アプローチ）の反転**であり、実験から得られた重要な知見。

### 4.4 Temporal Blur（補助手法）
過去フレームの latent に時間方向のガウシアンブラーを適用し、高周波成分を除去：
$$\text{clean\_latent}'_k = \mathcal{G}_{\sigma(k)} * \text{clean\_latent}_k$$
- σ=1.3 が最適値として発見された

---

## 5. 実装

### 5.1 変更ファイル（推論時のみ、再学習不要）
| ファイル | 変更内容 |
|---|---|
| `diffusers_helper/k_diffusion/wrapper.py` | `adaptive_cfg_scale()` 関数追加 |
| `diffusers_helper/pipelines/k_diffusion_hunyuan.py` | パラメータ伝播 |
| `demo_gradio_f1.py` | UIスライダー追加 |

### 5.2 最小変更で最大効果
- アーキテクチャ変更 **不要**
- 再学習 **不要**
- 推論時の数行のコード変更のみ

---

## 6. 実験結果

### 6.1 成功条件
| パラメータ | 値 |
|---|---|
| プロンプト | "Static background. A man walks forward and out of view. Empty background remains." |
| Beta | **-0.5** |
| Temporal Blur σ | **1.3** |
| Steps | 25 |
| CFG Scale | 6 |
| Distilled CFG Scale | 10 |
| Seed | 31337 |

**結果**：`successful_data_blur_1.3_beta_-0.5.mp4` — 人物が歩き去り、背景のみが残る消失タスクに成功。

### 6.2 失敗条件との比較
| Beta値 | 結果 |
|---|---|
| 0.0（ベースライン） | 消失せず、その場で震えるのみ |
| 0.7（正の適応） | ベースラインと差なし（初期CFG低下は逆効果） |
| **-0.5（負の適応）** | **消失成功** |

---

## 7. 評価手法

### 7.1 方針
- **既存の権威ある指標のみ使用**（自作指標は妥当性検証コストが高いため排除）
- VBench (CVPR 2024 Highlight) を中心に据える

### 7.2 使用指標

| 指標 | 出典 | 消失タスクでの解釈 |
|---|---|---|
| **CLIP Object/Empty Prob** | OpenAI CLIP | 最終フレームで Empty > Object なら成功 |
| **Frame-wise CLIP Score** | OpenAI CLIP | 正の slope = ターゲットに近づいている |
| **LPIPS** | Zhang et al. (CVPR 2018) | 高値 = 動的変化あり |
| **VBench Dynamic Degree** | CVPR 2024 | 動きの量（向上を期待） |
| **VBench Subject Consistency** | CVPR 2024 | **消失タスクでは低下が正解**（逆説的解釈） |
| **VBench Motion Smoothness** | CVPR 2024 | ガードレール指標：画質崩壊していないか |

### 7.3 カテゴリ別評価プロトコル

| カテゴリ | 成功の定義 |
|---|---|
| **A. 通常動作** | Dynamic Degree ↑ かつ Subject Consistency 維持 |
| **B. 消失・変形** | Dynamic Degree ↑ かつ Subject Consistency **低下**（＝消えている） かつ Motion Smoothness 維持 |
| **C. 大規模変化** | Dynamic Degree ↑ かつ Motion Smoothness 維持 |

---

## 8. 論文構成案

```
Title: Dynamic Stability: Overcoming Semantic Inertia in World Models
       via Frequency-Adaptive Hysteresis Control

1. Introduction
   - 世界モデルにおける安定性と可塑性のジレンマ
   - 意味的慣性（Semantic Inertia）の定義
   - 提案手法の概要

2. Related Work
   - 長尺動画生成（FramePack, StreamingT2V）
   - 制御可能な動画生成（ControlNet, ALG）
   - 拡散モデルの周波数特性分析

3. Problem Analysis
   - FramePackの構造的制約（圧縮・アンチドリフト・KV）
   - 条件画像の漏洩と静的バイアス
   - 意味的慣性のメカニズム

4. Method: FAHC / Step-Adaptive CFG
   - デノイジングステップ依存のCFG制御
   - Negative Beta（Initial Boost）の発見
   - Temporal Blur との併用

5. Experiments
   - 消失タスク / 大動作タスク / 長期一貫性
   - ベースライン (β=0) vs 提案手法 (β=-0.5)
   - アブレーション: βの値 / blur σ の値

6. Evaluation
   - VBench + LPIPS + Frame-wise CLIP Score
   - カテゴリ別評価プロトコル
   - 定性的分析（フレーム可視化）

7. Discussion
   - 世界モデルにおける「選択的忘却」の重要性
   - エージェント学習への影響
   - 限界と今後の課題

8. Conclusion
```

---

## 9. 今後の課題

### 短期（本論文内）
- [ ] 成功動画の定量評価を完了する
- [ ] 複数プロンプトでの比較実験（A/B/Cカテゴリ）
- [ ] アブレーション実験（β, blur σ の系統的探索）

### 中期（拡張）
- 周波数適応型コンテキスト圧縮
- セマンティック・ゲート付きKVキャッシュ
- シーン認識型圧縮スケジュール

### 長期（新テーマ）
- デュアルストリーム・メモリ（永続記憶 + 揮発記憶）
- セマンティック・コンテキスト・フラッシング
- 遷移計画を伴う逆方向アンチドリフト

---

## 10. ドキュメント構成

```
docs/
├── main.txt                        ← ドキュメントマップ（このファイルへのポインタ）
├── research_overview.md             ← ★ 本ファイル（研究全体の概要）
│
├── paper/                           ← 論文ドラフト
│   └── research_paper_draft.md
│
├── problem/                         ← 問題分析
│   ├── framepack_problem.txt        ← FramePackの構造的課題
│   ├── research_gap.txt             ← リサーチギャップの体系的整理
│   └── investigation_report_dynamics.md  ← Beta正値の失敗分析
│
├── related_work/                    ← 先行研究
│   └── related_work.txt             ← ALG, MotionRAG, SVI
│
├── method/                          ← 提案手法
│   ├── framepack_research_proposal.txt  ← 研究テーマ3案
│   ├── improvement_candidates.md        ← 改善候補一覧
│   └── step_adaptive_cfg_implementation.md  ← Step-Adaptive CFG詳細
│
├── experiments/                     ← 実験計画
│   ├── experiment_plan_human_perception.md  ← 仮説検証実験
│   └── ui_experiment_guide.md              ← UI実験手順
│
└── evaluation/                      ← 評価指標
    ├── evaluation_selection.md           ← VBench選定
    ├── evaluation_metrics.md             ← 指標の使い方
    ├── evaluation_metric_selection_rationale.md  ← カテゴリ別プロトコル
    ├── metric_revaluation.md             ← 自作指標排除方針
    └── standard_metrics_rationale.md     ← 総合評価ロジック
```
