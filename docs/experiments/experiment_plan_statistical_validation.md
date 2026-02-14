# 統計的信頼性を担保した実験計画書：Frequency-Adaptive Hysteresis Control (FAHC)

## 1. 実験の目的と背景
本実験の目的は、提案手法である **Step-Adaptive CFG (FAHC)** が、長尺動画生成モデル（FramePack）における「意味的慣性（Semantic Inertia）」の問題を解決し、特に「消失」「変形」といった高難易度タスクにおいて有意な改善をもたらすことを統計的に実証することである。

これまでの予備実験では定性的な成功例（"Disappearance" タスクでの成功）が得られているが、少数のサンプルに基づく評価では「偶然の成功（Cherry-picking）」である可能性を排除できない。信頼性の高い論文として発表するためには、**十分なサンプルサイズ**、**適切な対照実験**、および**統計的仮説検定**に基づいた客観的な評価が不可欠である。

## 2. 統計的実験デザイン

### 2.1 比較群の設定 (Comparison Groups)
実験は「対応のある設計（Paired Design）」を採用する。同一のプロンプト、同一のシード、同一の入力画像に対して、以下の2条件で生成を行い、その差分を評価する。これにより、プロンプトごとの難易度のばらつきを相殺し、純粋な手法間の差を検出力を高めて検証できる。

| 群名 | 設定 | 役割 |
| :--- | :--- | :--- |
| **Baseline Group** | Beta = 0.0 (FramePack Default) | 既存手法の性能基準 |
| **Proposed Group** | Beta = -0.5 (FAHC) | 提案手法の効果検証 |

*注: 探索的実験として Beta = -0.8 や +0.5 も実施するが、主たる検定は上記2群間で行う。*

### 2.2 対象データセットとサンプルサイズ (Population & Sample Size)
VBench (CVPR 2024) のプロンプトセットを母集団とし、タスク特性に基づいて層別化サンプリングを行う。

**サンプルサイズ設計**:
- 一般的なt検定において、効果量 (Cohen's d) が中程度 (0.5)、検出力 ($1-\beta$) が 0.8、有意水準 ($\alpha$) が 0.05 の場合、必要なサンプル数は約34ペアとなる。
- よって、各カテゴリごとに **N=30〜50** のプロンプトを使用する。

**カテゴリ分類**:
1.  **Category A: Standard Motion (通常動作)**
    *   内容: 歩行、笑顔、日常動作。
    *   目的: 「副作用がないこと」の確認。
    *   目標数: N=30
2.  **Category B: Disappearance/Transformation (消失・変形)**
    *   内容: 消える、溶ける、爆発する、変化する。
    *   目的: 「改善効果」の検証（本研究の主眼）。
    *   目標数: N=30
3.  **Category C: Large Scene Change (大規模変化)**
    *   内容: ズームイン/アウト、シーン遷移。
    *   目的: 「ダイナミクス向上」の検証。
    *   目標数: N=30

計 90 ペア（180動画）の生成を行う。

### 2.3 評価指標 (Metrics)

| 指標カテゴリ | 具体的な指標 | 測定内容 |
| :--- | :--- | :--- |
| **主要評価項目 (Primary Endpoint)** | **VBench Dynamic Degree** | 動画内の動きの大きさ。消失タスクでは上昇・変化を期待。 |
| | **CLIP Text-Image Similarity** (Start/End) | 最終フレームがプロンプト（"Empty background"等）と一致しているか。 |
| **副次評価項目 (Secondary Endpoint)** | **VBench Subject Consistency** | 被写体の一貫性。Category Aでは維持、Category Bでは**低下**（消えるため）を期待。 |
| **安全性評価項目 (Safety Endpoint)** | **VBench Motion Smoothness** | 動画の質の崩壊（フリッカーなど）がないか。 |

## 3. 仮説と統計手法

### 3.1 仮説の定式化

**仮説1: 消失タスクにおけるダイナミクスの向上**
*   対象: Category B
*   帰無仮説 ($H_0$): $\mu_{prop} - \mu_{base} \le 0$ (提案手法はDynamic Degreeを向上させない)
*   対立仮説 ($H_1$): $\mu_{prop} - \mu_{base} > 0$ (提案手法はDynamic Degreeを有意に向上させる)
*   検定手法: **片側 対応のある t検定 (Paired t-test)**（正規性が棄却される場合はウィルコクソンの符号順位検定）

**仮説2: 通常タスクにおける一貫性の維持（非劣性）**
*   対象: Category A
*   帰無仮説 ($H_0$): $\mu_{prop} - \mu_{base} < -\delta$ (提案手法は一貫性を許容範囲 $\delta$ を超えて悪化させる)
*   対立仮説 ($H_1$): $\mu_{prop} - \mu_{base} \ge -\delta$ (提案手法の一貫性低下は許容範囲内である)
*   検定手法: **非劣性検定 (Non-inferiority test)**、あるいは簡易的に信頼区間の下限確認。

**仮説3: 画質の維持**
*   対象: 全カテゴリ
*   帰無仮説 ($H_0$): $\mu_{prop} = \mu_{base}$ (Motion Smoothnessに差はない)
*   検定手法: **両側 対応のある t検定**

### 3.2 多重性の考慮
3つのカテゴリと複数の指標を扱うため、多重比較の問題が発生しうる。本研究では、Category BにおけるDynamic Degreeの向上を**主要結果 (Main Result)** と位置づけ、他は探索的結果とする。ただし、p値の解釈においては、ボンフェローニ補正等を適宜考慮する（例: 有意水準を 0.05/3 = 0.017 とみなすなど）。

## 4. 実験手順 (Procedure)

1.  **プロンプト抽出**:
    *   VBench full prompt list から、キーワード ("disappear", "vanish", "fade", "melt", etc.) に基づき Category B を抽出。
    *   残りのリストからランダムに Category A, C を抽出。
2.  **生成実験実施**:
    *   同一GPU環境にて、BaselineとProposedの順序をランダム化しつつ生成（順序効果の排除）。
    *   全動画を `experiments/results/[timestamp]/` に保存。
3.  **指標算出**:
    *   VBench 公式スクリプトを用いてスコア化。
4.  **統計解析**:
    *   Python (Scipy/Statsmodels) を用いて検定を実施。
    *   結果を 箱ひげ図 (Boxplot) および 信頼区間プロット (CI Plot) で可視化。

## 5. 期待される結果の記載イメージ（論文用）

論文の "Experiments" セクションには以下のように記述することを目指す。

> "To statistically validate the effectiveness of FAHC, we conducted a paired comparative study on 30 prompts categorized as 'Disappearance'. The proposed method (Beta=-0.5) showed a statistically significant increase within VBench Dynamic Degree compared to the baseline ($M=0.42$ vs $M=0.28$, $p < 0.001$, Cohen's $d=0.85$). Furthermore, in the 'Standard Motion' category, no significant degradation in Subject Consistency was observed ($p = 0.34$), suggesting that our method selectively enhances dynamics only when semantically required."

## 6. 具体的なアクションプラン

1.  `experiments/benchmark_prompts_expanded.json` の作成（各カテゴリ30件程度）。
2.  `experiments/run_benchmark.py` の改修（全件実行モード、カテゴリごとのフォルダ分け保存）。
3.  `analysis/statistical_test.py` の作成（t検定と可視化を行うスクリプト）。

---
※本計画はフィッシャーの3原則（無作為化、繰り返し、局所管理）に則り、科学的な妥当性を最大限に高めるよう設計されている。
