# FramePack Evaluation: VBench + Standard Metrics

## 1. 概要
本評価スキームは、FramePackの「静的バイアス」と「一貫性」のトレードオフを多角的に評価するため、**VBench**（最新の総合指標）に加え、伝統的かつ信頼性の高い **FVD** (Fréchet Video Distance) と **LPIPS** (Perceptual Similarity) を併用する。

## 2. 採用指標とその役割

### ① VBench (主指標: セマンティック評価)
- **Subject Consistency**: 被写体のID維持率。
- **Dynamic Degree**: 動きの量。
- **Motion Smoothness**: 動きの質。
- **役割**: 「被写体が何であるか」「どう動いているか」という意味的な正しさを測る。

### ② FVD (副指標: 分布間距離)
- **Fréchet Video Distance**: 生成動画の分布が、実世界の動画分布とどれだけ近いかを測る。
- **役割**: 「動画としての自然さ」を測る業界標準指標。
- **FramePackでの意義**: CFG操作によって動画が不自然（分布外）になっていないかを監視するガードレールとして機能する。

### ③ LPIPS (副指標: フレーム間類似度)
- **Learned Perceptual Image Patch Similarity**: 隣接フレーム間の知覚的な変化量を測る。
- **役割**: 「変化の大きさ（Dynamics）」をピクセルレベルではなく、特徴量レベルで測定する。
- **FramePackでの意義**:
    - **Maintenanceタスク**: LPIPSが **低い**（変化が少ない）ことが望ましい。
    - **Disappearanceタスク**: LPIPSが **高い**（変化が大きい）ことが「変動を生み出せた」証拠となり得る。

## 3. 総合評価ロジック

これらを組み合わせることで、自作指標に頼らずとも、既存指標だけで「消失」の成功を論証できる。

| タスク | 成功パターン | 解釈 |
| :--- | :--- | :--- |
| **Maintenance**<br>(通常動作) | - VBench Consistency: **維持**<br>- FVD: **低**<br>- LPIPS: **低〜中** | 品質と一貫性を保ちつつ、自然な動画になっている。 |
| **Disappearance**<br>(消失) | - VBench Consistency: **低下**<br>- FVD: **低** (維持)<br>- LPIPS: **上昇** | **品質（FVD）を保ったまま**、一貫性を崩し（Consistency低下）、大きな変化（LPIPS上昇）を生み出せた。 |

## 4. 実装計画

1.  `run_benchmark.py` に `torcheval` や `lpips` ライブラリを用いた計算ロジックを追加。
2.  VBenchのスコアと合わせてCSV出力し、上記のロジックに基づいてレポートを作成する。
