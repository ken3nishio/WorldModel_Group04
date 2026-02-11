# 評価指標の再検討（最終版）：脱・自作指標

## 1. 方針変更
**「自作指標（STF, GAS, DF）の使用を禁止する」。**
独自の評価指標を用いることは、その妥当性検証のために新たな論文執筆が必要となるため、本プロジェクトの趣旨（Step-Adaptive CFGの評価）から逸脱する。
よって、**完全に既存の権威あるベンチマーク（VBench）のみ**を用いて、論理的に「消失」を評価する手法を採用する。

## 2. VBenchのみを用いた「消失」の評価ロジック

VBenchの既存指標（Subject Consistency, Dynamic Degree, Text-to-Video Alignment）の**相関関係**を読み解くことで、「消失」の成功を定義する。

| 指標名 | 測定内容 | "通常" タスクでの理想 | **"消失" タスクでの理想** |
| :--- | :--- | :--- | :--- |
| **Subject Consistency** | 被写体の一貫性（ID維持） | **高** (維持されるべき) | **低** (消える＝一貫性が崩れるべき) |
| **Dynamic Degree** | 動きの量 | **高** | **高** (変化が起きるべき) |
| **Motion Smoothness** | 動きの滑らかさ | **高** | **高** (ノイズで消えるのはNG) |
| **Human Preference**<br>(Text Alignment) | プロンプト適合性 | **高** | **高** (「消える」という指示に合う) |

### 成功の定義（Logic）

1.  **通常プロンプト（Maintenance）**:
    *   `Subject Consistency` が **維持** され、かつ `Dynamic Degree` が向上する。

2.  **消失プロンプト（Disappearance）**:
    *   `Subject Consistency` が **ベースラインよりも有意に低下** し（被写体が消えた証拠）、
    *   かつ `Motion Smoothness` が **維持** されている（画質崩壊ではない証拠）。

このロジックであれば、新たな指標を導入することなく、**「VBenchのスコア分布の変化」** だけでStep-Adaptive CFGの有効性を主張できる。

## 3. 実施計画

1.  `evaluation/run_vbench_custom.py` から自作指標のコードを全削除。
2.  VBenchの標準機能のみを呼び出すように改修。
3.  評価レポートにおいて、スコアの絶対値ではなく「ベースライン（beta=0.0）との差分」を重視して記述する。
