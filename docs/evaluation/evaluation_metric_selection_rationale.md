# FramePack 評価指標選定に関する検討資料 (Rev.2)

## 1. 背景と改訂の目的
FramePackの課題である「記憶による変化の抑制（Static Bias）」を解消するため、Step-Adaptive CFG等の導入を進めている。
当初の計画では、VBenchの `Dynamic Degree`（動き）と `Subject Consistency`（一貫性）のトレードオフを一律に評価する予定であった。
しかし、**「消失（Disappearance）」などのタスクにおいては、被写体の一貫性が低下することこそが「成功」である**という論理的矛盾や、ノイズによる見かけ上の動きの増加（ダイナミクスの誤検知）のリスクが指摘された。

本資料（Rev.2）では、これらの批判的視点を取り入れ、プロンプトのカテゴリごとの評価基準と、品質担保のための追加指標を策定する。

## 2. 評価指標に求められる要件（改訂版）

1.  **タスク別評価**: 「維持する動き」と「変化・消失する動き」を区別して評価できること。
2.  **動きの「質」の担保**: 単に画素が動くだけでなく、その動きが滑らかで、プロンプトの指示通りであること。
3.  **信頼性**: CVPR等のトップカンファレンスで採択された指標であること。

## 3. 採用ベンチマーク：VBench + カテゴリ別評価プロトコル

引き続き **VBench** (CVPR 2024 Highlight) を採用するが、その運用方法を大きく変更する。

### 3.1 評価指標セットの拡張

単なる「動き vs 一貫性」ではなく、**「正しく、滑らかに動いているか」** を多角的に測定する。

| 指標名 | 役割 | FramePack評価における意義 |
| :--- | :--- | :--- |
| **Dynamic Degree** | 動きの量 | Static Bias解消の主指標。ただし単独では信頼しない。 |
| **Subject Consistency** | 被写体維持 | **カテゴリA（通常動作）でのみ**成功指標として扱う。 |
| **Motion Smoothness** | 動きの滑らかさ | **必須（ガードレール指標）**。無理なCFG操作で動画が崩壊（フリッカー/ノイズ）していないかを監視。 |
| **Text-to-Video Alignment** | 指示適合性 | プロンプト（「消える」「変形する」等）通りに動いているかの確認。 |

## 4. プロンプトカテゴリ別の評価基準

全プロンプトを一律評価せず、以下の3カテゴリに分割して評価を行う。

| カテゴリ | プロンプト特性 | 具体例 | 成功の定義 |
| :--- | :--- | :--- | :--- |
| **A. 通常動作** | 被写体は維持され、動作のみを行う | "A girl smiling", "Walking dog" | `Dynamic Degree` ↑ <br> かつ `Subject Consistency` **維持** (-5%以内) |
| **B. 消失・変形** | 被写体自体が変化・消失する | "Car fades away", "Melting ice" | `Dynamic Degree` ↑ <br> かつ `Text-to-Video Alignment` ↑ <br> (`Subject Consistency` の低下は許容/無視) |
| **C. 大規模変化** | シーン全体やカメラが大きく動く | "Zoom in", "Scene transition" | `Dynamic Degree` ↑ <br> かつ `Motion Smoothness` **維持** |

## 5. 評価実行プロトコル

### ステップ1: プロンプトセットの選定と分類
VBenchの標準プロンプトセットから、以下のキーワードを含むものを抽出し、カテゴリB（消失・変形）としてタグ付けを行う。
- keywords: *disappear, fade, melt, transform, explode, vanish*
それ以外をカテゴリA（通常動作）とする。

### ステップ2: ベースライン計測
現行FramePackで生成し、各カテゴリごとのベースラインスコア（Dynamic Degree, Subject Consistency, Motion Smoothness, Alignment）を算出。

### ステップ3: 改善版計測と判定
Step-Adaptive CFG適用版で生成し、以下の**複合条件**を満たすか判定する。

#### 【成功判定基準】
1.  **全体**: `Motion Smoothness` が著しく低下（例: -10%以上）していないこと（**必須前提**）。
2.  **カテゴリA（通常）**: `Dynamic Degree` が上昇しつつ、`Subject Consistency` が維持されていること。
3.  **カテゴリB（消失）**: `Subject Consistency` が「低下」してもよいので、`Text-to-Video Alignment` が上昇していること（指示通りに消えていること）。

---
この改訂版の方針に基づき、VBenchのスクリプト実装（カテゴリ分けロジックを含む）を進めます。
