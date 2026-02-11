# FramePack 評価指標選定ドキュメント

## 1. 選定方針
現在直面している「記憶による変化の抑制（Static Bias）」と「一貫性の維持（Consistency）」のトレードオフを正しく評価するため、信頼性の高い外部ベンチマークを採用する。自作指標ではなく、CVPR等のトップカンファレンスで採択された信頼性の高い実装を利用する。

## 2. 推奨される評価フレームワーク

### **VBench (CVPR 2024 Highlight)**
- **GitHub**: [https://github.com/Vchitect/VBench](https://github.com/Vchitect/VBench)
- **選定理由**:
  - 動画生成の品質を**16の次元**に分解して評価できる。
  - 特に、FramePackの抱える問題に直結する以下の指標が含まれている：
    1. **Dynamic Degree (ダイナミック度)**: 動画内の動きの大きさを測定。「変化が抑制される」問題を定量化できる。
    2. **Subject Consistency (被写体一貫性)**: 被写体のIDが保持されているかを測定。「ゴースト」や「IDの崩れ」を検知できる。
    3. **Motion Smoothness (動きの滑らかさ)**: 時間的な連続性を測定。
  - **Human Preferenceとの相関が高く**、信頼性が担保されている。

## 3. 具体的な評価指標セット

FramePackの改善（Step-Adaptive CFG等）の効果を測定するために、VBenchから以下のサブセットを使用することを提案する。

| 指標名 | 測定内容 | FramePackでの役割 | 期待される結果 |
|--------|----------|-------------------|----------------|
| **Dynamic Degree** | 動画内の動きの量 | Static Bias（静的バイアス）の解消確認 | **上昇**することを期待 |
| **Subject Consistency** | 被写体の一貫性 | 過剰な変化によるID崩壊の監視 | **維持**または微減に抑える |
| **CLIP Score (Frame)** | テキストとの整合性 | プロンプト（「消える」等）への忠実度 | **上昇**することを期待 |

## 4. 実装計画

### ステップ1: 環境構築
VBenchはPyTorchおよびCLIP、DINO等のモデルに依存するため、専用の仮想環境または依存関係のインストールが必要。

### ステップ2: ラッパースクリプトの作成
`evaluation/run_vbench.py` を作成し、生成された動画フォルダを指定するだけで上記3つの指標等を一括計算できるようにする。

### ステップ3: 比較実験
1. **Baseline**: 改修前のFramePackで動画生成
2. **Proposed**: Step-Adaptive CFG適用後のFramePackで動画生成
3. **Evaluation**: 両者のスコアをVBenchで算出し、Trade-off曲線（Consistency vs Dynamics）を描画

---

この方針で実装を進めてよろしいでしょうか？
承認いただければ、VBenchのセットアップとスクリプト作成を開始します。
