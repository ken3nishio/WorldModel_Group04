# 最終成果報告: バク転生成における「Static Death」の克服

## 1. 達成された成果
HunyuanVideoにおける「Static Death（静的死）」問題を、推論時のパラメータ制御のみ（Training-free）で解決する手法を確立しました。

### 定量的成果
- **VideoMAE Score**: 0.004 (Baseline) $\to$ **0.394 (Ours)**
- **改善率**: 約100倍
- **Top Class**: "dancing" $\to$ **"somersaulting"**
- **LPIPS**: 0.166 $\to$ 0.263（崩壊ライン0.5以下を維持）

## 2. 確立された理論: "Sustained Relaxation"
当初の「初期推力（Impulse/Boost）」仮説は棄却されました。
代わりに、「**初期の拘束バイアスを適度に、かつ持続的に緩める（Relaxation）**」ことが、高一貫性モデルに動的アクションを実行させるための鍵であることが実証されました。

### 最適パラメータ設定
- **Beta ($\beta$): 0.75** (Relaxation Strength)
  - 1.0では強すぎてアクションが定まらない。0.75がスイートスポット。
- **Power ($p$): 0.7** (Decay Rate)
  - 2.0（急減衰）では後半に静的バイアスが復活してしまう。バク転のような長い滞空時間を支えるには、緩やかな減衰（0.7）が必要。
- **Blur ($\sigma$): 0.6** (Temporal Unlearning)
  - 過去の立ち姿を忘却させるための必須要素。

## 3. 提出物
- `docs/paper/main.pdf`: 完成版論文
- `docs/paper/experiment_results.tex`: 実験結果・考察の詳細記述
- `experiments/results/full_report_backflip (3).md`: 根拠となる生データ

この研究成果は、大規模動画生成モデルの制御可能性における重要なマイルストーンとなります。
