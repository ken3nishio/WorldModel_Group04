# 成功条件の評価レポート

**日時**: 2026-02-11 13:57:23
**動画**: `/content/WorldModel_Group04/experiments/runs/run_20260211_133635/generated_beta_neg0_5_blur_1_3_20260211_133635.mp4`
**プロンプト**: `Static background. A man walks forward and out of view. Empty background remains.`

## 実験条件

| パラメータ | 値 |
|---|---|
| seed | 31337 |
| steps | 25 |
| cfg_scale | 6.0 |
| distilled_cfg_scale | 10.0 |
| beta | -0.5 |
| blur | 1.3 |
| mp4_compression | 16 |
| total_sections | 5 |

## 1. 消失分析 (CLIP Object vs Empty)

- **最終フレーム Empty確率**: 0.0172
- **最終フレーム Object確率**: 0.9828
- **初期Object確率**: 0.2339
- **Object確率の減少量**: -0.7489
- **Empty確率ピークフレーム**: 0
- **Empty確率ピーク値**: 0.7661
- **消失成功判定（最終フレーム）**: ❌ 失敗

> **解釈**: 最終フレームではまだObject確率がEmpty確率を上回っている。消失が完了していない可能性がある。

## 2. Frame-wise CLIP Score (ターゲットプロンプトとの整合性)

- **ターゲットプロンプト**: `empty background remains, static scene`
- **開始時スコア**: 0.1813
- **終了時スコア**: 0.1774
- **Slope（傾き）**: 0.000068
- **平均スコア**: 0.1918
- **標準偏差**: 0.0111

> **解釈**: 正のslopeはフレームが時間経過とともにターゲット状態に近づいていることを示す。消失・変化が確認できる。

## 3. LPIPS (知覚変化率)

- **平均LPIPS距離**: 0.0639
- **標準偏差**: 0.0520
- **最大変化**: 0.4972
- **前半平均**: 0.0539
- **後半平均**: 0.0738

> **解釈**: 後半の変化率の方が高い。動画全体を通じて変化が続いている。

## 総合判定

| 成功基準 | 結果 |
|---|---|
| 消失成功（最終フレーム） | ❌ |
| Object確率減少 | ❌ |
| CLIP Slope 正 | ✅ |
| 動的変化あり (LPIPS > 0.02) | ✅ |

**合格: 2/4**
