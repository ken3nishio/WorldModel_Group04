# 成功条件の評価レポート

**日時**: 2026-02-11 14:21:26
**動画**: `/content/WorldModel_Group04/experiments/runs/run_20260211_140232/generated_beta_neg0_5_blur_1_1_20260211_140232.mp4`
**プロンプト**: `Static background. A man walks forward and out of view. Empty background remains.`

## 実験条件

| パラメータ | 値 |
|---|---|
| seed | 31337 |
| steps | 25 |
| cfg_scale | 6.0 |
| distilled_cfg_scale | 10.0 |
| beta | -0.5 |
| blur | 1.1 |
| mp4_compression | 16 |
| total_sections | 5 |

## 1. 消失分析 (CLIP Object vs Empty)

- **最終フレーム Empty確率**: 0.0075
- **最終フレーム Object確率**: 0.9925
- **初期Object確率**: 0.2136
- **Object確率の減少量**: -0.7789
- **Empty確率ピークフレーム**: 0
- **Empty確率ピーク値**: 0.7864
- **消失成功判定（最終フレーム）**: ❌ 失敗

> **解釈**: 最終フレームではまだObject確率がEmpty確率を上回っている。消失が完了していない可能性がある。

## 2. Frame-wise CLIP Score (ターゲットプロンプトとの整合性)

- **ターゲットプロンプト**: `empty background remains, static scene`
- **開始時スコア**: 0.1818
- **終了時スコア**: 0.1802
- **Slope（傾き）**: -0.000075
- **平均スコア**: 0.1844
- **標準偏差**: 0.0112

> **解釈**: 負のslopeはフレームがターゲットから離れていることを示す。期待される変化が起きていない可能性がある。

## 3. LPIPS (知覚変化率)

- **平均LPIPS距離**: 0.0463
- **標準偏差**: 0.0358
- **最大変化**: 0.2285
- **前半平均**: 0.0576
- **後半平均**: 0.0350

> **解釈**: 前半の変化率が高く、後半は安定している。これは「人が歩き去り、その後は背景が静止する」という期待動作と一致する。

## 総合判定

| 成功基準 | 結果 |
|---|---|
| 消失成功（最終フレーム） | ❌ |
| Object確率減少 | ❌ |
| CLIP Slope 正 | ❌ |
| 動的変化あり (LPIPS > 0.02) | ✅ |

**合格: 1/4**
