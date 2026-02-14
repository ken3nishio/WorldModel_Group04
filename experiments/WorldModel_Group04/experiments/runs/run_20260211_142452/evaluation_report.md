# 成功条件の評価レポート

**日時**: 2026-02-11 14:43:25
**動画**: `/content/WorldModel_Group04/experiments/runs/run_20260211_142452/generated_beta_neg0_5_blur_0_95_20260211_142452.mp4`
**プロンプト**: `Static background. A man walks forward and out of view. Empty background remains.`

## 実験条件

| パラメータ | 値 |
|---|---|
| seed | 31337 |
| steps | 25 |
| cfg_scale | 6.0 |
| distilled_cfg_scale | 10.0 |
| beta | -0.5 |
| blur | 0.95 |
| mp4_compression | 16 |
| total_sections | 5 |

## 1. 消失分析 (CLIP Object vs Empty)

- **最終フレーム Empty確率**: 0.0581
- **最終フレーム Object確率**: 0.9419
- **初期Object確率**: 0.2042
- **Object確率の減少量**: -0.7377
- **Empty確率ピークフレーム**: 0
- **Empty確率ピーク値**: 0.7958
- **消失成功判定（最終フレーム）**: ❌ 失敗

> **解釈**: 最終フレームではまだObject確率がEmpty確率を上回っている。消失が完了していない可能性がある。

## 2. Frame-wise CLIP Score (ターゲットプロンプトとの整合性)

- **ターゲットプロンプト**: `empty background remains, static scene`
- **開始時スコア**: 0.1868
- **終了時スコア**: 0.1900
- **Slope（傾き）**: -0.000055
- **平均スコア**: 0.1871
- **標準偏差**: 0.0097

> **解釈**: 負のslopeはフレームがターゲットから離れていることを示す。期待される変化が起きていない可能性がある。

## 3. LPIPS (知覚変化率)

- **平均LPIPS距離**: 0.0523
- **標準偏差**: 0.0466
- **最大変化**: 0.3338
- **前半平均**: 0.0625
- **後半平均**: 0.0420

> **解釈**: 前半の変化率が高く、後半は安定している。これは「人が歩き去り、その後は背景が静止する」という期待動作と一致する。

## 総合判定

| 成功基準 | 結果 |
|---|---|
| 消失成功（最終フレーム） | ❌ |
| Object確率減少 | ❌ |
| CLIP Slope 正 | ❌ |
| 動的変化あり (LPIPS > 0.02) | ✅ |

**合格: 1/4**
