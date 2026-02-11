# 成功条件の評価レポート

**日時**: 2026-02-11 12:05:51
**動画**: `/content/WorldModel_Group04/experiments/runs/run_20260211_115816/generated_beta_neg0_5_blur_1_3_20260211_115816.mp4`
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
| total_sections | 1 |

## 1. 消失分析 (CLIP Object vs Empty)

- **最終フレーム Empty確率**: 0.0068
- **最終フレーム Object確率**: 0.9932
- **初期Object確率**: 0.2130
- **Object確率の減少量**: -0.7802
- **Empty確率ピークフレーム**: 0
- **Empty確率ピーク値**: 0.7870
- **消失成功判定（最終フレーム）**: ❌ 失敗

> **解釈**: 最終フレームではまだObject確率がEmpty確率を上回っている。消失が完了していない可能性がある。

## 2. Frame-wise CLIP Score (ターゲットプロンプトとの整合性)

- **ターゲットプロンプト**: `empty background remains, static scene`
- **開始時スコア**: 0.1843
- **終了時スコア**: 0.1992
- **Slope（傾き）**: -0.000091
- **平均スコア**: 0.1985
- **標準偏差**: 0.0063

> **解釈**: 負のslopeはフレームがターゲットから離れていることを示す。期待される変化が起きていない可能性がある。

## 3. LPIPS (知覚変化率)

- **平均LPIPS距離**: 0.0682
- **標準偏差**: 0.0333
- **最大変化**: 0.1428
- **前半平均**: 0.0722
- **後半平均**: 0.0642

> **解釈**: 前半の変化率が高く、後半は安定している。これは「人が歩き去り、その後は背景が静止する」という期待動作と一致する。

## 総合判定

| 成功基準 | 結果 |
|---|---|
| 消失成功（最終フレーム） | ❌ |
| Object確率減少 | ❌ |
| CLIP Slope 正 | ❌ |
| 動的変化あり (LPIPS > 0.02) | ✅ |

**合格: 1/4**
