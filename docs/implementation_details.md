# 実装の詳細

本ドキュメントでは、本リポジトリで実装された **Relaxation (Adaptive CFG)** および **Temporal Unlearning** に関する技術的な詳細を解説します。

## 1. Adaptive CFG (緩和手法)

Relaxation手法の中核ロジックは `diffusers_helper/k_diffusion/wrapper.py` に実装されています。
デノイジングプロセス全体で一定の Classifier-Free Guidance (CFG) スケールを使用するのではなく、ノイズレベル $\sigma_t$ （$\sigma_t=1.0$ が初期ステップに対応）に基づいてスケールを動的に調整します。

### 計算式
調整後の CFG スケール $s(\sigma_t)$ は以下の式で計算されます：

$$
s(\sigma_t) = s_{min} + (s_{base} - s_{min}) \cdot (1 - \beta \cdot \sigma_t^p)
$$

ここで：
- $s_{base}$: 目標とする基本CFGスケール (例: 6.0)。
- $s_{min}$: 緩和時の最小CFGスケール (例: 1.0)。
- $\beta$ (**Beta**): **Relaxation Strength**（緩和強度）。$\beta > 0$ であれば、初期段階でスケールを下げます。
- $p$ (**Power**): **Decay Power**（減衰指数）。$p < 1.0$ であれば緩和効果を長く持続させ、$p > 1.0$ であれば急速に減衰させます。

### コード解説 (`diffusers_helper/k_diffusion/wrapper.py`)

```python
def adaptive_cfg_scale(sigma, cfg_base, cfg_min=1.0, beta=0.7, power=0.7):
    # 安全のため sigma を [0, 1] にクランプ
    sigma_clamped = torch.clamp(sigma, 0.0, 1.0)
    
    if beta >= 0:
        # 正の Beta: Relaxation (低い値から始まり、高い値へ戻る)
        # Power は減衰曲線のカーブを制御する
        sigma_curve = sigma_clamped ** power
        cfg_adjusted = cfg_min + (cfg_base - cfg_min) * (1.0 - beta * sigma_curve)
    else:
        # 負の Beta: Boost (高い値から始まり、低い値へ戻る)
        # 比較実験用 (Force/Impulse 仮説の検証)
        boost = abs(beta) * 3.0
        sigma_curve = sigma_clamped ** power 
        cfg_adjusted = cfg_base + (cfg_base * boost * sigma_curve)
        
    return cfg_adjusted
```

## 2. パイプラインへの統合

`diffusers_helper/k_diffusion/wrapper.py` 内の `fm_wrapper` 関数において、Transformerの呼び出しをインターセプトし、計算された動的スケールを注入します。

```python
# パイプラインから渡された設定を取得
adaptive_cfg_config = extra_args.get('adaptive_cfg', None)

if adaptive_cfg_config and adaptive_cfg_config.get('enabled', False):
    # 動的な CFG スケールを計算
    cfg_scale = adaptive_cfg_scale(
        sigma.mean(),
        cfg_base=cfg_scale_base,
        cfg_min=adaptive_cfg_config.get('cfg_min', 1.0),
        beta=adaptive_cfg_config.get('beta', 0.7),
        power=adaptive_cfg_config.get('power', 0.7),
    )
```

この設定 (`adaptive_cfg_config`) は、`HunyuanVideoPipeline`（`diffusers_helper/pipelines/k_diffusion_hunyuan.py` で修正済み）から `extra_args` 辞書を通じて渡されます。

## 3. Temporal Unlearning (Gaussian Blur)

Temporal Unlearning は Attention Processor 内部（本ドキュメントでは概念的説明に留めますが）で適用されます。Self-Attention 層の Key および Value 行列に対し、時間軸方向に 1D ガウシアンブラーを適用します。

$$
\tilde{K}_{\tau} = \sum_{j} G(j - \tau; \sigma_{blur}) \cdot K_{j}
$$

これにより、特徴量の時間的な局所性を拡散させ、前のフレームの正確な状態（例: 立ち姿）の記憶を「ぼやけさせる」ことで、プロンプトが提示する新しい状態（例: ジャンプ）での上書きを容易にします。

## 4. UI 統合

パラメータは Gradio インターフェース (`demo_gradio_f1.py`) から制御可能です。
"Advanced Settings" アコーディオンを追加し、以下を公開しています：
- **Enable Adaptive CFG**:機能のオン/オフ。
- **Beta**: 緩和強度のスライダー。
- **Power**: 減衰曲線のスライダー。
- **CFG Min**: 最小スケールのスライダー。

デモの実行:
```bash
python demo_gradio_f1.py
```
