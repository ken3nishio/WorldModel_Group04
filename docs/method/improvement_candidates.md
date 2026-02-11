# FramePack 改善候補ドキュメント

## 概要
本ドキュメントは、FramePackの「記憶による変化の抑制」問題を解決するための、数式・コード変更・評価指標に関する具体的な改善候補を提案する。既存のドキュメント（`framepack_problem.txt`, `research_gap.txt`, `framepack_research_proposal.txt`, `related_work.txt`）と、`diffusers_helper/`以下のコード分析に基づいている。

---

## 1. 数式レベルの改善候補

### 1.1 周波数適応型コンテキスト圧縮（Frequency-Adaptive Context Compression）

#### 問題の背景
現在のFramePackは、過去フレームの圧縮において時間的距離のみを考慮し、幾何級数的に情報を圧縮している：

$$C = \sum_{k=0}^{\infty} \frac{T_{base}}{\lambda^k}$$

この圧縮は空間的なダウンサンプリング（3Dパッチ化カーネル）により実現されているが、**低周波成分（被写体の存在情報）**が過度に保持され、「消失」などの急激な変化を阻害する。

#### 提案する数式変更
圧縮コンテキストに対して、時間距離に応じた**ガウシアンブラー（ローパスフィルタ）**を適用する：

**現行**:
```
clean_latents = 3Dパッチ圧縮のみ
```

**提案**:
```python
# 時間距離 k に応じたボケ強度 σ(k)
σ(k) = σ_base * (1 + α * k)

# 周波数フィルタリング
blurred_latent = gaussian_blur_3d(latent, kernel_size, sigma=σ(k))
```

数学的定式化：
$$\text{clean\_latent}'_k = \mathcal{G}_{\sigma(k)} * \text{clean\_latent}_k$$

ここで $\mathcal{G}_{\sigma}$ はガウシアンカーネル、$\sigma(k) = \sigma_0 \cdot (1 + \alpha \cdot k)$ は時間距離に応じて増加するボケ強度。

#### 実装箇所
`diffusers_helper/pipelines/k_diffusion_hunyuan.py` の `sample_hunyuan()` 関数内、`clean_latents` 処理部分

---

### 1.2 セマンティック・ゲート付きKVキャッシュ（Semantic-Gated KV Cache）

#### 問題の背景
Transformer の Attention機構において、過去フレームのKVキャッシュが強く参照され、新しいプロンプトの指示が無視される：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 提案する数式変更
プロンプトの変化を検知し、過去のKVキャッシュへのアテンション重みを動的に減衰させる**ゲーティング機構**を導入：

$$\text{Attention}'(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} - \gamma \cdot M_{semantic}\right)V$$

ここで：
- $M_{semantic}$ : セマンティック変化マスク（プロンプト埋め込みの差異から計算）
- $\gamma$ : ゲート強度パラメータ

**プロンプト変化の検出**:
$$\Delta_{prompt} = 1 - \cos(\mathbf{e}_{current}, \mathbf{e}_{prev})$$

$\Delta_{prompt}$ が閾値を超えた場合、$M_{semantic}$ を活性化。

---

### 1.3 デノイジングステップ依存のCFGスケール（Step-Adaptive CFG）

#### 問題の背景
ALG (Adaptive Low-Pass Guidance) の知見によれば、デノイジング初期段階での条件画像の高周波成分が動きを抑制する。現在のCFGは全ステップで一定。

#### 提案する数式変更
デノイジングタイムステップ $t$ に応じて CFG スケールを変化させる：

**現行** (`k_diffusion_hunyuan.py` L84):
```python
distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size)
```

**提案**:
$$\text{cfg}(t) = \text{cfg}_{base} \cdot \left(1 - \beta \cdot \frac{t}{T}\right)$$

- 初期ステップ ($t \approx T$): CFG が弱く、大きな構造変化を許容
- 終盤ステップ ($t \approx 0$): CFG が強く、ディテールを維持

```python
def adaptive_cfg_scale(timestep, cfg_base, beta=0.5, T=1000):
    t_normalized = timestep / T
    return cfg_base * (1 - beta * t_normalized)
```

---

### 1.4 Flux Time Shift の動的調整

#### 問題の背景
現在の `flux_time_shift` 関数は固定パラメータ：

```python
def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
```

#### 提案する数式変更
シーン遷移時に $\mu$ を動的に調整し、急激な変化を許容：

$$\mu(t) = \mu_{base} + \delta \cdot \mathbb{1}_{transition}$$

- 通常時: $\mu = 1.15$（安定した生成）
- 遷移時: $\mu = 2.0+$（変化を促進）

---

## 2. コード変更候補

### 2.1 `utils.py`: 周波数フィルタリング関数の追加

#### 追加する関数
```python
import torch.nn.functional as F

def temporal_blur_latent(latent, sigma=1.0):
    """
    時間方向に対してガウシアンブラーを適用し、
    過去フレームの高周波成分を抑制する
    
    Args:
        latent: Tensor of shape (B, C, T, H, W)
        sigma: ブラー強度
    Returns:
        blurred_latent: 同じ形状のTensor
    """
    B, C, T, H, W = latent.shape
    
    if sigma <= 0 or T <= 1:
        return latent
    
    # ガウシアンカーネル作成（時間方向）
    kernel_size = int(6 * sigma + 1) | 1  # 奇数にする
    kernel_size = min(kernel_size, T)
    
    x = torch.arange(kernel_size, device=latent.device, dtype=latent.dtype)
    x = x - x.mean()
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # (B, C, T, H, W) -> (B*C*H*W, 1, T) に変形して1D conv
    latent_reshaped = latent.permute(0, 1, 3, 4, 2).reshape(-1, 1, T)
    kernel = kernel.view(1, 1, -1)
    
    # パディング
    padding = kernel_size // 2
    latent_padded = F.pad(latent_reshaped, (padding, padding), mode='replicate')
    
    blurred = F.conv1d(latent_padded, kernel)
    blurred = blurred.view(B, C, H, W, T).permute(0, 1, 4, 2, 3)
    
    return blurred


def adaptive_context_decay(latent, time_indices, decay_rate=0.1):
    """
    時間インデックスに基づいて過去のコンテキストを減衰させる
    
    Args:
        latent: Tensor of shape (B, C, T, H, W)
        time_indices: 各フレームの時間インデックス
        decay_rate: 減衰率
    Returns:
        decayed_latent: 減衰適用後のTensor
    """
    B, C, T, H, W = latent.shape
    
    # 時間方向の重み計算
    max_idx = time_indices.max()
    weights = torch.exp(-decay_rate * (max_idx - time_indices))
    weights = weights.view(1, 1, -1, 1, 1).to(latent.device, latent.dtype)
    
    return latent * weights
```

---

### 2.2 `wrapper.py`: Refusal Vector の注入

#### CFG計算部分への変更
```python
def fm_wrapper(transformer, t_scale=1000.0, refusal_vector=None):
    def k_model(x, sigma, **extra_args):
        # ... 既存コード ...
        
        pred_positive = transformer(hidden_states=hidden_states, timestep=timestep, 
                                    return_dict=False, **extra_args['positive'])[0].float()
        
        if cfg_scale == 1.0:
            pred_negative = torch.zeros_like(pred_positive)
        else:
            pred_negative = transformer(hidden_states=hidden_states, timestep=timestep, 
                                        return_dict=False, **extra_args['negative'])[0].float()
        
        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
        
        # Refusal Vector の適用（特定概念の抑制）
        if refusal_vector is not None:
            refusal_strength = extra_args.get('refusal_strength', 0.0)
            pred_cfg = pred_cfg - refusal_strength * refusal_vector.to(pred_cfg)
        
        pred = rescale_noise_cfg(pred_cfg, pred_positive, guidance_rescale=cfg_rescale)
        
        x0 = x.float() - pred.float() * append_dims(sigma, x.ndim)
        return x0.to(dtype=original_dtype)
    
    return k_model
```

---

### 2.3 `k_diffusion_hunyuan.py`: 適応型サンプリングスケジュール

#### シーン認識型シグマスケジュール
```python
def get_adaptive_sigmas(n, mu, scene_transition_points=None):
    """
    シーン遷移ポイントを考慮した適応型シグマスケジュール
    
    Args:
        n: 推論ステップ数
        mu: Flux time shift パラメータ
        scene_transition_points: 遷移ポイント（0-1の正規化された位置のリスト）
    """
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    
    if scene_transition_points is not None:
        for point in scene_transition_points:
            # 遷移ポイント周辺でシグマを増加（変化を促進）
            idx = int(point * n)
            window = max(1, n // 10)
            for i in range(max(0, idx - window), min(n + 1, idx + window)):
                boost = 1.0 + 0.3 * math.exp(-((i - idx) / window) ** 2)
                sigmas[i] = min(sigmas[i] * boost, 1.0)
    
    return sigmas
```

---

### 2.4 `memory.py`: 選択的メモリパージ機能

#### セマンティック・コンテキスト・フラッシング
```python
def selective_memory_purge(kv_cache, purge_mask, purge_ratio=0.5):
    """
    特定のトークンを選択的にKVキャッシュから削除
    
    Args:
        kv_cache: (keys, values) タプル、各々 (B, H, N, D)
        purge_mask: (B, N) のブールマスク、Trueのトークンを削除
        purge_ratio: マスクされたトークンの削除割合
    """
    keys, values = kv_cache
    B, H, N, D = keys.shape
    
    # スコアに基づく選択的削除
    purge_scores = purge_mask.float() * purge_ratio
    keep_mask = torch.rand(B, N, device=keys.device) > purge_scores
    keep_mask = keep_mask.unsqueeze(1).unsqueeze(-1).expand_as(keys)
    
    # マスクされたトークンをゼロ化（完全削除より安定）
    keys = keys * keep_mask
    values = values * keep_mask
    
    return (keys, values)


def compute_semantic_change_score(prompt_embedding_current, prompt_embedding_prev):
    """
    プロンプト埋め込み間のセマンティック変化スコアを計算
    
    Returns:
        change_score: 0-1のスコア（1が最大変化）
    """
    cos_sim = F.cosine_similarity(
        prompt_embedding_current.flatten(),
        prompt_embedding_prev.flatten(),
        dim=0
    )
    return 1.0 - cos_sim.item()
```

---

### 2.5 `hunyuan.py`: プロンプトエンコーディングの拡張

#### 遷移認識型プロンプトエンコーディング
```python
@torch.no_grad()
def encode_prompt_with_transition(
    prompt_current, 
    prompt_prev, 
    text_encoder, 
    text_encoder_2, 
    tokenizer, 
    tokenizer_2, 
    max_length=256,
    transition_strength=0.0
):
    """
    シーン遷移を考慮したプロンプトエンコーディング
    
    Args:
        prompt_current: 現在のプロンプト
        prompt_prev: 直前のプロンプト（遷移検知用）
        transition_strength: 遷移強度（0=通常、1=完全遷移）
    """
    llama_vec_current, clip_pooler_current = encode_prompt_conds(
        prompt_current, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length
    )
    
    if prompt_prev is not None and transition_strength > 0:
        llama_vec_prev, clip_pooler_prev = encode_prompt_conds(
            prompt_prev, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length
        )
        
        # セマンティック変化の計算
        change_score = 1.0 - F.cosine_similarity(
            llama_vec_current.flatten(), 
            llama_vec_prev.flatten(), 
            dim=0
        )
        
        # 変化が大きい場合、過去のベクトルとの混合を避ける
        # （遷移を促進するための「忘却」ベクトルを生成）
        if change_score > 0.3:  # 閾値
            return llama_vec_current, clip_pooler_current, change_score
    
    return llama_vec_current, clip_pooler_current, 0.0
```

---

## 3. 評価指標の提案

### 3.1 既存指標の限界

| 指標 | 評価対象 | FramePack問題への適用 |
|------|----------|----------------------|
| FVD (Fréchet Video Distance) | 全体的な生成品質 | ⚠ 変化の抑制を検出しにくい |
| LPIPS (Perceptual Similarity) | フレーム間の知覚的類似度 | ⚠ 高い類似度が「良い」と誤認 |
| VBench | 多面的な動画評価 | ⚠ 「一貫性」を重視しすぎ |

---

### 3.2 新規提案指標

#### 3.2.1 Semantic Transition Fidelity (STF)

**目的**: プロンプトによる意図的な変化がどれだけ忠実に反映されたかを測定

**定義**:
$$\text{STF} = \frac{1}{N_{trans}} \sum_{i=1}^{N_{trans}} \cos(\mathbf{v}_{generated}^{(i)}, \mathbf{v}_{target}^{(i)})$$

- $N_{trans}$: 遷移ポイントの数
- $\mathbf{v}_{generated}$: 生成された遷移後フレームの特徴ベクトル（CLIP等）
- $\mathbf{v}_{target}$: 新プロンプトから期待される特徴ベクトル

**実装例**:
```python
def compute_stf(generated_frames, prompts, clip_model, transition_indices):
    """
    Args:
        generated_frames: List[PIL.Image] 生成されたフレーム
        prompts: List[str] 各セクションのプロンプト
        clip_model: CLIPモデル
        transition_indices: 遷移が起きるフレームインデックス
    """
    scores = []
    for i, idx in enumerate(transition_indices):
        if i + 1 < len(prompts):
            frame = generated_frames[idx + 5]  # 遷移後のフレーム
            target_prompt = prompts[i + 1]
            
            # CLIP類似度計算
            frame_feat = clip_model.encode_image(frame)
            text_feat = clip_model.encode_text(target_prompt)
            score = F.cosine_similarity(frame_feat, text_feat)
            scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0
```

---

#### 3.2.2 Transition Latency (TL)

**目的**: プロンプト変更から視覚的変化が完了するまでのフレーム遅延を測定

**定義**:
$$\text{TL} = \frac{1}{N_{trans}} \sum_{i=1}^{N_{trans}} (f_{complete}^{(i)} - f_{prompt}^{(i)})$$

- $f_{prompt}$: プロンプトが切り替わったフレーム
- $f_{complete}$: 視覚的変化が安定したフレーム（LPIPS変化率が閾値以下になった時点）

**解釈**:
- 低い TL = 素早い遷移 = 良好
- 高い TL = 遅延した遷移 = 「記憶による変化の抑制」の症状

---

#### 3.2.3 Disappearance Fidelity (DF)

**目的**: 「消失」指示への追従度を直接評価

**定義**:
$$\text{DF} = 1 - \frac{1}{N_{post}} \sum_{j=1}^{N_{post}} D_{object}(f_{j})$$

- $N_{post}$: 消失指示後のフレーム数
- $D_{object}(f_j)$: フレーム $j$ における対象オブジェクトの検出確信度

**実装**: オブジェクト検出モデル（YOLO等）を使用

```python
def compute_disappearance_fidelity(frames, object_class, detector, disappear_frame_idx):
    """
    Args:
        frames: 生成されたフレームのリスト
        object_class: 消失させるべきオブジェクトのクラス
        detector: オブジェクト検出モデル
        disappear_frame_idx: 「消えろ」指示のフレーム
    """
    post_frames = frames[disappear_frame_idx:]
    detection_scores = []
    
    for frame in post_frames:
        detections = detector(frame)
        target_scores = [d.confidence for d in detections if d.class_name == object_class]
        max_score = max(target_scores) if target_scores else 0.0
        detection_scores.append(max_score)
    
    avg_detection = sum(detection_scores) / len(detection_scores)
    return 1.0 - avg_detection
```

---

#### 3.2.4 Dynamic Degree with Intentionality (DDI)

**目的**: VBenchのDynamic Degreeを拡張し、「意図された変化」と「意図せぬドリフト」を区別

**定義**:
$$\text{DDI} = \text{DD}_{aligned} - \lambda \cdot \text{DD}_{unaligned}$$

- $\text{DD}_{aligned}$: プロンプトと整合する動き量
- $\text{DD}_{unaligned}$: プロンプトと無関係な動き量
- $\lambda$: ペナルティ係数

---

#### 3.2.5 Ghosting Artifact Score (GAS)

**目的**: シーン遷移時に発生する「ゴースト」（前シーンの残像）を定量化

**定義**:
$$\text{GAS} = \frac{1}{W} \sum_{w=1}^{W} \max_{t \in T_w} \text{Sim}(f_t, f_{prev\_scene})$$

- $W$: 遷移後の分析ウィンドウ数
- $T_w$: 各ウィンドウ内のフレームセット
- $\text{Sim}$: 前シーン代表フレームとの類似度

**解釈**:
- 低い GAS = クリーンな遷移 = 良好
- 高い GAS = ゴーストが多い = 「記憶による変化の抑制」の症状

---



## 4. 実装優先度と工数見積もり

| 改善候補 | 優先度 | 実装難易度 | 期待効果 | 工数 |
|----------|--------|-----------|----------|------|
| 周波数適応型圧縮 (1.1) | ★★★ | 中 | 高 | 2-3日 |
| Step-Adaptive CFG (1.3) | ★★★ | 低 | 中 | 1日 |
| 時間方向ブラー (2.1) | ★★★ | 低 | 中 | 1日 |
| セマンティック・ゲート (1.2) | ★★ | 高 | 高 | 1週間 |
| Refusal Vector (2.2) | ★★ | 中 | 中 | 2-3日 |
| 適応型シグマ (2.3) | ★★ | 低 | 中 | 1日 |
| STF指標 (3.2.1) | ★★★ | 中 | - | 2日 |
| TL指標 (3.2.2) | ★★★ | 低 | - | 1日 |
| DF指標 (3.2.3) | ★★ | 低 | - | 1日 |


---

## 5. 実験計画

### フェーズ1: ベースライン確立（1週間）
1. 現行FramePackの評価指標ベースラインを取得
2. テストケース作成（消失、シーン遷移、大動作）
3. 新規評価指標の実装と検証

### フェーズ2: 低リスク改善（1週間）
1. Step-Adaptive CFG の実装・評価
2. 時間方向ブラーの実装・評価
3. 適応型シグマスケジュールの実装・評価

### フェーズ3: 中リスク改善（2週間）
1. 周波数適応型圧縮の実装・評価
2. Refusal Vector 機構の実装・評価
3. 総合評価とパラメータチューニング

### フェーズ4: 高リスク改善（オプション、2週間）
1. セマンティック・ゲート付きKVキャッシュの実装
2. デュアルストリーム・メモリ・アーキテクチャの検討

---

## 6. 結論

本ドキュメントでは、FramePackにおける「記憶による変化の抑制」問題に対して、以下の3つの観点から具体的な改善候補を提案した：

1. **数式レベル**: 周波数フィルタリング、セマンティック・ゲーティング、適応型CFG
2. **コードレベル**: 各モジュールへの具体的な関数追加・変更
3. **評価指標**: STF, TL, DF, DDI, GAS の5つの新規指標

これらの改善を段階的に実装・評価することで、一貫性を維持しながらも「意図的な変化」を許容するFramePackの実現が期待される。
