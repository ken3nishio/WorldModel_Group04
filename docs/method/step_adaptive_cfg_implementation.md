# Step-Adaptive CFG 実装詳細ドキュメント

## 1. 背景と問題意識

### 1.1 FramePackにおける「記憶による変化の抑制」問題

FramePackは長尺動画生成において、計算効率と時間的一貫性を両立させる画期的なアーキテクチャである。しかし、その強力な「記憶」メカニズムには副作用が存在する。

**核心的な問題**:
> ユーザーが「被写体が消える」「シーンが大きく変化する」といったプロンプトを指示しても、モデルが保持する「過去の記憶（被写体が存在する状態）」が強力な制約として働き、生成プロセスにおいて被写体を再描画し続けてしまう。

この現象は `docs/framepack_problem.txt` の Section 5 で詳細に分析されている：

> 「デノイジングプロセスの初期段階（タイムステップ T が大きい時）において、条件画像（Image Condition）の高周波成分がモデルに『過剰な構造的制約』を与えていることを発見した。」

### 1.2 ALG (Adaptive Low-Pass Guidance) 論文からの知見

2025年に発表されたALG論文は、Image-to-Videoモデルにおける「動きの抑制」を周波数領域から分析した。

**主要な発見**:
1. 拡散モデルは**初期段階で全体的な構図（低周波）を決定**し、**終盤で詳細（高周波）を描き込む**
2. 条件画像の全周波数成分を最初から入力すると、モデルは「最初から詳細を維持しなければならない」と誤認する
3. これにより、大きな構造変化（移動、変形、消失）が起こせなくなる

**ALGの解決策**:
- 初期ステップで条件画像にローパスフィルタを適用し、高周波成分を除去
- VBench-I2Vテストで**動画のダイナミック度が平均36%向上**

---

## 2. 元論文・実装・数式における課題点の詳細分析

本セクションでは、FramePackおよびその基盤となるHunyuanVideoの実装において、「変化の抑制」を引き起こしている技術的要因をステップバイステップで分析する。

---

### Step 1: FramePackのコンテキスト圧縮における課題

#### 1.1 元論文の設計意図

FramePackは、長尺動画生成における計算コスト問題を解決するため、**幾何級数的なコンテキスト圧縮**を採用している。

**元論文の数式**:
$$C = \sum_{k=0}^{\infty} \frac{T_{base}}{\lambda^k}$$

ここで：
- $C$: 総コンテキスト量
- $T_{base}$: 最新フレームに割り当てられるトークン数
- $\lambda$: 圧縮率（通常 $\lambda=2$）
- $k$: 過去へのフレーム距離

**収束性**: $\lambda=2$ の場合、この級数は $2 \cdot T_{base}$ に収束し、無限の過去を持つ動画でも計算コストは一定。

#### 1.2 この設計の問題点

**問題**: 圧縮は**時間的距離のみ**を基準としており、**映像の内容や意味的な区切り**を考慮していない。

```
フレーム位置:    [t-8] [t-7] [t-6] [t-5] [t-4] [t-3] [t-2] [t-1] [t]
圧縮率:           8x    8x    4x    4x    2x    2x    1x    1x   1x
                 └──シーンA──┘└────────シーンB────────┘└─シーンC─┘
```

上記の例では、シーンが `t-5` で切り替わっているにもかかわらず、`[t-8]` ～ `[t-6]` のシーンAの情報が**まだ4x〜8xの圧縮率で保持されている**。これが次のシーンCに「ゴースト」として漏れ出す原因となる。

#### 1.3 コード上の問題箇所

`demo_gradio_f1.py` L223-228:
```python
indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)

clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
```

**指摘**: 
- `[1, 16, 2, 1]` という圧縮スケジュールは**固定値**であり、シーン変化に応じた動的調整が行われない
- `history_latents` から機械的に過去フレームを取得しており、セマンティックな重要度は考慮されていない

---

### Step 2: 3Dパッチ化カーネルによる空間圧縮の課題

#### 2.1 元論文の設計

FramePackは時間・空間の両方向で圧縮を行う**3Dパッチ化カーネル**を使用：

| 時間距離 | カーネルサイズ | 圧縮効果 |
|----------|----------------|----------|
| 直近 | $(1, 1, 1)$ | 圧縮なし |
| 中期 | $(2, 4, 4)$ | 時間2倍、空間4倍圧縮 |
| 長期 | $(8, 16, 16)$ | 時間8倍、空間16倍圧縮 |

#### 2.2 この設計の問題点

**問題**: 空間的なダウンサンプリングは**低周波成分（大まかな形状、存在感）を保持**し、**高周波成分（詳細なテクスチャ）を破棄**する。

これは一見合理的だが、以下の副作用を生む：

$$\text{圧縮後の情報} \approx \text{lowpass}(\text{元の情報})$$

- 「被写体が存在する」という情報は**空間的に大きな低周波情報**であるため、強く圧縮されても**生き残る**
- 「被写体が消失する」という遷移は**時間的に高周波な変化**を要求するが、圧縮されたコンテキストには「存在」という低周波信号が残存

#### 2.3 数式レベルの問題

圧縮コンテキスト $\mathbf{c}_k$ における被写体存在確率の推定：

$$P(\text{存在} | \mathbf{c}_k) \propto \|\text{lowpass}(\mathbf{c}_k)\|$$

圧縮が進むほど高周波成分が失われるため、「存在」という低周波情報の相対的な重みが増加し、Attention機構が過去の「存在」信号を過剰に参照してしまう。

---

### Step 3: アンチドリフティング機構の過剰適用

#### 3.1 元論文の設計意図

FramePackは長尺生成における**エラー蓄積（ドリフト）**を防ぐため、以下の機構を導入：

1. **アンカーフレームの先行生成**: 始点・終点を先に確定
2. **双方向コンテキスト**: 過去だけでなく未来も参照
3. **反転サンプリング**: 終点から逆順に生成

#### 3.2 この設計の問題点

**問題**: アンチドリフト機構は「急激な変化」を「ドリフト（エラー）」と誤認する。

`docs/framepack_problem.txt` より：
> もしモデルが「被写体は消えないものだ（物体恒常性）」という一般的な物理法則を事前学習で強く持っている場合、アンカーフレーム（例えば終点）を生成する際、プロンプトの「消える」という指示よりも、学習された「物体は急には消えない」というバイアスが優先される可能性が高い。

#### 3.3 数式レベルの問題

アンチドリフト損失関数の暗黙的な定義：

$$\mathcal{L}_{drift} = \sum_{t} \|\mathbf{f}_t - g(\mathbf{f}_{t-1})\|^2$$

ここで $g(\cdot)$ は「滑らかな変化」を仮定した予測関数。この損失は：
- アーティファクト（不自然なノイズ）を抑制 ✓
- 意図的な急激な変化（消失等）も抑制 ✗

**結果**: シーン遷移をアーティファクトと区別できず、両方を平滑化してしまう。

---

### Step 4: CFG計算における固定スケールの課題

#### 4.1 元実装のCFG計算

`wrapper.py` の元のコード：
```python
def fm_wrapper(transformer, t_scale=1000.0):
    def k_model(x, sigma, **extra_args):
        cfg_scale = extra_args['cfg_scale']  # 固定値
        # ...
        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
```

**問題**: `cfg_scale` は全デノイジングステップで**一定値**として適用される。

#### 4.2 デノイジングプロセスにおける問題

拡散モデルのデノイジングは以下のフェーズで進行：

| フェーズ | タイムステップ $t$ | 決定される内容 | 理想的なCFG |
|----------|-------------------|----------------|-------------|
| 初期 | $t \approx T$ | 全体構図、大構造 | **低い**（柔軟性確保） |
| 中期 | $t \approx T/2$ | 中間的な構造 | 中程度 |
| 終盤 | $t \approx 0$ | 細部、テクスチャ | **高い**（忠実度確保） |

**現行実装の問題**:
- 初期段階でも高いCFGが適用されると、条件画像の構造に強く拘束される
- 「被写体がここにいる」という条件が初期から強制され、大きな構造変化が不可能に

#### 4.3 数式レベルの問題

固定CFGの下でのCFGガイダンス：

$$\mathbf{v}_{guided} = \mathbf{v}_{uncond} + w \cdot (\mathbf{v}_{cond} - \mathbf{v}_{uncond})$$

ここで $w$ は固定の `cfg_scale`。

**問題**: $t$ が大きい（初期段階）とき、$\mathbf{v}_{cond}$ には条件画像の全周波数成分が含まれており、これが大構造決定フェーズで過剰な制約となる。

ALG論文の知見によれば：
$$\mathbf{v}_{cond}^{(t)}_{ideal} = \text{lowpass}_{\sigma(t)}(\mathbf{v}_{cond})$$

つまり、初期段階では条件ベクトル自体をローパスフィルタリングすべきだが、現行実装ではこれが行われていない。

---

### Step 5: TransformerのKVキャッシュにおける課題

#### 5.1 Attention機構の数式

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 5.2 問題点

KVキャッシュには過去の全フレームの特徴量が蓄積される。FramePackの圧縮コンテキストでは：

- 大半の $K$ が「被写体が存在する過去のフレーム」由来
- Attentionスコアはこれらのフレームに高く配分される（**Attention Sink現象**）
- 新しいプロンプト（「消える」）による $Q$ がこれらの古い $K$ と高いスコアを持ち、過去の状態を再現してしまう

#### 5.3 コード上の問題

`sample_hunyuan` では `clean_latents` として過去フレームを直接参照：
```python
clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
```

**指摘**: `clean_latents` は過去の「確定した状態」を表し、これがTransformerへの強い制約として機能。シーン遷移時にもこの制約が維持されるため、新しい状態への遷移が阻害される。

---

### Step 6: 学習データバイアスの課題

#### 6.1 静的バイアス (Static Bias)

`docs/framepack_problem.txt` より：
> 高品質な動画データセットには、以下のようなバイアスが含まれている：
> - カメラワークの安定性
> - 2Dアニメーションの影響（背景が静止画で口だけ動く等）
> - 静的シーンの優位性

#### 6.2 これがモデルに与える影響

損失関数の最小化において：
$$\mathcal{L} = \mathbb{E}[\|\mathbf{f}_{pred} - \mathbf{f}_{gt}\|^2]$$

静的なシーンは $\mathbf{f}_{pred} \approx \mathbf{f}_{gt}$ が容易であり、損失が低い。動的なシーンは予測が困難で損失が高い。

**結果**: モデルは「動かないこと」を安全な戦略として学習し、**静的バイアス**が内在化される。FramePackの長尺生成時にこのバイアスが増幅され、「Body shaking in place」現象（その場で震えるだけ）が発生。

---

### Step 7: 課題の総括と本実装の位置づけ

| 課題レベル | 問題点 | 対処の困難さ | 本実装での対処 |
|------------|--------|--------------|----------------|
| コンテキスト圧縮 | 時間距離のみ基準、セマンティック無視 | 高（アーキテクチャ変更必要） | 間接的に対処 |
| 3Dパッチ化 | 低周波存在信号の保持 | 高 | 間接的に対処 |
| アンチドリフト | 意図的変化を誤抑制 | 中 | 間接的に対処 |
| **CFG固定スケール** | 初期段階での過剰制約 | **低** | **直接対処** ✓ |
| KVキャッシュ | Attention Sink | 高 | 間接的に対処 |
| 学習データバイアス | 静的バイアスの内在化 | 極高（再学習必要） | 対処困難 |

**本実装（Step-Adaptive CFG）の位置づけ**:
- 最も**実装コストが低く**、**効果が期待できる**「CFG固定スケール」の課題に直接対処
- 推論時のみの変更で再学習不要
- 他の課題にも間接的に寄与（初期段階での柔軟性向上により、過去コンテキストへの依存を緩和）

---

## 3. Step-Adaptive CFG の設計思想

### 3.1 CFG (Classifier-Free Guidance) の役割

CFGスケールは、生成プロセスにおいて「条件（プロンプト・画像）への忠実度」を制御するパラメータである。

```
pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
```

- **CFGが高い**: 条件への忠実度が高い → 一貫性は高いが変化が抑制される
- **CFGが低い**: 条件への拘束が弱い → 変化は許容されるが一貫性が低下する可能性

### 3.2 時間依存CFGの着想

ALGの「初期段階で高周波を抑制する」というアイデアを、CFGスケールの動的調整として再解釈した：

| デノイジング段階 | 状態 | CFGスケール |
|------------------|------|-------------|
| 初期 (σ ≈ 1) | ノイズが多い、大構造を決定 | **低い** (変化を許容) |
| 終盤 (σ ≈ 0) | ノイズが少ない、詳細を描画 | **高い** (一貫性を維持) |

### 3.3 数式設計

**Step-Adaptive CFGの数式**:

$$\text{cfg}(\sigma) = \text{cfg}_{min} + (\text{cfg}_{base} - \text{cfg}_{min}) \times (1 - \beta \times \sigma)$$

| パラメータ | 意味 | 推奨値 |
|------------|------|--------|
| `cfg_base` | 元のCFGスケール（Distilled CFG Scale） | 10.0 |
| `cfg_min` | 初期段階での最小CFG | 1.0 |
| `β` (beta) | 適応の強さ (0=無効, 1=最大適応) | 0.5〜0.8 |
| `σ` (sigma) | 現在のノイズレベル (0〜1) | 自動計算 |

**動作例** (`cfg_base=10.0`, `cfg_min=1.0`, `beta=0.7`):
- σ=1.0 (初期): cfg = 1.0 + (10.0 - 1.0) × (1 - 0.7 × 1.0) = **3.7**
- σ=0.5 (中間): cfg = 1.0 + (10.0 - 1.0) × (1 - 0.7 × 0.5) = **6.85**
- σ=0.0 (終盤): cfg = 1.0 + (10.0 - 1.0) × (1 - 0.7 × 0.0) = **10.0**

---

## 4. 実装の詳細

### 4.1 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `diffusers_helper/k_diffusion/wrapper.py` | `adaptive_cfg_scale()` 関数の追加、`fm_wrapper()` の改修 |
| `diffusers_helper/pipelines/k_diffusion_hunyuan.py` | パラメータの追加と伝播 |
| `demo_gradio_f1.py` | UIスライダーの追加 |

---

### 4.2 wrapper.py の変更

#### 4.2.1 新規関数: `adaptive_cfg_scale()`

```python
def adaptive_cfg_scale(sigma, cfg_base, cfg_min=1.0, beta=0.7, t_scale=1000.0):
    """
    デノイジングタイムステップに応じてCFGスケールを適応的に変化させる。
    
    ALG (Adaptive Low-Pass Guidance) の知見に基づき、初期段階ではCFGを弱くして
    大きな構造変化（消失、大動作）を許容し、終盤でCFGを強くしてディテールを維持する。
    
    Args:
        sigma: 現在のノイズレベル (0〜1、高いほど初期段階)
        cfg_base: 基本CFGスケール
        cfg_min: 最小CFGスケール（初期段階の下限）
        beta: 減衰率 (0=常にcfg_base, 1=初期は完全にcfg_min)
        t_scale: タイムスケール（互換性のため保持）
    Returns:
        調整されたCFGスケール
    
    数式: cfg(σ) = cfg_min + (cfg_base - cfg_min) * (1 - β * σ)
    """
    # sigmaが高い = 初期段階 = CFGを弱くする
    sigma_clamped = torch.clamp(sigma, 0.0, 1.0)
    cfg_adjusted = cfg_min + (cfg_base - cfg_min) * (1.0 - beta * sigma_clamped)
    return cfg_adjusted
```

**設計意図**:
- `sigma` は拡散モデルのノイズレベルを表し、1に近いほど初期段階
- `torch.clamp()` で範囲を0〜1に制限し、数値的安定性を確保
- `beta=0.0` の場合、`cfg_adjusted = cfg_base` となり、既存動作と完全に同等（後方互換性）

#### 4.2.2 `fm_wrapper()` の改修

```python
def fm_wrapper(transformer, t_scale=1000.0):
    def k_model(x, sigma, **extra_args):
        dtype = extra_args['dtype']
        cfg_scale_base = extra_args['cfg_scale']  # 変数名変更
        cfg_rescale = extra_args['cfg_rescale']
        concat_latent = extra_args['concat_latent']
        
        # Step-Adaptive CFG の設定を取得
        adaptive_cfg_config = extra_args.get('adaptive_cfg', None)

        original_dtype = x.dtype
        sigma = sigma.float()
        
        # 適応型CFGスケールの計算
        if adaptive_cfg_config is not None and adaptive_cfg_config.get('enabled', False):
            cfg_scale = adaptive_cfg_scale(
                sigma.mean(),  # バッチ内で平均を取る
                cfg_base=cfg_scale_base,
                cfg_min=adaptive_cfg_config.get('cfg_min', 1.0),
                beta=adaptive_cfg_config.get('beta', 0.7),
            )
        else:
            cfg_scale = cfg_scale_base
        
        # ... 以降は既存コードと同様 ...
```

**設計意図**:
- `extra_args` から `adaptive_cfg` 設定を取得し、`enabled=True` の場合のみ適応型CFGを適用
- `sigma.mean()` でバッチ内の平均ノイズレベルを使用（バッチ間での一貫性）
- 設定が `None` または `enabled=False` の場合、既存の固定CFGを使用

---

### 4.3 k_diffusion_hunyuan.py の変更

#### 4.3.1 パラメータの追加

```python
@torch.inference_mode()
def sample_hunyuan(
        transformer,
        # ... 既存パラメータ ...
        negative_kwargs=None,
        callback=None,
        # Step-Adaptive CFG パラメータ（新規追加）
        adaptive_cfg_beta=0.0,  # 0.0で無効（既存動作を維持）、0.5-0.8で有効
        adaptive_cfg_min=1.0,   # 初期段階の最小CFG
        **kwargs,
):
```

**設計意図**:
- `adaptive_cfg_beta=0.0` をデフォルトとすることで、既存のコード・ワークフローに影響を与えない
- 明示的に `beta > 0` を指定した場合のみ機能が有効化される

#### 4.3.2 sampler_kwargs への設定追加

```python
    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        # Step-Adaptive CFG 設定（新規追加）
        adaptive_cfg=dict(
            enabled=adaptive_cfg_beta > 0.0,
            beta=adaptive_cfg_beta,
            cfg_min=adaptive_cfg_min,
        ),
        positive=dict(...),
        negative=dict(...),
    )
```

**設計意図**:
- `enabled` フラグによる明示的な有効/無効制御
- 設定を辞書としてまとめることで、将来のパラメータ追加に対応しやすい構造

---

### 4.4 demo_gradio_f1.py の変更

#### 4.4.1 UIスライダーの追加

```python
mp4_crf = gr.Slider(...)

adaptive_cfg_beta = gr.Slider(
    label="Adaptive CFG Beta", 
    minimum=0.0, 
    maximum=1.0, 
    value=0.0,  # デフォルトは無効
    step=0.05, 
    info="0=disabled, 0.5-0.8=enables dynamic motion. Experimental feature for large movements/disappearance."
)
```

**設計意図**:
- `value=0.0` で既存ユーザーへの影響を排除
- `info` テキストで実験的機能であることを明示
- 0.05刻みで細かい調整を可能に

#### 4.4.2 関数シグネチャの更新

`worker()`, `process()` 関数にパラメータを追加し、`sample_hunyuan()` まで伝播させる。

---

## 5. 期待される効果と使用ガイドライン

### 5.1 効果の比較

| 設定 | 一貫性 | ダイナミクス | 用途 |
|------|--------|--------------|------|
| `beta=0.0` | 高 | 低 | 通常の動画生成、一貫性重視 |
| `beta=0.5` | 中-高 | 中 | 中程度の動きを含む動画 |
| `beta=0.7` | 中 | 高 | 消失、大動作、シーン変化 |
| `beta=0.8+` | 低 | 高 | 極端な変化（画質低下リスクあり） |

### 5.2 推奨テストケース

1. **消失**: `"The person gradually fades away and disappears"`
2. **大動作**: `"The dancer spins and jumps across the stage"`
3. **シーン変化**: `"The scene transitions from day to night"`

### 5.3 注意事項

- `beta > 0.8` は画質低下の可能性があるため非推奨
- 実験的機能であり、結果はプロンプトや入力画像に依存
- 効果が不十分な場合、`cfg_min` の調整も検討可能

---

## 6. 今後の拡張可能性

1. **周波数適応型圧縮との併用**: 過去フレームの圧縮時にローパスフィルタを適用
2. **セマンティック・ゲーティング**: プロンプト変化を検知してKVキャッシュを動的に制御
3. **評価指標の整備**: STF (Semantic Transition Fidelity) 等の新指標による定量評価

---

## 7. 参考文献

- FramePack 問題分析: `docs/framepack_problem.txt`
- リサーチギャップ: `docs/research_gap.txt`
- 研究提案: `docs/framepack_research_proposal.txt`
- 先行研究調査: `docs/related_work.txt`
- ALG論文: Adaptive Low-Pass Guidance (2025)
