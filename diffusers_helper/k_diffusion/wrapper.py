import torch


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0):
    if guidance_rescale == 0:
        return noise_cfg

    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg
    return noise_cfg


def adaptive_cfg_scale(sigma, cfg_base, cfg_min=1.0, beta=0.7, power=0.7, t_scale=1000.0):
    """
    デノイジングタイムステップに応じてCFGスケールを適応的に変化させる。
    
    Args:
        sigma: 現在のノイズレベル (0〜1、高いほど初期段階)
        cfg_base: 基本CFGスケール
        cfg_min: 最小CFGスケール（初期段階の下限）
        beta: 減衰率 (0=常にcfg_base, 1=初期は完全にcfg_min)
        t_scale: タイムスケール（互換性のため保持）
    Returns:
        調整されたCFGスケール
    
    数式: cfg(σ) = cfg_min + (cfg_base - cfg_min) * (1 - β * σ^p)
    """
    # sigmaが高い = 初期段階
    sigma_clamped = torch.clamp(sigma, 0.0, 1.0)
    
    if beta >= 0:
        # Positive Beta: Start Low -> End High (Decay/Relaxation)
        sigma_curve = sigma_clamped ** power
        cfg_adjusted = cfg_min + (cfg_base - cfg_min) * (1.0 - beta * sigma_curve)
    else:
        # Negative Beta: Start High -> End Low (Boost)
        boost = abs(beta) * 3.0
        # Boost時もPowerを適用できるようにする（デフォルト挙動維持のため条件分岐しても良いが、今回は統一）
        sigma_curve = sigma_clamped ** power 
        
        cfg_adjusted = cfg_base + (cfg_base * boost * sigma_curve)
        
    return cfg_adjusted


def fm_wrapper(transformer, t_scale=1000.0):
    def k_model(x, sigma, **extra_args):
        dtype = extra_args['dtype']
        cfg_scale_base = extra_args['cfg_scale']
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
                power=adaptive_cfg_config.get('power', 0.7),
            )
            # Debug output to verify behavior (print only for step 0 and every 5 steps)
            # if sigma > 0.9 or (int(sigma * 100) % 20 == 0):
            print(f"DEBUG: Sigma={sigma.mean():.4f}, Base={cfg_scale_base}, Calc_CFG={cfg_scale:.4f}")
        else:
            cfg_scale = cfg_scale_base

        x = x.to(dtype)
        timestep = (sigma * t_scale).to(dtype)

        if concat_latent is None:
            hidden_states = x
        else:
            hidden_states = torch.cat([x, concat_latent.to(x)], dim=1)

        pred_positive = transformer(hidden_states=hidden_states, timestep=timestep, return_dict=False, **extra_args['positive'])[0].float()

        if cfg_scale == 1.0:
            pred_negative = torch.zeros_like(pred_positive)
        else:
            pred_negative = transformer(hidden_states=hidden_states, timestep=timestep, return_dict=False, **extra_args['negative'])[0].float()

        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
        pred = rescale_noise_cfg(pred_cfg, pred_positive, guidance_rescale=cfg_rescale)

        x0 = x.float() - pred.float() * append_dims(sigma, x.ndim)

        return x0.to(dtype=original_dtype)

    return k_model
