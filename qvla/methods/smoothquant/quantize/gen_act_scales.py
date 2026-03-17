import torch
import torch.nn as nn

import functools
from tqdm import tqdm


def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    # These masks are produced by get_libero_calib_dataset but are not accepted by the LLM forward.
    inputs.pop("vision_mask", None)
    inputs.pop("caption_mask", None)
    masks = {
        "vision_patch_mask": inputs.pop("vision_patch_mask", None),
        "proprio_mask": inputs.pop("proprio_mask", None),
        "action_mask": inputs.pop("action_mask", None),
        "text_mask": inputs.pop("text_mask", None),
    }
    # Drop OpenVLA-specific targets that the LLM forward doesn't accept.
    inputs.pop("ground_truth_actions", None)
    return inputs, masks


def _trim_pads(mini_inputs: dict, mini_masks: dict) -> tuple[dict, dict]:
    if "inputs_embeds" not in mini_inputs or "attention_mask" not in mini_inputs:
        return mini_inputs, mini_masks

    input_embeds = mini_inputs["inputs_embeds"]
    attention_mask = mini_inputs["attention_mask"]
    B, N, C = input_embeds.shape
    keep = attention_mask.reshape(B * N).bool()

    mini_inputs["inputs_embeds"] = input_embeds.reshape(B * N, C)[keep].reshape(B, -1, C)
    mini_inputs["attention_mask"] = attention_mask.reshape(B * N)[keep].reshape(B, -1)
    if "labels" in mini_inputs and isinstance(mini_inputs["labels"], torch.Tensor):
        mini_inputs["labels"] = mini_inputs["labels"].reshape(B * N)[keep].reshape(B, -1)

    for k, v in list(mini_masks.items()):
        if isinstance(v, torch.Tensor) and v.shape[:2] == (B, N):
            mini_masks[k] = v.reshape(B * N)[keep].reshape(B, -1)

    return mini_inputs, mini_masks


def _build_token_weights(
    mini_masks: dict,
    *,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    w_vision: float,
    w_proprio: float,
    w_text: float,
    w_action: float,
) -> torch.Tensor:
    weights = torch.ones((seq_len,), device=device, dtype=dtype)
    for key, w in (
        ("vision_patch_mask", w_vision),
        ("proprio_mask", w_proprio),
        ("text_mask", w_text),
        ("action_mask", w_action),
    ):
        mask = mini_masks.get(key, None)
        if isinstance(mask, torch.Tensor) and mask.numel() == seq_len:
            weights[mask.reshape(-1).bool()] = w
    return weights


def _require_vla_masks(mini_masks: dict, seq_len: int) -> None:
    for key in ("vision_patch_mask", "proprio_mask", "text_mask", "action_mask"):
        mask = mini_masks.get(key, None)
        if not isinstance(mask, torch.Tensor):
            raise RuntimeError(f"Missing required mask for weighted_max: {key}")
        if mask.numel() != seq_len:
            raise RuntimeError(f"Mask length mismatch for {key}: got {mask.numel()} vs expected {seq_len}")
        if mask.dtype is not torch.bool:
            raise RuntimeError(f"Mask dtype must be bool for {key}, got {mask.dtype}")

@torch.no_grad()
def get_act_scales(
    model,
    prompt_inputs,
    prompt_kwargs,
    *,
    act_scale_mode: str = "max",
    act_scale_quantile: float = 0.999,
    w_vision: float = 1.0,
    w_proprio: float = 1.0,
    w_text: float = 1.0,
    w_action: float = 1.0,
):
    if act_scale_mode not in {"max", "weighted_max", "quantile"}:
        raise ValueError(f"Invalid act_scale_mode: {act_scale_mode}")
    if act_scale_mode == "quantile" and not (0.0 < act_scale_quantile <= 1.0):
        raise ValueError(f"Invalid act_scale_quantile: {act_scale_quantile}")
    # --- modified by lyy 12.20 ---
    # Align with MBQ: collect scales on the true LLM backbone if available.
    base_model = model.model if hasattr(model, "model") else model
    model.eval()
    base_model.eval()
    device = next(base_model.parameters()).device
    # --- end modified by lyy 12.20 ---
    act_scales = {}
    ctx = {"token_weights": None}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        flat = tensor.reshape(-1, hidden_dim).detach().abs()
        if act_scale_mode == "weighted_max":
            token_weights = ctx["token_weights"]
            if not isinstance(token_weights, torch.Tensor) or token_weights.numel() == 0:
                raise RuntimeError("weighted_max requires token weights for each sample.")
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                expected = int(tensor.shape[1])
            elif tensor.dim() == 2:
                expected = int(tensor.shape[0])
            else:
                raise RuntimeError(
                    f"weighted_max only supports Linear inputs shaped (1, T, C) or (T, C), got {tuple(tensor.shape)}"
                )
            if token_weights.numel() != expected:
                raise RuntimeError(f"Token weight length mismatch: got {token_weights.numel()} vs expected {expected}")
            flat = flat * token_weights.to(device=flat.device, dtype=flat.dtype).view(-1, 1)

        if act_scale_mode == "quantile":
            stat = torch.quantile(flat.float(), act_scale_quantile, dim=0).cpu()
        else:
            stat = flat.max(dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], stat)
        else:
            act_scales[name] = stat

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    # --- modified by lyy 12.20 ---
    for name, m in base_model.named_modules():
    # --- end modified by lyy 12.20 ---
        if isinstance(m, nn.Linear) and "vis" not in name and "mm_projector" not in name and "lm_head" not in name and "output" not in name and "mlp1" not in name:
            print(f"register hook for {name}")
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )
    
    num_samples = len(next(iter(prompt_inputs.values())))

    inputs, masks = process_input(prompt_inputs, prompt_kwargs)

    for i in tqdm(range(num_samples)):
        mini_inputs = {}
        mini_masks = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                mini_inputs[k] = v[i : i + 1].detach().to(device)
        for k, v in masks.items():
            if isinstance(v, torch.Tensor):
                mini_masks[k] = v[i : i + 1].detach().to(device)
        
        mini_inputs, mini_masks = _trim_pads(mini_inputs, mini_masks)

        if act_scale_mode == "weighted_max" and "attention_mask" in mini_inputs:
            seq_len = int(mini_inputs["attention_mask"].shape[-1])
            _require_vla_masks(mini_masks, seq_len)
            ctx["token_weights"] = _build_token_weights(
                mini_masks,
                seq_len=seq_len,
                device=device,
                dtype=torch.float32,
                w_vision=w_vision,
                w_proprio=w_proprio,
                w_text=w_text,
                w_action=w_action,
            )
        else:
            ctx["token_weights"] = None

        model(**mini_inputs)
        del mini_inputs

    for h in hooks:
        h.remove()

    return act_scales


def _get_vision_featurizers(model) -> list[tuple[str, nn.Module]]:
    vision_backbone = getattr(model, "vision_backbone", None)
    if vision_backbone is None:
        raise ValueError("Vision SmoothQuant requires model.vision_backbone.")

    featurizers: list[tuple[str, nn.Module]] = []
    if hasattr(vision_backbone, "featurizer"):
        featurizers.append(("vision_backbone.featurizer", vision_backbone.featurizer))
    if hasattr(vision_backbone, "fused_featurizer"):
        featurizers.append(("vision_backbone.fused_featurizer", vision_backbone.fused_featurizer))
    if not featurizers:
        raise ValueError("Vision SmoothQuant requires timm featurizer modules on vision_backbone.")
    return featurizers


@torch.no_grad()
def get_act_scales_vision_timm(
    model,
    prompt_inputs,
    prompt_kwargs,
    *,
    act_scale_mode: str = "max",
    act_scale_quantile: float = 0.999,
):
    del prompt_kwargs

    if act_scale_mode not in {"max", "quantile"}:
        raise ValueError(
            f"Vision SmoothQuant supports act_scale_mode in {{'max', 'quantile'}}, got {act_scale_mode!r}."
        )
    if act_scale_mode == "quantile" and not (0.0 < act_scale_quantile <= 1.0):
        raise ValueError(f"Invalid act_scale_quantile: {act_scale_quantile}")
    if "pixel_values" not in prompt_inputs:
        raise ValueError("Vision SmoothQuant expects calibration inputs with key `pixel_values`.")

    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        flat = tensor.reshape(-1, hidden_dim).detach().abs()
        if act_scale_mode == "quantile":
            stat = torch.quantile(flat.float(), act_scale_quantile, dim=0).cpu()
        else:
            stat = flat.max(dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], stat)
        else:
            act_scales[name] = stat

    def stat_input_hook(m, x, y, name):
        del m, y
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for prefix, featurizer in _get_vision_featurizers(model):
        for name, module in featurizer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            full_name = f"{prefix}.{name}" if name else prefix
            hooks.append(module.register_forward_hook(functools.partial(stat_input_hook, name=full_name)))

    pixel_values = prompt_inputs["pixel_values"]
    if not isinstance(pixel_values, torch.Tensor):
        raise ValueError("`pixel_values` must be a torch.Tensor for vision calibration.")
    num_samples = int(pixel_values.shape[0])

    for i in tqdm(range(num_samples)):
        mini_pixel_values = pixel_values[i : i + 1].detach().to(device)
        model.vision_backbone(mini_pixel_values)

    for hook in hooks:
        hook.remove()

    return act_scales
