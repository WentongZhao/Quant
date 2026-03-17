import os
import torch

from qvla.methods.smoothquant.quantize.smooth import smooth_lm, smooth_vit_timm
from qvla.methods.smoothquant.quantize.quantizer import (
    pseudo_quantize_model_weight_act,
    pseudo_quantize_vision_weight_act,
)
from qvla.methods.smoothquant.quantize.gen_act_scales import (
    get_act_scales,
    get_act_scales_vision_timm,
)
from qvla.utils.model_utils import get_llm_backbone, get_llm_model


def _save_scales(path: str, scales: dict, label: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    torch.save(scales, path)
    print(f"SmoothQuant {label} scales saved at {path}")


def smoothquant_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_sq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    vision_scale_path: str = None,
    w_bit: int = 4,
    a_bit: int = 8,
    alpha: float = 0.5,
    sq_target: str = "llm",
    act_scale_mode: str = "max",
    act_scale_quantile: float = 0.999,
    w_vision: float = 1.0,
    w_proprio: float = 1.0,
    w_text: float = 1.0,
    w_action: float = 1.0,
    w_zero_point: bool = False,
    a_zero_point: bool = False,
):
    '''
    SmoothQuant entrypoint for OpenVLA.

    - sq_target="llm": legacy LLM-only SmoothQuant path.
    - sq_target="vision": timm vision encoder SmoothQuant path.
    '''
    if not run_sq_process and not pseudo_quant:
        raise ValueError("SmoothQuant requires --run_process and/or --pseudo_quant.")
    if sq_target not in {"llm", "vision"}:
        raise ValueError(f"Unsupported sq_target={sq_target!r}; expected 'llm' or 'vision'.")

    act_scales = None

    if sq_target == "llm":
        if scale_path is None:
            raise ValueError("SmoothQuant LLM path requires --scale_path.")

        if run_sq_process:
            calib_model = get_llm_model(model)
            act_scales = get_act_scales(
                calib_model,
                prompt_inputs,
                prompt_kwargs,
                act_scale_mode=act_scale_mode,
                act_scale_quantile=act_scale_quantile,
                w_vision=w_vision,
                w_proprio=w_proprio,
                w_text=w_text,
                w_action=w_action,
            )
            _save_scales(scale_path, act_scales, label="LLM")

        if pseudo_quant:
            if act_scales is None:
                act_scales = torch.load(scale_path, map_location="cpu")
            # SmoothQuant scales are keyed by the backbone module names (e.g., "layers.0.*").
            backbone = get_llm_backbone(model)
            smooth_lm(backbone, act_scales, alpha)
            pseudo_quantize_model_weight_act(
                get_llm_model(model),
                w_bit=w_bit,
                a_bit=a_bit,
                w_zero_point=w_zero_point,
                a_zero_point=a_zero_point,
            )

        return model

    # sq_target == "vision"
    if vision_scale_path is None:
        raise ValueError("SmoothQuant vision path requires --sq_vision_scale_path.")
    if not hasattr(model, "vision_backbone"):
        raise ValueError("sq_target='vision' expects an OpenVLA-like model with `vision_backbone`.")

    if run_sq_process:
        act_scales = get_act_scales_vision_timm(
            model,
            prompt_inputs,
            prompt_kwargs,
            act_scale_mode=act_scale_mode,
            act_scale_quantile=act_scale_quantile,
        )
        _save_scales(vision_scale_path, act_scales, label="vision")

    if pseudo_quant:
        if act_scales is None:
            act_scales = torch.load(vision_scale_path, map_location="cpu")
        smooth_vit_timm(model.vision_backbone, act_scales, alpha)
        pseudo_quantize_vision_weight_act(
            model,
            w_bit=w_bit,
            a_bit=a_bit,
            w_zero_point=w_zero_point,
            a_zero_point=a_zero_point,
        )

    return model
