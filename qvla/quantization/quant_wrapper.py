# from qvla.methods.awq.entry import awq_entry
# from qvla.methods.smoothquant.entry import smoothquant_entry
# from qvla.methods.mbq.entry import mbq_entry
# from qvla.methods.rtn.entry import rtn_entry
# from qvla.methods.spinquant.entry import spinquant_entry  # modified by lyy 12.18
# from qvla.methods.spinquantv2.entry import spinquantv2_entry  # spinquantv2
# from qvla.methods.omniquant.entry import omniquant_entry
# from qvla.methods.ostquant.entry import ostquant_entry


# def _resolve_wa_quant(args) -> bool:
#     return args.wa_quant or (args.w_bit < 16 and args.a_bit < 16)


# def _resolve_weight_quant(args, wa_quant: bool) -> str:
#     weight_quant = args.weight_quant
#     if weight_quant == "auto":
#         weight_quant = "per_channel" if wa_quant else "per_group"
#     return weight_quant

# # --- modified by lyy ---
# def qwrapper(
#     model,
#     prompt_inputs,
#     prompt_kwargs,
#     args,
#     action_head=None,
#     grad_loss_type: str = "l1_regression",
# ):
#     """Route quantization to the requested method and return the (possibly) modified model."""
# # --- end modified by lyy ---
#     if args.method == "awq":
#         model = awq_entry(
#             model,
#             prompt_inputs,
#             prompt_kwargs,
#             run_awq_process=args.run_process,
#             scale_path=args.scale_path,
#             q_group_size=args.w_group,
#             w_bit=args.w_bit,
#         )
#     elif args.method == "smoothquant":
#         model = smoothquant_entry(
#             model,
#             prompt_inputs,
#             prompt_kwargs,
#             run_sq_process=args.run_process,
#             pseudo_quant=args.pseudo_quant,
#             scale_path=args.scale_path,
#             w_bit=args.w_bit,
#             a_bit=args.a_bit,
#             alpha=args.alpha,
#             act_scale_mode=args.sq_scale_mode,
#             act_scale_quantile=args.sq_quantile,
#             w_vision=args.sq_w_vision,
#             w_proprio=args.sq_w_proprio,
#             w_text=args.sq_w_text,
#             w_action=args.sq_w_action,
#         )
#     elif args.method == "mbq":
#         wa_quant = _resolve_wa_quant(args)
#         model = mbq_entry(
#             model,
#             prompt_inputs,
#             prompt_kwargs,
#             run_mbq_process=args.run_process,
#             pseudo_quant=args.pseudo_quant,
#             scale_path=args.scale_path,
#             q_group_size=args.w_group,
#             w_bit=args.w_bit,
#             a_bit=args.a_bit,
#             wa_quant=wa_quant,
#             reweight=args.reweight,
#             distort=args.distort,
#             loss_mode=args.loss_mode,
#             # --- modified by lyy ---
#             capture_modality_grad=getattr(args, "capture_modality_grad", False),
#             action_head=action_head,
#             grad_loss_type=grad_loss_type,
#         )
#         # --- end modified by lyy ---
#     elif args.method == "rtn":
#         wa_quant = _resolve_wa_quant(args)
#         weight_quant = _resolve_weight_quant(args, wa_quant)
#         model = rtn_entry(
#             model,
#             pseudo_quant=args.pseudo_quant,
#             wa_quant=wa_quant,
#             q_group_size=args.w_group,
#             w_bit=args.w_bit,
#             a_bit=args.a_bit,
#             weight_quant=weight_quant,
#             act_quant=args.act_quant,
#         )
#     # --- modified by lyy 12.18 ---
#     elif args.method == "spinquant":
#         # spinquantv2/spinquantv3: opt-in patch paths; keep default spinquant behavior unchanged.
#         spin_quantv2 = bool(getattr(args, "spin_quantv2", False))
#         entry_fn = spinquant_entry
#         if spin_quantv2:
#             entry_fn = spinquantv2_entry

#         entry_kwargs = dict(
#             model=model,
#             prompt_inputs=prompt_inputs,
#             prompt_kwargs=prompt_kwargs,
#             run_spin_process=args.run_process,
#             pseudo_quant=args.pseudo_quant,
#             scale_path=args.scale_path,
#             w_bit=args.w_bit,
#             a_bit=args.a_bit,
#             w_group=args.w_group,
#             spin_rotate=getattr(args, "spin_rotate", True),
#             spin_fp32_had=getattr(args, "spin_fp32_had", False),
#             spin_w_asym=getattr(args, "spin_w_asym", False),
#             spin_a_asym=getattr(args, "spin_a_asym", False),
#             spin_w_rtn=getattr(args, "spin_w_rtn", False),
#             spin_gptq_percdamp=getattr(args, "spin_gptq_percdamp", 0.01),
#             spin_gptq_blocksize=getattr(args, "spin_gptq_blocksize", 128),
#             spin_gptq_actorder=getattr(args, "spin_gptq_actorder", False),
#             spin_int8_down_proj=getattr(args, "spin_int8_down_proj", False),
#             spin_grad_checkpointing=getattr(args, "spin_grad_checkpointing", True),
#             spin_w_clip=getattr(args, "spin_w_clip", False),
#             spin_a_clip_ratio=getattr(args, "spin_a_clip_ratio", 1.0),
#             k_bit=getattr(args, "k_bit", 16),
#             v_bit=getattr(args, "v_bit", 16),
#             k_group=getattr(args, "k_group", -1),
#             v_group=getattr(args, "v_group", -1),
#             spin_k_asym=getattr(args, "spin_k_asym", False),
#             spin_v_asym=getattr(args, "spin_v_asym", False),
#             spin_k_clip_ratio=getattr(args, "spin_k_clip_ratio", 1.0),
#             spin_v_clip_ratio=getattr(args, "spin_v_clip_ratio", 1.0),
#             spin_k_pre_rope=getattr(args, "spin_k_pre_rope", False),
#             spin_lr=getattr(args, "spin_lr", 1e-2),
#             spin_steps=getattr(args, "spin_steps", 200),
#             spin_batch_size=getattr(args, "spin_batch_size", 1),
#             spin_log_every=getattr(args, "spin_log_every", 10),
#             spin_overwrite=getattr(args, "spin_overwrite", False),
#             # --- modified by lyy 12.18 ---
#             action_head=action_head,
#             # --- end modified by lyy 12.18 ---
#             seed=getattr(args, "seed", 0),
#         )
#         # spinquantv2: only pass v2-only kwargs when enabled.
#         if spin_quantv2:
#             entry_kwargs.update(
#                 spin_v2_r2_mse_weight=getattr(args, "spin_v2_r2_mse_weight", 1.0),
#                 # xiugai_lyy: choose recon loss mode (mse|mae) for SpinQuantV2
#                 spin_v2_recon_loss_mode=args.spin_v2_recon_loss_mode,
#                 spin_v2_modality_weighting=getattr(args, "spin_v2_modality_weighting", True),
#                 spin_v2_weight_keys=getattr(args, "spin_v2_weight_keys", ""),
#                 spin_v2_grad_dump_path=getattr(args, "spin_v2_grad_dump_path", None),
#                 spin_v2_grad_hook_target=getattr(args, "spin_v2_grad_hook_target", "linear"),
#                 spin_v2_grad_hook_scope=getattr(args, "spin_v2_grad_hook_scope", "proj_linear"),
#             )
#         model = entry_fn(**entry_kwargs)
#     # --- end modified by lyy 12.18 ---
#     elif args.method == "dfrot":
#         # Optional dependency: keep import lazy so this repo can run without dfrot files present.
#         try:
#             from qvla.methods.dfrot.entry import dfrot_entry  # type: ignore
#         except Exception as e:  # noqa: BLE001
#             raise ModuleNotFoundError(
#                 "Requested method=dfrot, but `qvla/methods/dfrot` is not available in this checkout."
#             ) from e
#         model = dfrot_entry(
#             model,
#             prompt_inputs,
#             prompt_kwargs,
#             run_dfrot_process=args.run_process,
#             pseudo_quant=args.pseudo_quant,
#             scale_path=args.scale_path,
#             w_bit=args.w_bit,
#             a_bit=args.a_bit,
#             w_group=args.w_group,
#             dfrot_rotate=getattr(args, "dfrot_rotate", True),
#             dfrot_init_mode=getattr(args, "dfrot_init_mode", "hadamard"),
#             dfrot_alpha=getattr(args, "dfrot_alpha", 100.0),
#             dfrot_alter_steps=getattr(args, "dfrot_alter_steps", 100),
#             dfrot_procrustes_steps=getattr(args, "dfrot_procrustes_steps", 1),
#             dfrot_bits=getattr(args, "dfrot_bits", 4),
#             dfrot_clip_ratio=getattr(args, "dfrot_clip_ratio", 1.0),
#             dfrot_massive_threshold=getattr(args, "dfrot_massive_threshold", 8.5),
#             dfrot_calib_samples=getattr(args, "dfrot_calib_samples", 16),
#             dfrot_max_tokens=getattr(args, "dfrot_max_tokens", 16384),
#             dfrot_log_every=getattr(args, "dfrot_log_every", 10),
#             dfrot_overwrite=getattr(args, "dfrot_overwrite", False),
#             dfrot_fp32_had=getattr(args, "dfrot_fp32_had", False),
#             dfrot_w_asym=getattr(args, "dfrot_w_asym", False),
#             dfrot_a_asym=getattr(args, "dfrot_a_asym", False),
#             dfrot_w_rtn=getattr(args, "dfrot_w_rtn", False),
#             dfrot_gptq_percdamp=getattr(args, "dfrot_gptq_percdamp", 0.01),
#             dfrot_gptq_blocksize=getattr(args, "dfrot_gptq_blocksize", 128),
#             dfrot_gptq_actorder=getattr(args, "dfrot_gptq_actorder", False),
#             dfrot_w_clip=getattr(args, "dfrot_w_clip", False),
#             dfrot_a_clip_ratio=getattr(args, "dfrot_a_clip_ratio", 1.0),
#             dfrot_int8_down_proj=getattr(args, "dfrot_int8_down_proj", False),
#             k_bit=getattr(args, "k_bit", 16),
#             v_bit=getattr(args, "v_bit", 16),
#             k_group=getattr(args, "k_group", -1),
#             v_group=getattr(args, "v_group", -1),
#             dfrot_k_asym=getattr(args, "dfrot_k_asym", False),
#             dfrot_v_asym=getattr(args, "dfrot_v_asym", False),
#             dfrot_k_clip_ratio=getattr(args, "dfrot_k_clip_ratio", 1.0),
#             dfrot_v_clip_ratio=getattr(args, "dfrot_v_clip_ratio", 1.0),
#             dfrot_disable_qk_rotation=getattr(args, "dfrot_disable_qk_rotation", False),
#             dfrot_o_proj_head_dim_group=getattr(args, "dfrot_o_proj_head_dim_group", False),
#             dfrot_separate_fwd=getattr(args, "dfrot_separate_fwd", False),
#             action_head=action_head,
#             seed=getattr(args, "seed", 0),
#         )
#     elif args.method == "ostquant":
#         # --- modified by lyy ---
#         # OSTQuant needs the external OpenVLA action head for action-level distillation losses.
#         model = ostquant_entry(
#             model,
#             prompt_inputs,
#             prompt_kwargs,
#             args,
#             action_head=action_head,
#         )
#         # --- end modified by lyy ---
#     elif args.method == "omniquant":
#         model = omniquant_entry(
#             model,
#             prompt_inputs,
#             prompt_kwargs,
#             args,
#         )
#     else:
#         raise NotImplementedError

#     return model
from qvla.methods.awq.entry import awq_entry
from qvla.methods.smoothquant.entry import smoothquant_entry
# from qvla.methods.mbq.entry import mbq_entry
# from qvla.methods.rtn.entry import rtn_entry
# from qvla.methods.spinquant.entry import spinquant_entry  # modified by lyy 12.18
# from qvla.methods.spinquantv2.entry import spinquantv2_entry  # spinquantv2
# from qvla.methods.omniquant.entry import omniquant_entry
# from qvla.methods.ostquant.entry import ostquant_entry


def _resolve_wa_quant(args) -> bool:
    return args.wa_quant or (args.w_bit < 16 and args.a_bit < 16)


def _resolve_weight_quant(args, wa_quant: bool) -> str:
    weight_quant = args.weight_quant
    if weight_quant == "auto":
        weight_quant = "per_channel" if wa_quant else "per_group"
    return weight_quant

# --- modified by lyy ---
def qwrapper(
    model,
    prompt_inputs,
    prompt_kwargs,
    args,
    action_head=None,
    grad_loss_type: str = "l1_regression",
):
    """Route quantization to the requested method and return the (possibly) modified model."""
# --- end modified by lyy ---
    if args.method == "awq":
        model = awq_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_awq_process=args.run_process,
            scale_path=args.scale_path,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
        )
    elif args.method == "smoothquant":
        model = smoothquant_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_sq_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            vision_scale_path=args.sq_vision_scale_path,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            alpha=args.alpha,
            sq_target=args.sq_target,
            act_scale_mode=args.sq_scale_mode,
            act_scale_quantile=args.sq_quantile,
            w_vision=args.sq_w_vision,
            w_proprio=args.sq_w_proprio,
            w_text=args.sq_w_text,
            w_action=args.sq_w_action,
            w_zero_point=bool(getattr(args, "sq_w_asym", False)),
            a_zero_point=bool(getattr(args, "sq_a_asym", True)),
        )
    elif False and args.method == "mbq":  # disabled (method not in this repo)
        wa_quant = _resolve_wa_quant(args)
        model = mbq_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_mbq_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            wa_quant=wa_quant,
            reweight=args.reweight,
            distort=args.distort,
            loss_mode=args.loss_mode,
            # --- modified by lyy ---
            capture_modality_grad=getattr(args, "capture_modality_grad", False),
            action_head=action_head,
            grad_loss_type=grad_loss_type,
        )
        # --- end modified by lyy ---
    elif False and args.method == "rtn":  # disabled (method not in this repo)
        wa_quant = _resolve_wa_quant(args)
        weight_quant = _resolve_weight_quant(args, wa_quant)
        model = rtn_entry(
            model,
            pseudo_quant=args.pseudo_quant,
            wa_quant=wa_quant,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            weight_quant=weight_quant,
            act_quant=args.act_quant,
        )
    # --- modified by lyy 12.18 ---
    elif False and args.method == "spinquant":  # disabled (method not in this repo)
        # spinquantv2/spinquantv3: opt-in patch paths; keep default spinquant behavior unchanged.
        spin_quantv2 = bool(getattr(args, "spin_quantv2", False))
        entry_fn = spinquant_entry
        if spin_quantv2:
            entry_fn = spinquantv2_entry

        entry_kwargs = dict(
            model=model,
            prompt_inputs=prompt_inputs,
            prompt_kwargs=prompt_kwargs,
            run_spin_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            w_group=args.w_group,
            spin_rotate=getattr(args, "spin_rotate", True),
            spin_fp32_had=getattr(args, "spin_fp32_had", False),
            spin_w_asym=getattr(args, "spin_w_asym", False),
            spin_a_asym=getattr(args, "spin_a_asym", False),
            spin_w_rtn=getattr(args, "spin_w_rtn", False),
            spin_gptq_percdamp=getattr(args, "spin_gptq_percdamp", 0.01),
            spin_gptq_blocksize=getattr(args, "spin_gptq_blocksize", 128),
            spin_gptq_actorder=getattr(args, "spin_gptq_actorder", False),
            spin_int8_down_proj=getattr(args, "spin_int8_down_proj", False),
            spin_grad_checkpointing=getattr(args, "spin_grad_checkpointing", True),
            spin_w_clip=getattr(args, "spin_w_clip", False),
            spin_a_clip_ratio=getattr(args, "spin_a_clip_ratio", 1.0),
            k_bit=getattr(args, "k_bit", 16),
            v_bit=getattr(args, "v_bit", 16),
            k_group=getattr(args, "k_group", -1),
            v_group=getattr(args, "v_group", -1),
            spin_k_asym=getattr(args, "spin_k_asym", False),
            spin_v_asym=getattr(args, "spin_v_asym", False),
            spin_k_clip_ratio=getattr(args, "spin_k_clip_ratio", 1.0),
            spin_v_clip_ratio=getattr(args, "spin_v_clip_ratio", 1.0),
            spin_k_pre_rope=getattr(args, "spin_k_pre_rope", False),
            spin_lr=getattr(args, "spin_lr", 1e-2),
            spin_steps=getattr(args, "spin_steps", 200),
            spin_batch_size=getattr(args, "spin_batch_size", 1),
            spin_log_every=getattr(args, "spin_log_every", 10),
            spin_overwrite=getattr(args, "spin_overwrite", False),
            # --- modified by lyy 12.18 ---
            action_head=action_head,
            # --- end modified by lyy 12.18 ---
            seed=getattr(args, "seed", 0),
        )
        # spinquantv2: only pass v2-only kwargs when enabled.
        if spin_quantv2:
            entry_kwargs.update(
                spin_v2_r2_mse_weight=getattr(args, "spin_v2_r2_mse_weight", 1.0),
                # xiugai_lyy: choose recon loss mode (mse|mae) for SpinQuantV2
                spin_v2_recon_loss_mode=args.spin_v2_recon_loss_mode,
                spin_v2_modality_weighting=getattr(args, "spin_v2_modality_weighting", True),
                spin_v2_weight_keys=getattr(args, "spin_v2_weight_keys", ""),
                spin_v2_grad_dump_path=getattr(args, "spin_v2_grad_dump_path", None),
                spin_v2_grad_hook_target=getattr(args, "spin_v2_grad_hook_target", "linear"),
                spin_v2_grad_hook_scope=getattr(args, "spin_v2_grad_hook_scope", "proj_linear"),
            )
        model = entry_fn(**entry_kwargs)
    # --- end modified by lyy 12.18 ---
    elif False and args.method == "dfrot":  # disabled (method not in this repo)
        # Optional dependency: keep import lazy so this repo can run without dfrot files present.
        try:
            pass  # dfrot disabled in this checkout
        except Exception as e:  # noqa: BLE001
            raise ModuleNotFoundError(
                "Requested method=dfrot, but `qvla/methods/dfrot` is not available in this checkout."
            ) from e
        model = dfrot_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_dfrot_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            w_group=args.w_group,
            dfrot_rotate=getattr(args, "dfrot_rotate", True),
            dfrot_init_mode=getattr(args, "dfrot_init_mode", "hadamard"),
            dfrot_alpha=getattr(args, "dfrot_alpha", 100.0),
            dfrot_alter_steps=getattr(args, "dfrot_alter_steps", 100),
            dfrot_procrustes_steps=getattr(args, "dfrot_procrustes_steps", 1),
            dfrot_bits=getattr(args, "dfrot_bits", 4),
            dfrot_clip_ratio=getattr(args, "dfrot_clip_ratio", 1.0),
            dfrot_massive_threshold=getattr(args, "dfrot_massive_threshold", 8.5),
            dfrot_calib_samples=getattr(args, "dfrot_calib_samples", 16),
            dfrot_max_tokens=getattr(args, "dfrot_max_tokens", 16384),
            dfrot_log_every=getattr(args, "dfrot_log_every", 10),
            dfrot_overwrite=getattr(args, "dfrot_overwrite", False),
            dfrot_fp32_had=getattr(args, "dfrot_fp32_had", False),
            dfrot_w_asym=getattr(args, "dfrot_w_asym", False),
            dfrot_a_asym=getattr(args, "dfrot_a_asym", False),
            dfrot_w_rtn=getattr(args, "dfrot_w_rtn", False),
            dfrot_gptq_percdamp=getattr(args, "dfrot_gptq_percdamp", 0.01),
            dfrot_gptq_blocksize=getattr(args, "dfrot_gptq_blocksize", 128),
            dfrot_gptq_actorder=getattr(args, "dfrot_gptq_actorder", False),
            dfrot_w_clip=getattr(args, "dfrot_w_clip", False),
            dfrot_a_clip_ratio=getattr(args, "dfrot_a_clip_ratio", 1.0),
            dfrot_int8_down_proj=getattr(args, "dfrot_int8_down_proj", False),
            k_bit=getattr(args, "k_bit", 16),
            v_bit=getattr(args, "v_bit", 16),
            k_group=getattr(args, "k_group", -1),
            v_group=getattr(args, "v_group", -1),
            dfrot_k_asym=getattr(args, "dfrot_k_asym", False),
            dfrot_v_asym=getattr(args, "dfrot_v_asym", False),
            dfrot_k_clip_ratio=getattr(args, "dfrot_k_clip_ratio", 1.0),
            dfrot_v_clip_ratio=getattr(args, "dfrot_v_clip_ratio", 1.0),
            dfrot_disable_qk_rotation=getattr(args, "dfrot_disable_qk_rotation", False),
            dfrot_o_proj_head_dim_group=getattr(args, "dfrot_o_proj_head_dim_group", False),
            dfrot_separate_fwd=getattr(args, "dfrot_separate_fwd", False),
            action_head=action_head,
            seed=getattr(args, "seed", 0),
        )
    elif False and args.method == "ostquant":  # disabled (method not in this repo)
        # --- modified by lyy ---
        # OSTQuant needs the external OpenVLA action head for action-level distillation losses.
        model = ostquant_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            args,
            action_head=action_head,
        )
        # --- end modified by lyy ---
    elif False and args.method == "omniquant":  # disabled (method not in this repo)
        model = omniquant_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            args,
        )
    else:
        raise NotImplementedError

    return model
