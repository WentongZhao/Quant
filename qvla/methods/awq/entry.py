import torch

from qvla.methods.awq.quantize.pre_quant import run_awq, apply_awq
from qvla.methods.awq.quantize.quantizer import pseudo_quantize_model_weight


# --- modified by lyy 12.20 ---
def awq_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_awq_process: bool,
    scale_path: str = None,
    zero_point: str = True,
    q_group_size: int = 128,
    w_bit: int = 4,
):
# --- end modified by lyy 12.20 ---
    '''
    model: LLM (OpenVLAForActionPrediction or language model).
    prompt_inputs/prompt_kwargs: calibration inputs from get_libero_calib_dataset.
    run_awq_process: whether to run the AWQ process
    '''
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    assert scale_path is not None

    # reparameterization
    if run_awq_process:
        awq_results = run_awq(
            model,
            prompt_inputs,
            prompt_kwargs,
            w_bit=w_bit,
            q_config=q_config,
            auto_scale=True,
        )
        torch.save(awq_results, scale_path)
        print("AWQ results saved at", scale_path)
    else:
        awq_results = torch.load(scale_path, map_location="cpu")
        apply_awq(model, awq_results)

    # weight quantization
    pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)
    # --- modified by lyy 12.20 ---
    # Ensure quantized model is on GPU for downstream eval.
    model.cuda()
    # --- end modified by lyy 12.20 ---
        
    return model
