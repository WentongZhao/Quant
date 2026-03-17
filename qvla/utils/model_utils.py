def is_openvla(model) -> bool:
    return model.__class__.__name__ == "OpenVLAForActionPrediction"


def get_llm_model(model):
    """Return the LLM wrapper to quantize (OpenVLA -> language_model)."""
    return model.language_model if is_openvla(model) else model


def get_llm_backbone(model):
    """Return the backbone module where transformer blocks live (e.g., LlamaModel)."""
    llm = get_llm_model(model)
    return llm.model if hasattr(llm, "model") else llm
