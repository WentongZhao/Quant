"""
get_libero.py

Calibration-set builder for OpenVLA-OFT that mirrors the *official* training preprocessing.

Key contract for PTQ:
- Returns `(prompt_inputs, prompt_kwargs)` where `prompt_inputs["inputs_embeds"]` is the multimodal embedding
  sequence consumed by the language model, and `prompt_kwargs` includes `labels`, `attention_mask`,
  and modality masks (`vision_mask`, `caption_mask`, `vision_patch_mask`, `proprio_mask`, `action_mask`, `text_mask`).
- For L1 regression gradient collection, `prompt_kwargs["ground_truth_actions"]` equals the training ground-truth
  `batch["actions"]` (shape: (B, NUM_ACTIONS_CHUNK, ACTION_DIM)).

Sampling knobs (`calib_sampling`, `calib_prompt_balance`) only choose which transitions are included; they do not
change the preprocessing / normalization behavior.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, IGNORE_INDEX, NUM_ACTIONS_CHUNK, STOP_INDEX
from prismatic.vla.datasets.rlds.dataset import make_single_dataset
from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs


def _split_segments(length: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    third = max(length // 3, 1)
    segs = [(0, min(third, length)), (min(third, length), min(2 * third, length)), (min(2 * third, length), length)]
    return [seg for seg in segs if seg[0] < seg[1]]


def _pick_representative(indices: list[int], values: np.ndarray, rng: random.Random) -> int:
    vals = values[indices]
    p40 = np.quantile(vals, 0.4)
    p60 = np.quantile(vals, 0.6)
    cand = [idx for idx in indices if p40 <= values[idx] <= p60]
    if cand:
        return rng.choice(cand)
    median = np.quantile(vals, 0.5)
    return indices[int(np.argmin(np.abs(vals - median)))]


def _pick_high(indices: list[int], values: np.ndarray, rng: random.Random, top_k: int = 3, q: float = 0.9) -> int:
    vals = values[indices]
    thresh = np.quantile(vals, q)
    cand = [idx for idx in indices if values[idx] >= thresh]
    if not cand:
        return indices[int(np.argmax(vals))]
    cand = sorted(cand, key=lambda i: values[i])
    if len(cand) > top_k:
        cand = cand[-top_k:]
    return rng.choice(cand)


def _select_indices_random(effective_len: int, rng: random.Random) -> list[int]:
    if effective_len <= 0:
        return []
    return [rng.randint(0, effective_len - 1)]


def _select_indices_proprio_stratified(
    image_delta: np.ndarray,
    proprio_delta: np.ndarray,
    effective_len: int,
    rng: random.Random,
) -> list[int]:
    image_delta = image_delta[:effective_len]
    proprio_delta = proprio_delta[:effective_len]
    indices: list[int] = []

    for start, end in _split_segments(effective_len):
        seg_indices = list(range(start, end))
        rep = _pick_representative(seg_indices, proprio_delta, rng)
        high_prop = _pick_high(seg_indices, proprio_delta, rng)
        high_img = _pick_high(seg_indices, image_delta, rng)
        for idx in (rep, high_prop, high_img):
            if idx not in indices:
                indices.append(idx)

    if effective_len > 0:
        p20_prop = np.quantile(proprio_delta, 0.2)
        p80_prop = np.quantile(proprio_delta, 0.8)
        p20_img = np.quantile(image_delta, 0.2)
        p80_img = np.quantile(image_delta, 0.8)

        mismatch_img = [i for i in range(effective_len) if image_delta[i] >= p80_img and proprio_delta[i] <= p20_prop]
        mismatch_prop = [i for i in range(effective_len) if proprio_delta[i] >= p80_prop and image_delta[i] <= p20_img]
        if mismatch_img:
            idx = rng.choice(mismatch_img)
            if idx not in indices:
                indices.append(idx)
        if mismatch_prop:
            idx = rng.choice(mismatch_prop)
            if idx not in indices:
                indices.append(idx)

    return sorted(indices)


def _normalize_instruction(s: object) -> str:
    return str(s).strip().lower()


def _derive_tfds_paths(data_path: str) -> tuple[Path, str]:
    """
    Expect TFDS-on-disk layout: <root>/<dataset_name>/<version>/dataset_info.json
    Returns (data_root_dir, dataset_name).
    """
    version_dir = Path(os.path.abspath(os.path.expanduser(data_path)))
    if not (version_dir / "dataset_info.json").exists():
        raise FileNotFoundError(f"data_path must be a TFDS version dir containing dataset_info.json: {version_dir}")
    dataset_dir = version_dir.parent
    data_root_dir = dataset_dir.parent
    return data_root_dir, dataset_dir.name


@torch.no_grad()
def get_libero_calib_dataset(
    data_path: str,
    model,
    processor: Optional[AutoProcessor],
    proprio_projector=None,
    n_samples: int = 128,
    rotate: bool = False,
    shuffle: bool = True,
    unnorm_key: Optional[str] = None,
    calib_sampling: str = "random_single",
    calib_prompt_balance: bool = True,
    seed: Optional[int] = None,
    return_mode: str = "embeds",
    return_debug: bool = False,
):
    """
    Build a calibration set using the official RLDS pipeline (same preprocessing as training).

    Args:
        data_path: TFDS version directory, e.g. `.../libero_spatial_no_noops/1.0.0`.
        model: OpenVLAForActionPrediction.
        processor: HuggingFace AutoProcessor for the model.
        proprio_projector: Projector module; used only to decide whether proprio token is inserted.
        rotate: Extra 180° rotation (NOT part of official preprocessing; keep False for strict alignment).
        shuffle: Controls TFDS file read order (set False for deterministic iteration).
        unnorm_key: Ignored here (kept only for API compatibility with existing scripts).
        calib_sampling: "random_single" or "proprio_stratified".
        calib_prompt_balance: If True, enforce uniform prompt coverage over the sampled set.
        seed: Seed for sampling (python RNG).
        return_mode: "embeds" returns LLM-ready inputs_embeds (legacy behavior);
            "raw" returns raw multimodal tensors for vision calibration.
    """
    del unnorm_key  # not used in this official-pipeline builder

    if processor is None:
        raise ValueError("processor is required to build calibration inputs")
    if return_mode not in {"embeds", "raw"}:
        raise ValueError(f"Unsupported return_mode={return_mode!r}; expected 'embeds' or 'raw'.")

    rng = random.Random(seed) if seed is not None else random
    data_root_dir, dataset_name = _derive_tfds_paths(data_path)

    dataset_kwargs = make_oxe_dataset_kwargs(
        dataset_name,
        data_root_dir,
        load_camera_views=("primary", "wrist"),
        load_depth=False,
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
    )
    dataset_kwargs["shuffle"] = bool(shuffle)

    traj_transform_kwargs = dict(
        window_size=1,
        future_action_window_size=NUM_ACTIONS_CHUNK - 1,
        skip_unlabeled=True,
        goal_relabeling_strategy="uniform",
    )
    # RLDS frame resize expects an (H, W) tuple. OpenVLA stores per-backbone resolutions (e.g., [224] or [224, 224]),
    # so we derive a square resize size from the processor / model config.
    resize_hw = None
    image_processor = getattr(processor, "image_processor", None)
    input_sizes = getattr(image_processor, "input_sizes", None)
    if (
        isinstance(input_sizes, list)
        and input_sizes
        and isinstance(input_sizes[0], (list, tuple))
        and len(input_sizes[0]) >= 3
    ):
        resize_hw = (int(input_sizes[0][-2]), int(input_sizes[0][-1]))
    else:
        image_sizes = getattr(getattr(model, "config", None), "image_sizes", None)
        if isinstance(image_sizes, (list, tuple)) and len(image_sizes) > 0:
            size = int(image_sizes[0])
            resize_hw = (size, size)
    if resize_hw is None:
        raise RuntimeError("Failed to infer image resize resolution for RLDS calibration dataset.")
    frame_transform_kwargs = dict(
        resize_size=resize_hw,
        num_parallel_calls=16,
    )

    dataset, num_trajectories, _dataset_statistics = make_single_dataset(
        dataset_kwargs,
        train=True,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
    )

    class CalibrationRLDSBatchTransform:
        def __init__(self):
            self.action_tokenizer = ActionTokenizer(processor.tokenizer)
            self.base_tokenizer = processor.tokenizer
            self.image_transform = processor.image_processor.apply_transform
            self.prompt_builder_fn = PurePromptBuilder
            self.predict_stop_token = True
            self.use_wrist_image = int(model.vision_backbone.get_num_images_in_input()) > 1
            self.use_proprio = proprio_projector is not None

        def __call__(self, rlds_batch: dict) -> dict:
            ds_name = rlds_batch["dataset_name"]
            current_action = rlds_batch["action"][0]
            img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
            if rotate:
                img = img.rotate(180)
            lang = rlds_batch["task"]["language_instruction"].decode().lower()
            actions = rlds_batch["action"]

            prompt_builder = self.prompt_builder_fn("openvla")
            future_actions = rlds_batch["action"][1:]
            future_actions_string = "".join(self.action_tokenizer(future_actions))
            current_action_string = self.action_tokenizer(current_action)
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)

            input_ids = torch.tensor(input_ids)
            labels = torch.tensor(labels)
            pixel_values = self.image_transform(img)

            labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
            if not self.predict_stop_token:
                labels[-1] = IGNORE_INDEX

            out = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=ds_name, actions=actions)

            if self.use_wrist_image:
                all_wrist_pixels = []
                for k in rlds_batch["observation"].keys():
                    if "wrist" in k:
                        img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                        if rotate:
                            img_wrist = img_wrist.rotate(180)
                        all_wrist_pixels.append(self.image_transform(img_wrist))
                if all_wrist_pixels:
                    out["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)

            if self.use_proprio and "proprio" in rlds_batch["observation"]:
                out["proprio"] = rlds_batch["observation"]["proprio"]

            out["instruction"] = lang.strip().lower()
            return out

    batch_transform = CalibrationRLDSBatchTransform()

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    instances: list[dict] = []
    debug_traj_ids: list[int] = []
    debug_timesteps: list[int] = []
    debug_prompts: list[str] = []
    debug_dataset_names: list[str] = []
    prompt_targets: dict[str, int] | None = None
    prompt_counts: dict[str, int] | None = None

    if calib_prompt_balance:
        # Compute targets from dataset-level unique prompts, then fill them by scanning trajectories until satisfied.
        prompts = set()
        scanned_prompts = 0
        for traj in dataset.as_numpy_iterator():
            if scanned_prompts >= int(num_trajectories):
                break
            lang_bytes = traj["task"]["language_instruction"][0]
            if isinstance(lang_bytes, (bytes, bytearray)):
                instr = _normalize_instruction(lang_bytes.decode("utf-8"))
            else:
                instr = _normalize_instruction(lang_bytes)
            if instr:
                prompts.add(instr)
            scanned_prompts += 1

        unique_prompts = sorted(prompts)
        if not unique_prompts:
            raise RuntimeError("Failed to collect any language instructions for prompt-balanced sampling.")
        if n_samples < len(unique_prompts):
            raise RuntimeError(
                f"n_samples ({n_samples}) is smaller than number of prompts ({len(unique_prompts)}); cannot guarantee coverage."
            )
        base = n_samples // len(unique_prompts)
        rem = n_samples % len(unique_prompts)
        order = unique_prompts[:]
        rng.shuffle(order)
        prompt_targets = {p: base for p in unique_prompts}
        for p in order[:rem]:
            prompt_targets[p] += 1
        prompt_counts = {k: 0 for k in prompt_targets}

    scanned = 0
    for traj in dataset.as_numpy_iterator():
        if scanned >= int(num_trajectories):
            break

        traj_len = int(traj["action"].shape[0])
        if traj_len <= 0:
            scanned += 1
            continue

        if prompt_targets is not None:
            traj_lang_bytes = traj["task"]["language_instruction"][0]
            if isinstance(traj_lang_bytes, (bytes, bytearray)):
                traj_instr = _normalize_instruction(traj_lang_bytes.decode("utf-8"))
            else:
                traj_instr = _normalize_instruction(traj_lang_bytes)
            if traj_instr and prompt_counts[traj_instr] >= prompt_targets[traj_instr]:
                scanned += 1
                continue

        effective_len = traj_len
        if calib_sampling == "proprio_stratified":
            if proprio_projector is None:
                raise RuntimeError("proprio_stratified requires proprio_projector (use_proprio=True).")

            prim = traj["observation"]["image_primary"][:, 0]  # (T, H, W, 3) uint8
            wrist = traj["observation"].get("image_wrist", None)
            wrist = wrist[:, 0] if wrist is not None else None

            img_delta = np.zeros(effective_len, dtype=np.float32)
            for t in range(1, effective_len):
                d0 = float(np.mean(np.abs(prim[t].astype(np.float32) - prim[t - 1].astype(np.float32))))
                if wrist is not None:
                    d1 = float(np.mean(np.abs(wrist[t].astype(np.float32) - wrist[t - 1].astype(np.float32))))
                    img_delta[t] = 0.5 * d0 + 0.5 * d1
                else:
                    img_delta[t] = d0

            proprio_arr = traj["observation"]["proprio"][:, 0].astype(np.float32)
            prop_delta = np.zeros(effective_len, dtype=np.float32)
            if effective_len > 1:
                diffs = np.linalg.norm(proprio_arr[1:] - proprio_arr[:-1], axis=1)
                prop_delta[1:] = diffs.astype(np.float32)

            indices = _select_indices_proprio_stratified(img_delta, prop_delta, effective_len, rng)
        else:
            indices = _select_indices_random(effective_len, rng)

        for idx in indices:
            if len(instances) >= int(n_samples):
                break
            step = tree_map(lambda x: x[idx], traj)
            ex = batch_transform(step)

            if prompt_targets is not None:
                instr = _normalize_instruction(ex.get("instruction", ""))
                if not instr or instr not in prompt_targets:
                    continue
                if prompt_counts[instr] >= prompt_targets[instr]:
                    continue
                prompt_counts[instr] += 1

            instances.append(ex)
            debug_traj_ids.append(int(scanned))
            debug_timesteps.append(int(idx))
            debug_prompts.append(_normalize_instruction(ex.get("instruction", "")))
            ds_name = ex.get("dataset_name", "")
            if isinstance(ds_name, (bytes, bytearray)):
                ds_name = ds_name.decode("utf-8")
            debug_dataset_names.append(str(ds_name))

        scanned += 1
        if len(instances) >= int(n_samples):
            if prompt_targets is None:
                break
            unmet = {k: (prompt_counts[k], prompt_targets[k]) for k in prompt_targets if prompt_counts[k] < prompt_targets[k]}
            if not unmet:
                break

    if len(instances) < int(n_samples):
        if prompt_targets is not None:
            unmet = {k: (prompt_counts[k], prompt_targets[k]) for k in prompt_targets if prompt_counts[k] < prompt_targets[k]}
            raise RuntimeError(f"Prompt-balanced sampling underfilled targets: {unmet}")
        raise RuntimeError(f"Underfilled calibration set: collected {len(instances)} / {n_samples} samples.")

    batch = collator(instances[: int(n_samples)])
    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    proprio = batch.get("proprio", None)
    if proprio is not None and proprio_projector is not None:
        proj_dtype = next(proprio_projector.parameters()).dtype
        proprio = proprio.to(device=device, dtype=proj_dtype)

    if return_mode == "raw":
        prompt_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        prompt_kwargs = {
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if proprio is not None and proprio_projector is not None:
            prompt_kwargs["proprio"] = proprio

        if return_debug:
            debug = {
                "dataset_name": debug_dataset_names[: int(n_samples)],
                "traj_index": debug_traj_ids[: int(n_samples)],
                "timestep": debug_timesteps[: int(n_samples)],
                "instruction": debug_prompts[: int(n_samples)],
            }
            return prompt_inputs, prompt_kwargs, debug
        return prompt_inputs, prompt_kwargs

    # Multimodal embedding construction mirrors `OpenVLAForActionPrediction.forward` (non-diffusion path).
    input_embeddings = model.get_input_embeddings()(input_ids)  # (B, S, D)
    all_actions_mask = model._process_action_masks(labels)  # (B, S)
    input_embeddings = input_embeddings * (~all_actions_mask.unsqueeze(-1))
    language_embeddings = input_embeddings[~all_actions_mask].reshape(input_embeddings.shape[0], -1, input_embeddings.shape[2])

    projected_patch_embeddings = model._process_vision_features(pixel_values, language_embeddings, use_film=False)
    projected_patch_embeddings = model._process_proprio_features(projected_patch_embeddings, proprio, proprio_projector)

    multimodal_embeddings, multimodal_attention_mask = model._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )
    multimodal_labels = model._build_multimodal_labels(labels, projected_patch_embeddings)

    num_patches = projected_patch_embeddings.shape[1]
    vision_mask = torch.zeros_like(multimodal_attention_mask, dtype=torch.bool)
    vision_mask[:, 1 : 1 + num_patches] = True
    caption_mask = multimodal_labels.ne(IGNORE_INDEX)

    has_proprio_token = proprio is not None and proprio_projector is not None
    proprio_mask = torch.zeros_like(multimodal_attention_mask, dtype=torch.bool)
    if has_proprio_token:
        proprio_mask[:, 1 + (num_patches - 1)] = True
    vision_patch_mask = vision_mask & (~proprio_mask)

    stop_mask = multimodal_labels.eq(STOP_INDEX)
    action_mask = caption_mask & (~stop_mask)
    text_mask = multimodal_attention_mask.to(torch.bool) & (~vision_mask) & multimodal_labels.eq(IGNORE_INDEX)

    prompt_inputs = {"inputs_embeds": multimodal_embeddings}
    prompt_kwargs = {
        "labels": multimodal_labels,
        "attention_mask": multimodal_attention_mask,
        "vision_mask": vision_mask,
        "caption_mask": caption_mask,
        "vision_patch_mask": vision_patch_mask,
        "proprio_mask": proprio_mask,
        "action_mask": action_mask,
        "text_mask": text_mask,
        "ground_truth_actions": batch["actions"][: int(n_samples)],
    }
    if return_debug:
        debug = {
            "dataset_name": debug_dataset_names[: int(n_samples)],
            "traj_index": debug_traj_ids[: int(n_samples)],
            "timestep": debug_timesteps[: int(n_samples)],
            "instruction": debug_prompts[: int(n_samples)],
        }
        return prompt_inputs, prompt_kwargs, debug
    return prompt_inputs, prompt_kwargs


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be imported by qvla/main_quant.py. "
        "Invoke that script to build calibration inputs."
    )
