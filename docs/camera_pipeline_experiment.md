# Camera-aware pi0 — experiment runbook

Companion to [`camera_pipeline_review.md`](./camera_pipeline_review.md). That doc covers what was broken and why the fixes are correct. This one covers **what was added**, **what to watch out for when running**, and **end-to-end commands**.

Target model: a pi3-inspired camera-conditioning variant of pi0 (PyTorch path), combining:

- **Ray embedding** from intrinsics (`K⁻¹·pix` → 2-channel `(x/z, y/z)` → zero-init `PatchEmbed` → added to SigLIP patch tokens **before** the SigLIP transformer).
- **PRoPE** extrinsic-aware attention injected after the `CrossViewFusion` global block.
- **View embedding** per camera (learnable token added to every patch of a view).
- **CrossViewFusion** with `aa_order="fg"` (one frame-attn + one global-attn block).

The dataset is LIBERO-Camera: modified LIBERO HDF5 episodes that include per-camera intrinsics in `obs.attrs` and per-frame extrinsics in `obs/{agent,wrist}_extrinsic`.

---

## 1. What was added

### 1.1 New / extended model knobs (`src/openpi/models/pi0_config.py`)

| field | type | default | meaning |
|---|---|---|---|
| `pose_enc_type` | `"null"\|"relative_pose"\|"absolute_pose"\|"prope"` | `"null"` | how extrinsics enter the model |
| `ray_enc_type` | `bool` | `False` | enable pre-SigLIP `K⁻¹·pix` ray embedding |
| `view_enc_type` | `bool` | `False` | enable learnable per-view embedding |
| `cross_view` | `CrossViewFusionConfig` | default | see below |

`CrossViewFusionConfig` (`src/openpi/models/cross_view_config.py`) now has:

- `type: "none" | "simple" | "standard"`
- `aa_order: str` (e.g. `"fg"`, `"fgfg"`) — which attention block types, in order.
- `prope_layer_idx: tuple[int, ...]` — indices into the `"g"` sub-sequence at which a `PoseInjectBlock` is injected. `CrossViewFusion.__init__` will raise at startup if any index ≥ number of `g` blocks.

### 1.2 New modules (`src/openpi/models_pytorch/`)

- `gemma_pytorch.PaliGemmaWithExpertModel`: wires `ray_embed` (zero-init `PatchEmbed(in_chans=2)`), `view_embedding`, `cam_pose_encoder`, and `cross_view_fusion` into the vision path.
- `gemma_pytorch.CrossViewFusion`: frame/global block stack with optional PRoPE injection.
- `gemma_pytorch.PoseInjectBlock` + `layers/prope.py`: PRoPE math ported from pi3.
- `layers/point_head.py` + `PI0Pytorch.aux_point_head`: optional pi3x point head (distillation target). Off by default — `Pi0Config.aux_point_head.enabled=False`.
- `transformers_replace/models/siglip/modeling_siglip.py`: patched `SiglipVisionEmbeddings.forward` reads `_pending_ray_emb` attribute and adds it to patch tokens before the transformer encoder.

### 1.3 Data pipeline changes

- **`examples/libero/convert_libero_hdf5_to_lerobot.py`**: new LIBERO-Camera → LeRobot converter. Reads K from HDF5 attrs, scales K from render resolution to LeRobot `image_size`, writes `agent_intrinsic`/`wrist_intrinsic`/`agent_extrinsic`/`wrist_extrinsic` as LeRobot features.
- **`src/openpi/policies/libero_policy.py`**: `_mujoco_to_opencv_extrinsic` (`T @ diag(1,-1,-1,1)`) and `_adjust_K_for_openpi_image_flip` (`fx → -fx`) absorb the converter's `[::-1, ::-1]` image flip into K and T so `K @ T_wc_opencv @ X_world` lands on the correct pixel of the preprocessed image.
- **`src/openpi/models/model.py`**: `Observation` dataclass extended with `agent_extrinsic`, `wrist_extrinsic`, `agent_intrinsic`, `wrist_intrinsic` (all `Optional[Tensor]`). `from_dict` and `preprocess_observation` carry them through.
- **`src/openpi/models_pytorch/preprocessing_pytorch.py`**: composes pixel-space affine transforms from resize-with-pad, random crop, and random rotation, then left-multiplies them into K so intrinsics match the augmented image that reaches the model.
- **`src/openpi/training/config.py`**: `LeRobotLiberoDataConfig.include_cam_extrinsics: bool` — when True, repacks the four camera fields alongside state/action/images.

### 1.4 New train config

**`pi0_libero_cam_pytorch_prope_ray_view`** (in `_CONFIGS`, right after the `_standard` config). Bakes in:

```
pose_enc_type="prope",
ray_enc_type=True,
view_enc_type=True,
cross_view=CrossViewFusionConfig(
    type="standard",
    aa_order="fg",
    prope_layer_idx=(0,),
),
include_cam_extrinsics=True,
pytorch_weight_path=<LOCAL_GEO_ROOT>/pi0_base,
num_train_steps=30_000,
```

---

## 2. Things to watch out for

### 2.1 ⚠️ Dataset image size must equal the model input (224)

`ModelTransformFactory` unconditionally applies `ResizeImages(224, 224)` (`src/openpi/training/config.py:136,148,168`). `ResizeImages` **does not update intrinsics** (`src/openpi/transforms.py:185-191`). The in-model `preprocess_observation_pytorch:214` only re-scales K when it actually resizes the image — if the image is already 224×224, it skips that branch.

**Consequence:** if you generate `glbreeze/libero_cam` at `image_size=256`, K will be scaled for 256 but images fed to the model are 224. `fx,fy` off by `256/224 ≈ 1.143×`, principal point off by ~16 pixels. Ray embeddings and PRoPE K-normalization will both be wrong.

**Fix:** always convert with `--image-size 224` (see §3.1).

### 2.2 `ray_embed` must be in `OPENPI_TRAINABLE_PREFIXES`

`ray_embed.proj.{weight,bias}` are zero-initialized on purpose (identity-on-init, so the pretrained backbone is unperturbed at step 0). If the trainable-prefix allowlist doesn't include it, gradients are masked off and it stays at zero forever — silently disabled.

Always export:
```bash
export OPENPI_TRAINABLE_PREFIXES="paligemma_with_expert.cross_view_fusion,paligemma_with_expert.cam_pose_encoder,paligemma_with_expert.view_embedding,paligemma_with_expert.ray_embed"
```
Already set in `scripts/run.sh:46` and `scripts/sbatch/train_pi0_libero_object_ft.sbatch:79`.

### 2.3 `prope_layer_idx` must be valid for `aa_order`

`aa_order="fg"` has exactly one `g` block, so `prope_layer_idx=(0,)` is the only valid choice. `CrossViewFusion.__init__` raises a `ValueError` at startup if you pass `(0, 1)`. To inject PRoPE at more than one depth, extend `aa_order` (e.g. `"fgfg"` and `prope_layer_idx=(0, 1)`).

### 2.4 `cross_view.type="simple"` silently ignores `pose_enc_type="prope"`

`SimpleCrossViewFusion` doesn't accept a `poses` argument; if you pair `type="simple"` with `pose_enc_type="prope"`, PRoPE is a no-op. Use `type="standard"` whenever `pose_enc_type="prope"`.

### 2.5 Coordinate-frame conventions

- **Dataset.** `agent_extrinsic` / `wrist_extrinsic` are raw MuJoCo camera-to-world (C2W) in OpenGL frame (x-right, y-up, z-back). `agent_intrinsic` / `wrist_intrinsic` are OpenCV pinhole `K` at LeRobot image resolution.
- **Entering the model (after `LiberoInputs`).** Extrinsics are OpenCV-convention C2W (post `T @ diag(1,-1,-1,1)`). Intrinsics have `fx < 0` (post `fx → -fx`).
- **Inside `PoseInjectBlock`.** `poses` are the OpenCV C2W above; the block inverts to W2C before calling `prope.prepare_apply_fns` (which expects `camera<-world`, i.e. W2C).

These are internally consistent. If you ever change the image flip in `_preprocess_image`, you must update `_adjust_K_for_openpi_image_flip` and `_mujoco_to_opencv_extrinsic` in lockstep.

### 2.6 Missing K / identity fallback

LIBERO has two cameras; `right_wrist` is a zero-padded stand-in. For that view:

- Image mask is `False`.
- `cam_pos["right_wrist"] = None`, `cam_intr["right_wrist"] = None` in `embed_prefix`.
- `embed_image` fills identity for missing extrinsic and identity for missing K, but:
  - Ray embed zeroes the per-view embedding for views with `K=None` (`gemma_pytorch.py:599-605`), so gradient cannot leak back to `ray_embed` via fake-identity-K views.
  - PRoPE attention masks the view as keys AND multiplies the residual by `view_valid` so fabricated poses don't drive gradient (`gemma_pytorch.py:326-349`).

If you add a third real camera later, update `_IMAGE_TO_CAMERA_FIELDS` in `preprocessing_pytorch.py:19-23` and the `cam_pos` / `cam_intr` dict construction in `pi0_pytorch.py:262-268`.

### 2.7 Norm stats

`compute_norm_stats.py` normalizes only `state` and `actions` (camera fields are intentionally skipped — can't normalize an SE(3) matrix component-wise). You need to recompute stats **only when the repo_id changes**, not when the model architecture changes. Stats generated for `pi0_libero_cam` are reusable by `pi0_libero_cam_pytorch_prope_ray_view` as long as the data transforms match.

### 2.8 Fine-tuning from `pi0_base`

The new config loads from `<LOCAL_GEO_ROOT>/pi0_base` via `pytorch_weight_path`. New modules (`cross_view_fusion`, `view_embedding`, `cam_pose_encoder`, `ray_embed`, `pose_inject_blocks`) are allowed-missing keys (`scripts/train_pytorch.py:631-636`). Pre-trained SigLIP / Gemma weights load into unchanged modules; new heads start fresh.

### 2.9 `torch.compile` interactions

`sample_actions` is `torch.compile`d at `__init__` unless `cross_view.type != "none"`, in which case compile is skipped (`pi0_pytorch.py:140-152`). This is intentional — the PRoPE path uses cached RoPE tensors that trip TorchDynamo. Training always runs uncompiled.

### 2.10 DDP `find_unused_parameters`

`find_unused_parameters=True` is set in `train_pytorch.py:612`. Required because (a) `ray_embed` is inactive when `cam_intr is None`, and (b) `PoseInjectBlock` is only used after the global block fires. Do not turn it off.

---

## 3. How to run the experiment

### 3.1 Build the LIBERO-Camera LeRobot dataset at 224

Prerequisite: LIBERO-Camera HDF5 files (see `$GEO_ROOT/LIBERO-Camera/scripts/create_dataset.py`), each with:

- `data/<ep>/obs/agentview_rgb`, `eye_in_hand_rgb` at the rendered resolution (typically 128).
- `data/<ep>/obs/agent_extrinsic`, `wrist_extrinsic` of shape `(T, 4, 4)` (MuJoCo C2W, per frame).
- `data/<ep>/obs.attrs["agent_intrinsic"]`, `"wrist_intrinsic"` of shape `(3, 3)`.
- `data/<ep>/obs.attrs["agent_image_size"]`, `"wrist_image_size"` = render `(H, W)`.

Run the converter at the model's native input resolution (**always 224**):

```bash
python examples/libero/convert_libero_hdf5_to_lerobot.py \
    --dataset-root $GEO_ROOT/LIBERO-Camera/datasets/libero_object_camvar \
    --repo-id glbreeze/libero_cam \
    --mode all_views \
    --image-size 224
```

The converter will scale `K` from `agent_image_size` → 224 automatically (`_scale_intrinsic`).

### 3.2 Compute norm stats

```bash
python scripts/compute_norm_stats.py --config-name pi0_libero_cam_pytorch_prope_ray_view
```

Writes `<LOCAL_GEO_ROOT>/pi0_libero/glbreeze/libero_cam/norm_stats.json`. Reusable across configs that share this repo_id + data transforms.

### 3.3 Smoke test (2 steps, batch 2)

Confirms the pipeline runs end to end before burning GPU hours:

```bash
export OPENPI_TRAINABLE_PREFIXES="paligemma_with_expert.cross_view_fusion,paligemma_with_expert.cam_pose_encoder,paligemma_with_expert.view_embedding,paligemma_with_expert.ray_embed"

python scripts/train_pytorch.py pi0_libero_cam_pytorch_prope_ray_view \
    --exp_name smoke \
    --batch_size 2 \
    --num_train_steps 2
```

Success criteria:
- No import error about `transformers_replace` — if you see it, reinstall per the README's "PyTorch Support" section.
- No `ValueError` from `CrossViewFusion.__init__` about `prope_layer_idx`.
- Two gradient steps complete with finite loss.
- Allowed-missing keys logged on checkpoint load: `cross_view_fusion`, `view_embedding`, `cam_pose_encoder`, `ray_embed`, `pose_inject_blocks`. No *unexpected* missing keys.

### 3.4 Full single-node run

```bash
export OPENPI_TRAINABLE_PREFIXES="paligemma_with_expert.cross_view_fusion,paligemma_with_expert.cam_pose_encoder,paligemma_with_expert.view_embedding,paligemma_with_expert.ray_embed"

python scripts/train_pytorch.py pi0_libero_cam_pytorch_prope_ray_view \
    --exp_name pi3x_fg_run \
    --batch_size 8
```

### 3.5 Multi-GPU (torchrun)

```bash
export OPENPI_TRAINABLE_PREFIXES="paligemma_with_expert.cross_view_fusion,paligemma_with_expert.cam_pose_encoder,paligemma_with_expert.view_embedding,paligemma_with_expert.ray_embed"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi0_libero_cam_pytorch_prope_ray_view \
    --exp_name pi3x_fg_run \
    --batch_size 8
```

Resume with `--resume` (picks up the latest checkpoint under `checkpoints/<config_name>/<exp_name>/`).

### 3.6 Serving / inference

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_libero_cam_pytorch_prope_ray_view \
    --policy.dir=checkpoints/pi0_libero_cam_pytorch_prope_ray_view/pi3x_fg_run/<step>
```

The inference-time LIBERO environment must emit `observation/agent_extrinsic`, `observation/wrist_extrinsic`, `observation/agent_intrinsic`, `observation/wrist_intrinsic` (see `examples/libero/main_multicam.py` for the pattern). Without them, `LiberoInputs` skips the four `if "observation/*" in data` branches and the camera-aware branches in `embed_image` fall back to non-camera behavior (ray_emb skipped, PRoPE with identity poses).

---

## 4. Sanity checks to consider before a long run

1. **Coordinate-frame consistency.** Render a known 3D world point (e.g. robot base at `[0,0,0,1]`), project through `K @ T_wc_opencv @ X_world` (with `K` and `T_wc` exactly as they enter the model), and verify the result lands on the expected pixel in the preprocessed (post-`[::-1, ::-1]`) agent-view image. If the pixel is off, something in §2.5 disagrees.
2. **Ray-embed no-op at step 0.** At initialization the zero-init `ray_embed` must produce exactly the same loss as a run with `ray_enc_type=False`. If they differ, the zero-init is broken or something is adding nonzero bias.
3. **Intrinsic post-preprocessing.** Log `observation.agent_intrinsic[0]` at the top of `embed_image` (after `_preprocess_observation`) and confirm `cx ≈ W/2 = 112`, `cy ≈ H/2 = 112`, `fx ≈ -fy` (the negative `fx` is the image-flip absorption from §2.5).
4. **Gradient flow.** After a few steps, confirm `ray_embed.proj.weight.grad.abs().mean() > 0` and `cross_view_fusion.*.grad.abs().mean() > 0`. If gradients are zero, check the trainable-prefix allowlist.

---

## 5. Related docs

- [`camera_pipeline_review.md`](./camera_pipeline_review.md) — what was broken, why each fix is correct, with math and file:line citations.
- [`norm_stats.md`](./norm_stats.md) — how norm stats are generated and reused across configs.
- `scripts/LG.md` — scratchpad of design notes and data-cache sizing tables.
