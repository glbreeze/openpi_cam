# Camera-aware pi0 pipeline — review & fixes

Review target: the pi3-inspired camera-conditioning stack (ViewEmbedding + PRoPE + RayEmbedding + CrossViewFusion) as wired into openpi's PyTorch pi0 backbone for the LIBERO-Camera dataset.

Each fixed problem below lists the original **symptom**, the **ideology** (why the fix is correct — math where relevant), the **code** (what changed, with file pointers), and a **verification** note (sanity check / end-to-end trace). Items still open appear under **LOW — not in scope for this pass** at the bottom.

---

## CRITICAL — training won't run, or runs as a silent no-op

### 1. Five leftover `pdb.set_trace()` calls

**Symptom.** Five interactive breakpoints were live across the hot path:
- `scripts/train_pytorch.py:644` (before `apply_trainable_prefixes`)
- `src/openpi/models_pytorch/pi0_pytorch.py:409` (`PI0Pytorch.forward`, every batch)
- `src/openpi/models_pytorch/gemma_pytorch.py:523` (`PaliGemmaWithExpertModel.embed_image`, every batch)
- `src/openpi/models_pytorch/gemma_pytorch.py:183` (`PoseInjectBlock.forward`)
- `src/openpi/models_pytorch/gemma_pytorch.py:283` (`CrossViewFusion.forward`)

**Ideology.** These are debug breakpoints that were never removed; any one of them halts any training or inference run.

**Code.** All five removed (four were removed by the user, the last by this pass).

**Verification.** `grep -rn "pdb.set_trace" src/openpi/ scripts/train_pytorch.py` returns only one commented-out line in `data_loader.py:326`.

### 2. LIBERO-Camera dataset lacked intrinsics

**Symptom.** `examples/libero/convert_libero_hdf5_to_lerobot.py` declared/wrote only `agent_extrinsic` / `wrist_extrinsic`. The downstream pipeline (`LeRobotLiberoDataConfig` → `LiberoInputs` → `Observation.from_dict` → `embed_image`) silently saw `agent_intrinsic = None`, so `ray_enc_type=True` was a no-op and PRoPE's `prepare_apply_fns` ran with `Ks=None` (losing the K-normalized projmat path).

**Ideology.** MuJoCo exposes a per-camera vertical FOV and render resolution. From those, a standard OpenCV pinhole `K` is
$$K = \begin{bmatrix} f_y & 0 & W/2 \\ 0 & f_y & H/2 \\ 0 & 0 & 1 \end{bmatrix}, \quad f_y = \frac{H/2}{\tan(\text{fovy}/2)}$$
(square pixels, principal point at image center). When the stored image is resized from the render resolution `(src_h, src_w)` to the LeRobot resolution `(dst_h, dst_w)`, `K` updates by $K_\text{dst} = \mathrm{diag}(s_x, s_y, 1) \cdot K_\text{src}$ with $s_x = \text{dst\_w}/\text{src\_w}$, $s_y = \text{dst\_h}/\text{src\_h}$, i.e. each $(f_x, c_x)$ column scales by $s_x$ and each $(f_y, c_y)$ column scales by $s_y$.

**Code.** `convert_libero_hdf5_to_lerobot.py`:
- Added `_scale_intrinsic(K, src_h, src_w, dst_h, dst_w)` that scales `fx, cx` by `dst_w/src_w` and `fy, cy` by `dst_h/src_h`.
- Added `_read_episode_camera_params(obs_group, frame_count, image_size)` that reads per-episode `agent_intrinsic` / `wrist_intrinsic` plus the source `agent_image_size` / `wrist_image_size` from HDF5 attrs, applies `_scale_intrinsic`, and asserts extrinsic shapes.
- Declared `agent_intrinsic` / `wrist_intrinsic` as `(3, 3) float32` features on the LeRobot dataset and writes them per frame.

**Verification.** End-to-end trace after the change: dataset column → `LeRobotLiberoDataConfig.repack_structure` (`observation/agent_intrinsic`) → `LiberoInputs` → `Observation.from_dict` → `pi0_pytorch.embed_prefix.cam_intr` → `embed_image` (non-None). `ray_enc_type=True` now produces a real ray embedding, and PRoPE receives the K-normalized projmat.

### 3. `ray_embed` missing from `OPENPI_TRAINABLE_PREFIXES`

**Symptom.** `ray_embed.proj.{weight,bias}` are zero-initialized on purpose (identity-on-init). If the trainable-prefix filter doesn't include them, their gradient is masked off and they stay at zero forever — another silent no-op, independent of #2.

**Ideology.** The prefix filter is an env-var allowlist of substrings; a module participates in training iff its full parameter name contains at least one listed prefix. For a zero-init module, being absent from the allowlist is equivalent to deletion at training time.

**Code.** Added `paligemma_with_expert.ray_embed` to every place the allowlist is declared:
- `scripts/run.sh:46` (interactive shell default)
- `scripts/sbatch/train_pi0_libero_object_ft.sbatch:79` (cluster default)
- `scripts/sbatch/train_pi0_libero_object_cam_simple_twostage.sbatch:77` (two-stage override)

**Verification.** `grep -rn "paligemma_with_expert\." scripts/` returns all three sites including `ray_embed`. Also consistent with `scripts/train_pytorch.py:631-636` which already whitelists `ray_embed` in `allowed_missing` when loading base weights.

### 4. `view_enc_type` never turned on in the CLI

**Symptom.** `Pi0Config.view_enc_type` defaults to `False`. None of the training commands in `scripts/run.sh` passed `--model.view_enc_type`, so ViewEmbedding was silently off even when the user's notes asserted it was on.

**Ideology.** Tyro interprets a bare boolean flag as `True`. Adding `--model.view_enc_type` on the CLI flips the config field from `False` to `True`, which in turn causes `PaliGemmaWithExpertModel.__init__` to construct `self.view_embedding` and `embed_image` to add it onto the per-view patch tokens.

**Code.** `scripts/run.sh:59, 68` — added `--model.view_enc_type` to both PRoPE+RayEmbed commands. The `absolute_pose + simple` command at line 51 was left alone because that experiment uses the cam-token pathway, not ViewEmbedding.

**Verification.** `grep -n "view_enc_type" scripts/run.sh` shows the flag on both relevant commands.

### 4b. `view_embedding` wasn't identity-on-init (added during this review)

**Symptom.** `nn.Embedding` defaults to `N(0, 1)` per entry. For embedding dim 1152, each per-view vector has L2 norm ≈ √1152 ≈ 34, added to every SigLIP patch token at step 0. Resuming from `pi0_base` would inject a step-1 perturbation several orders of magnitude larger than anything the pretrained backbone was normalized against → loss spike.

**Ideology.** The other new module, `ray_embed`, uses the pi3x "identity-on-init" trick: zero-initialize the projection so the layer is a no-op at step 0 and the pretrained model sees unchanged activations; gradients still escape the zero on the first backward pass. `view_embedding` should follow the same rule.

**Code.** `gemma_pytorch.py:494` — appended `nn.init.zeros_(self.view_embedding.weight)` right after the `nn.Embedding` construction.

**Verification.** At step 0, `view_embedding(view_ids)` returns zeros, so `tokens + view_embed == tokens`. The first backward pass computes a non-zero gradient via the add, allowing the embedding to escape zero over training.

---

## HIGH — geometry correctness bugs

### 5. Intrinsic resolution now tracked through every image transform

**Symptom.** `K` was computed at the render resolution, the converter resized images to 256, the model further resized to 224, and the ray-embed pixel grid was built at 224 — so `K⁻¹ · pix_224` silently mixed a source-resolution `K` with a model-resolution pixel grid.

**Ideology.** An image transform that maps source pixel `p_src` to destination pixel `p_dst = M · p_src` for some 3×3 homography `M` transforms the intrinsic by left-multiplication: `K_dst = M · K_src`. This is the universal "update the intrinsic whenever you warp the image" rule — because `p = K · X_cam` for any point `X_cam`, so `p_dst = M · K_src · X_cam = K_dst · X_cam`.

Three concrete cases matter:
- **Resize-with-pad** from `(src_h, src_w)` → `(dst_h, dst_w)`: uniform scale $s = 1/\max(\text{src\_w}/\text{dst\_w}, \text{src\_h}/\text{dst\_h})$ plus letterbox offset $(\text{pad\_w0}, \text{pad\_h0})$. `M = [[s_x, 0, pad_{w0}], [0, s_y, pad_{h0}], [0, 0, 1]]`.
- **Random crop then resize** from `(input_h, input_w)` cropping `(crop_h, crop_w)` at start `(start_h, start_w)`: scale `scale_{x,y} = input_{w,h}/crop_{w,h}`, translation `(-s_x \cdot \text{start\_w}, -s_y \cdot \text{start\_h})`.
- **Rotation about image center**: `M = N2P \cdot R_\alpha^{-1} \cdot P2N` where `P2N` and `N2P` are the pixel↔normalized conversions; `R_\alpha^{-1}` is the inverse of `grid_sample`'s sampling rotation.

**Code.**
- **Dataset-side scaling** (`convert_libero_hdf5_to_lerobot.py:_scale_intrinsic` + `_read_episode_camera_params`): scales source-resolution `K` to the LeRobot `image_size` before writing.
- **Model-side pixel-transform algebra** (`preprocessing_pytorch.py:58-179`): `_make_pixel_transform`, `_resize_with_pad_pixel_transform`, `_crop_resize_pixel_transform`, `_rotation_pixel_transform`, and `_apply_pixel_transform_to_intrinsic` — composed onto the intrinsic in lockstep with every image transform.
- **Wiring** (`pi0_pytorch.py:201-211, 395, 479`): `_preprocess_observation` now returns the preprocessed observation as its first tuple element; `forward` and `sample_actions` pass it as `obs=` to `embed_prefix`, which reads the scaled `K` via `obs.agent_intrinsic` / `obs.wrist_intrinsic`.

**Verification.** Algebra checks:
- Resize-with-pad: `pixel_dst = diag(s_x, s_y) · pixel_src + (pad_{w0}, pad_{h0})`. `_apply_pixel_transform_to_intrinsic` computes `M @ K`. ✓
- Rotation: `grid_sample` reads source at `(cos α · x − sin α · y, sin α · x + cos α · y)`; the inverse rotation applied to `K` is the composition of normalized↔pixel coord changes around the image center. ✓
- Principal-point convention: `get_intrinsic` uses `c_x = W/2, c_y = H/2`, `get_pixel` uses `u+0.5, v+0.5` (both center-of-pixel); the ray `K⁻¹·\text{pix}` is consistent through all resizes.

One minor open limitation: the train-time *rotation* updates `K` but not the extrinsic. A 2-D rotation about the optical center is equivalent to a rigid rotation of the camera around its optical axis; strictly you'd compose `R_z(α)` onto the extrinsic too. `prope.prepare_apply_fns` at `prope.py:114-117` reads only the axis-aligned `K` entries so it drops the resulting skew. Rare path (only fires on non-wrist views when `|angle| > 0.1` rad), noted but not fixed.

### 6. 180° image flip now paired with OpenCV-convention extrinsic

**Symptom.** `convert_libero_hdf5_to_lerobot.py:66` does `image[::-1, ::-1]` (= flipud ∘ fliplr, a 180° rotation). The stored extrinsic remained raw MuJoCo `cam_xpos` + `cam_xmat`, which is *y-up / z-back* — not the OpenCV *y-down / z-forward* convention that the pinhole `K` (with positive `fy` and `c_y = H/2`) assumes. So `K · \text{viewmat} · X_w` didn't land on the correct pixel in the stored (flipped) image.

**Ideology.** Separate the 180° flip into its two factors:
- **flipud** corresponds to the OpenGL-y-up → OpenCV-y-down *convention change*. This is absorbed by switching the camera frame from MuJoCo (y-up / z-back) to OpenCV (y-down / z-forward) — equivalent to right-multiplying the C2W matrix by $\mathrm{diag}(1, -1, -1, 1)$, which negates columns 1 and 2 of the rotation block while leaving the translation column and the homogeneous row untouched.
- **fliplr** is an additional horizontal mirror. With a centered principal point it absorbs cleanly into `K` as `f_x \to -f_x`.

With both factors applied consistently, $K_\text{adj} \cdot \text{viewmat}_\text{OpenCV} \cdot X_w$ lands on the stored-image pixel, and $K_\text{adj}^{-1} \cdot \text{pix}$ gives a ray in a standard OpenCV camera frame that matches the OpenCV extrinsic — so PRoPE and `ray_embed` share one pinhole convention.

**Code.** `src/openpi/policies/libero_policy.py`:
- Kept `_adjust_K_for_openpi_image_flip` (`f_x \to -f_x`) as-is.
- Added `_mujoco_to_opencv_extrinsic(T)` (lines 50-81): `T[..., :3, 1] *= -1; T[..., :3, 2] *= -1`, which is exactly $T \cdot \mathrm{diag}(1, -1, -1, 1)$ (bottom row zeros for cols 1-2 are preserved).
- `LiberoInputs.__call__` (lines 124, 127) applies `_mujoco_to_opencv_extrinsic` to both `agent_extrinsic` and `wrist_extrinsic`. Train and inference both flow through `LiberoInputs`, so the conversion is unified.

**Verification.** Numerical sanity check. Let camera be at world origin with identity C2W. Point $P_w = (1, 2, -2)$.
1. MuJoCo raw projection (y-up buffer, row 0 at bottom): $u_\text{raw} = c_x + f_x \cdot X / (-Z) = c_x + f_x/2$, $v_\text{raw} = c_y + f_y \cdot Y / (-Z) = c_y + f_y$.
2. After flipud+fliplr: pixel $(W-1-u_\text{raw},\, H-1-v_\text{raw}) \approx (c_x - f_x/2,\, c_y - f_y)$.
3. Model computes $K_\text{adj} \cdot \text{viewmat}_\text{OpenCV} \cdot P_w$. With identity C2W, $\text{viewmat}_\text{OpenCV} = \mathrm{diag}(1, -1, -1, 1)$. Applied to $(1, 2, -2, 1)$: $(1, -2, 2)$. Then $K_\text{adj} \cdot (1, -2, 2) = (-f_x + 2c_x,\, -2f_y + 2c_y,\, 2)$. Normalize by $z=2$: $(c_x - f_x/2,\, c_y - f_y)$. ✓ Matches step 2.
4. Ray check: $K_\text{adj}^{-1} \cdot (c_x - f_x/2,\, c_y - f_y,\, 1) = (1/2, -1, 1)$. The physical point in OpenCV cam frame (from $P_\text{cam,mj} = (1, 2, -2)$ mapped by $\mathrm{diag}(1,-1,-1)$) is $(1, -2, 2)$; at $z=2$, direction is $(0.5, -1, 1)$. ✓ Matches.

### 7. PRoPE attention now respects the view mask

**Symptom.** `CrossViewFusion.forward` used to call the pose-inject block without any `attn_mask`, so padded/invalid views (e.g. LIBERO's `right_wrist` with a zero image and a `stack_per_cam` identity-extrinsic fallback) were free to participate as attention keys and contaminate valid queries. Separately, `PRoPEAttention.forward` passed its `attn_mask` straight through to `F.scaled_dot_product_attention`, which uses the opposite boolean convention from `openpi.models_pytorch.layers.attn.Attention` — so any caller who did supply a mask would have had its semantics inverted.

**Ideology.** The codebase's native convention (established by `Attention.forward` at `attn.py:47-51`) is: `attn_mask: (B, S)` boolean where `True = blocked`. `scaled_dot_product_attention` wants the *opposite* — `True = allowed` — and expects a broadcastable shape like `(B, H, L, S)`. Every attention module therefore has to invert and broadcast internally. Making `PRoPEAttention` follow the same pattern means callers can use a single mask convention throughout.

The view mask that already flows into `CrossViewFusion.forward` is `(B, V, T)` with `True = valid`; flattened to `(B, V·T)` and negated, it becomes the required blocked mask for PRoPE's attention keys.

**Code.**
- `gemma_pytorch.py:127-133` (`PRoPEAttention.forward`): computes `allowed_mask = (~attn_mask)[:, None, None, :]` and hands *that* to `scaled_dot_product_attention`. Default `None` case preserved.
- `gemma_pytorch.py:306-315` (`CrossViewFusion.forward`): builds `prope_mask = ~mask.reshape(bsz, num_views * num_tokens)` and passes it as `attn_mask=prope_mask` into `pose_inject_blocks[...]`.
- `PoseInjectBlock.forward` already forwarded `attn_mask` down into `self.attn`, so no change there.

**Verification.** Convention consistency: `prope_mask[b, v·t] == True` iff `mask[b, v, t] == False` iff the token is invalid. Inside `PRoPEAttention`, `allowed_mask = ~prope_mask` recovers the sdpa-native "`True = attend`" convention. Shape `(B, 1, 1, V·T)` broadcasts over heads and queries, masking the same set of key positions for every query. Matches what `_process_global_attention` does for the regular `Block`s.

### 8. Identity fallback in `stack_per_cam` no longer leaks geometric signal

**Symptom.** `stack_per_cam` fills missing per-view entries with `torch.eye(...)`. Identity extrinsic means "camera at origin aligned with world"; identity `K` means "unit focal, principal point at `(0, 0)`". Neither is a meaningful null. For LIBERO the `right_wrist` slot is *always* `None`, so this leaks a deterministic, position-encoding-like signal every step through `ray_embed` and `pose_inject_blocks`.

With #7, those invalid-view tokens are blocked as *keys* in attention — so they can no longer contaminate *valid* views. But they're still *queries* (sdpa's `attn_mask` masks keys, not queries), which means they still accumulate residuals through `ray_embed(+identity·pix) + pose_inject_blocks(·identity)` and those residuals still drive gradient through the new modules.

**Ideology.** Keep the identity fallback (so downstream shape/dtype logic doesn't have to special-case missing views), but zero the contribution to `tokens` *after* the module runs. This preserves the "add-don't-subtract" invariant (the fallback is still there) while guaranteeing:
- `self.ray_embed.weight`'s gradient is the sum over valid views only.
- `self.pose_inject_blocks[*]`'s gradient is the sum over valid views only.

Equivalent to the standard "zero out padded positions before the residual add" trick used for variable-length sequences.

**Code.**
- `gemma_pytorch.py:598-606` (ray branch): after computing `ray_emb`, build `intr_valid = torch.tensor([cam_intr.get(k) is not None for k in cam_keys], dtype=bool)` and multiply the view axis: `ray_emb *= intr_valid[None, :, None, None]` before adding to `pending_ray_emb`.
- `gemma_pytorch.py:346-349` (PRoPE branch in `CrossViewFusion.forward`): after `pose_feat`, derive `view_valid = mask.any(dim=-1)` shape `(B, V)` and multiply `pose_feat *= view_valid[:, :, None, None]` before adding to `tokens`.

**Verification.** LIBERO trace: `right_wrist` is `None` → `intr_valid = [True, True, False]` → `ray_emb[:, right_wrist] = 0`. Since the image for that view is zeroed and its mask is `False`, `view_valid[:, right_wrist] = False` → `pose_feat[:, right_wrist] = 0`. Gradient of `ray_embed` and `pose_inject_blocks` only receives contributions from `base` and `left_wrist`, never from the fabricated identity entries.

### 9. `prope_layer_idx` validated at construction; CLI aligned with `aa_order="fg"`

**Symptom.** With `aa_order="fg"` there is exactly one global block, so `global_idx` only ever reaches 0 in `CrossViewFusion.forward`. `prope_layer_idx=(0, 1)` silently instantiated `len(pose_inject_blocks) = 2`, wasting parameters for `pose_inject_blocks[1]` which was never called, and forcing `find_unused_parameters=True` in DDP.

**Ideology.** The invariant is `max(prope_layer_idx) < num_global_blocks`. Encoding this as a startup precondition turns a silent waste into a loud construction error. The CLI then has to respect it.

**Code.**
- `gemma_pytorch.py:230-247` (`CrossViewFusion.__init__`): after `self.prope_after_global = set(prope_layer_idx)`, count `num_global_blocks = sum(1 for t in aa_order if t == "g")` and raise `ValueError` listing the offending indices and both remediation paths (reduce `prope_layer_idx` or extend `aa_order`).
- `scripts/run.sh:69-74`: `--model.cross_view.prope_layer_idx 0 1` → `--model.cross_view.prope_layer_idx 0` to match the stated `aa_order="fg"` intent, with a trailing comment documenting both valid configurations and pointing at the guard.

**Verification.** `num_global_blocks` derived from the same string the forward loop iterates over, so the guard's range `[0, num_global_blocks)` matches exactly the indices the forward pass can reach. Any future mismatch fails at model construction with a pointed error message.

---

## MEDIUM — design divergence / plumbing traps

### 10. Ray embedding relocated into SigLIP (pi3x pattern)

**Symptom.** The ray embedding was added to `vision_tower(images).last_hidden_state` — i.e. *after* the SigLIP transformer encoder. The encoder itself never saw intrinsic-aware patch tokens, which defeats the point of the ray signal in the pi3x design.

**Ideology.** Inject the per-view ray embedding into the patch-token stream *between* SigLIP's `patch_embedding` and the encoder layers, so every attention block gets intrinsic awareness from layer 1. The injection has to survive the `SiglipVisionEmbeddings` class boundary without forcing callers to thread a new kwarg through three HF classes (`SiglipVisionModel` → `SiglipVisionTransformer` → `SiglipVisionEmbeddings`). A transient attribute on the embeddings module, cleared in a `finally` block, does the job without touching the intermediate classes.

Zero-init is preserved: `ray_embed.proj` starts as all zeros, so `pending_ray_emb == 0` at step 0 and the embedding is a no-op; gradients still flow on the first backward pass.

**Code.**
- `transformers_replace/models/siglip/modeling_siglip.py:271-295` (`SiglipVisionEmbeddings.forward`): after `embeddings = patch_embeds.flatten(2).transpose(1, 2)`, read a transient `_pending_ray_emb` attribute with `getattr(self, "_pending_ray_emb", None)` and add it to `embeddings` before the position embedding. `getattr(..., None)` preserves default SigLIP behavior for any other caller. (Reminder: this file must be re-copied into the installed `transformers/models/siglip/` on any environment that runs the model — the repo README documents the `cp -r` step.)
- `gemma_pytorch.py:569-613` (`embed_image`): the ray-embedding computation (intrinsic K-stack with identity fallback, `K⁻¹ · pix` for `(H, W)=(224, 224)`, zero-init `PatchEmbed`, per-view `intr_valid` mask from #8) now runs *before* the `vision_tower` call. Its output `(B·V, N, D_vision)` is stored on `vision_tower.vision_model.embeddings._pending_ray_emb` inside a `try/finally` that clears the attribute even under exception or gradient-checkpoint forward-rerun.
- `gemma_pytorch.py:644-649`: the old post-SigLIP `tokens = tokens + ray_emb` block is replaced with a short pointer comment to the new location.

**Verification.** Shape: `ray_embed.proj` is `Conv2d(in=2, out=vision_hidden_dim, kernel=patch_size, stride=patch_size)`, so for `H=W=224` and `patch_size=14` its output is `(B·V, 256, D_vision)`, matching SigLIP's `patch_embeds.flatten(2).transpose(1, 2)`. Dtype: `pending_ray_emb.to(dtype=embeddings.dtype)` handles the bf16→fp32 upcast (SigLIP keeps `patch_embedding.weight` in fp32 per `params_to_keep_float32`). Gradient checkpointing: `_apply_checkpoint` reruns `image_embed_func` in backward, which re-enters `embed_image`, which re-sets the attribute before calling `vision_tower` and re-clears it in `finally` — consistent with the forward path.

### 11. `preprocess_observation_pytorch` carries camera fields through (already resolved)

**Symptom.** The original `SimpleProcessedObservation` omitted `agent_extrinsic / wrist_extrinsic / agent_intrinsic / wrist_intrinsic`. Not fatal in practice because `PI0Pytorch.forward` was passing the *original* observation into `embed_prefix`, but a latent footgun: any refactor that reads from the preprocessed object silently loses camera conditioning.

**Ideology.** Preprocessing is the one place in the pipeline that *must* update intrinsics (#5 applies image transforms that change `K`). So the preprocessed observation, not the original, should be the authoritative input to downstream layers.

**Code.** Already in place before this review pass:
- `preprocessing_pytorch.py:195-202, 322-344`: extrinsics/intrinsics are cloned upfront, updated per image transform, and carried through into `SimpleProcessedObservation`.
- JAX equivalent `src/openpi/models/model.py:214-221`: the returned `Observation` includes all four fields.
- `pi0_pytorch.py:201-211, 395, 479`: `_preprocess_observation` returns the processed observation and `forward` / `sample_actions` now pass it on.

**Verification.** `grep -n "agent_extrinsic\|agent_intrinsic" src/openpi/models_pytorch/preprocessing_pytorch.py` shows all four fields in both the input-clone block and the return object. The latent footgun is closed.

### 12. `poses` convention comment corrected

**Symptom.** `PoseInjectBlock.forward` had a one-line comment claiming "poses are world->camera; invert to get camera->world extrinsics." In reality, after #6, `poses` reaching this block are OpenCV-convention **camera-to-world** (C2W) matrices, and `se3_inverse(poses)` produces OpenCV **world-to-camera** (W2C) — which is what `prope.prepare_apply_fns` actually expects (`viewmats: camera<-world` per its docstring). The math was correct by coincidence; the comment was wrong.

**Ideology.** Comments should name (a) the upstream producer, (b) the downstream consumer, and (c) what the local code does between them. That way future readers can verify the convention chain without replaying the analysis.

**Code.** `gemma_pytorch.py:190-197` — rewrote the comment to spell out all three:
```python
# `poses` here are OpenCV-convention camera-to-world (C2W) matrices —
# `libero_policy._mujoco_to_opencv_extrinsic` upstream converts the raw
# MuJoCo C2W (y-up / z-back) into OpenCV (y-down / z-forward). PRoPE's
# `prope.prepare_apply_fns` expects `viewmats` in world-to-camera (W2C)
# form (see its docstring: "viewmats: camera<-world"), so we invert
# here. The local variable name `extrinsics` is legacy; it holds W2C.
extrinsics = se3_inverse(poses)
```

**Verification.** Matches the implemented #6 chain: LiberoInputs applies `_mujoco_to_opencv_extrinsic` → `obs.agent_extrinsic / wrist_extrinsic` are OpenCV C2W → `embed_image` stacks these into `prope_poses` (with identity fallback for missing views) → `PoseInjectBlock.forward` receives them as `poses` → inverts to OpenCV W2C → `prope.prepare_apply_fns` receives `viewmats` in the documented convention.

---

## LOW — not in scope for this pass

- `gemma_pytorch.py:443` hardcodes `self.num_views = 3`, coupling the stack to pi0's (base, left_wrist, right_wrist) layout.
- `find_unused_parameters=True` in DDP (`train_pytorch.py:612`) is still set. With #9's guard, the PRoPE "unused `pose_inject_blocks[1]`" case is fixed, but other conditionally-active modules (e.g. `ray_embed` when `cam_intr is None`) can still produce unused parameters, so the flag stays pragmatically.
- `OPENPI_TRAINABLE_PREFIXES` is still an env var read by the shell. Moving the allowlist into a versioned config would remove the "forgot to export" footgun.

---

## Pipeline trace (LIBERO, post-fix)

End-to-end, for a frame with `(agent, wrist, None)` cameras:

1. **Dataset**: `convert_libero_hdf5_to_lerobot.py` reads HDF5, computes per-camera `K` from `cam_fovy`, scales to LeRobot resolution via `_scale_intrinsic`, writes `{image, wrist_image, agent_intrinsic, wrist_intrinsic, agent_extrinsic, wrist_extrinsic}` per frame.
2. **Repack** (`training/config.py:LeRobotLiberoDataConfig`): maps `observation/agent_intrinsic` → `agent_intrinsic`, etc.
3. **LiberoInputs**: applies `_adjust_K_for_openpi_image_flip` (f_x → −f_x) and `_mujoco_to_opencv_extrinsic` (C2W_MuJoCo → C2W_OpenCV) so K, viewmat, and the stored flipped image all live in one OpenCV pinhole convention.
4. **Observation**: `Observation.from_dict` routes all four fields onto the dataclass.
5. **preprocess_observation_pytorch**: resizes 256→224 while composing the matching pixel-transform onto `K`; train-time crop/rotation compose further transforms. Carries camera fields into `SimpleProcessedObservation`.
6. **pi0_pytorch**: `_preprocess_observation` returns the processed obs; `forward` / `sample_actions` hand it to `embed_prefix`, which builds `cam_pos` / `cam_intr` dicts (base, left_wrist, right_wrist=None).
7. **embed_image (pre-SigLIP)**: computes rays from the scaled K, runs `ray_embed`, zeroes invalid views via `intr_valid`, queues the result onto `vision_tower.vision_model.embeddings._pending_ray_emb`.
8. **SigLIP (patched)**: `SiglipVisionEmbeddings.forward` adds `_pending_ray_emb` between `patch_embedding` and the position embedding; encoder layers see intrinsic-aware patch tokens from layer 1.
9. **embed_image (post-SigLIP)**: clears the ray attribute in `finally`; `view_embedding` (zero-init) adds a per-view bias; `cross_view_fusion` runs frame + global blocks with the view mask; after the configured global block, `pose_inject_blocks[...]` runs with `prope_mask`-blocked keys, and its output is zeroed on invalid views via `view_valid`.
10. **Projector → language model**: `masks.reshape(B, V·P)` propagates the per-token validity into the language-model attention mask so invalid views are excluded in the joint prefix too.

Every step is now consistent in its geometric conventions and in its masking semantics.
