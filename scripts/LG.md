

  - Ray embed (intrinsics): ray_enc_type=True on Pi0Config. Zero-init PatchEmbed(in=2, D) in gemma_pytorch.py:528, applied post-SigLIP pre-projector at gemma_pytorch.py:590-598 (K⁻¹·pix → xy/z).                                                                              
  - PRoPE (extrinsics): pose_enc_type="prope" + cross_view.prope_layer_idx=(...). PoseInjectBlock uses prope.prepare_apply_fns, inserted after global blocks in CrossViewFusion (gemma_pytorch.py:305-355).                                                                     
  - Cross-view attention: cross_view.type="standard", aa_order="fg" — CrossViewFusion alternates frame and global blocks (gemma_pytorch.py:249-378).  
  - Cam_emb_type 
  
| Head        | Decoder → Head                                      | Output shape                                                  | What it predicts                          |
|-------------|-----------------------------------------------------|----------------------------------------------------------------|-------------------------------------------|
| Point       | point_decoder → point_head (ConvHead, 2 outputs)    | xy: (B,N,H,W,2), z: (B,N,H,W,1) → local_points = (xy*z, z)     | Per-pixel 3D point in that camera's frame |
| Camera      | camera_decoder → camera_head (CameraHead)           | (B,N,4,4)                                                      | Camera pose relative to cam_0             |
| Metric      | metric_decoder (ContextOnly) → metric_head (Linear→1)| (B,)                                                           | Global metric scale factor                |
| Confidence  | conf_decoder → conf_head (ConvHead, 1 output)       | (B,N,H,W,1)                                                    | Per-pixel confidence                      |


Size of data cache

| What                               | Shape per frame     | fp16 size | Notes                                                                                                                                                              |
|------------------------------------|---------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ray-emb teacher                    | (V, 256, 1024)      | ~1.5 MB   | Cache once per cam-config, not per frame — depends only on intrinsics. Libero intrinsics are static → one tensor for the whole dataset. Effectively free.         |
| Decoder features @ 2 layers        | (V, 256, 1024) × 2  | ~3 MB     | Biggest cost.                                                                                                                                                      |
| Point head outputs (patch-level)   | (V, 16, 16, 3)      | ~5 KB     | Store (xy, z) at patch resolution, not 224×224. Pi0's aux head predicts at the same resolution, then uses the same ConvHead upsampler at train time if needed.    |
| Conf (patch-level)                 | (V, 16, 16, 1)      | ~1.5 KB   | Cheap, can gate point loss.                                                                                                                                        |v                                                                                                    |               |




| Scope                                                | Per-frame fp16    | Total   | Dataset total     |
|------------------------------------------------------|-------------------|---------|-------------------|
| Ray-emb teacher                                      | — (static, 2 cams)| ~1 MB   | 36.00 GB          |
| + Phase-1: point (V,16,16,3) + conf (V,16,16,1)      | ~6 KB             | ~2.2 GB | ~38.2 GB (+6%)    |
| + Phase-2 (feat distill, PCA-64 × 2 layers)          | ~192 KB           | ~70 GB  | ~108 GB (3×)      |
| + Phase-2 (feat distill, full 1024 × 2 layers, no PCA)| ~3 MB             | ~1.1 TB | ~1.1 TB — avoid   |



| Aspect          | pi3x                                  | pi0 PointHead                                      | Reason                                                                                  |
|-----------------|---------------------------------------|----------------------------------------------------|-----------------------------------------------------------------------------------------|
| Decoder depth   | 5 BlockRope                           | 2 _PointBlock                                      | "Light" — enough capacity to refine patch features without bloating params              |
| Attention class | pi3's FlashAttentionRope              | layers.attn.Attention + RotaryPositionEmbedding2D  | Reuse existing pi0 primitives, same 2D RoPE math                                        |
| Hidden dim      | 1024                                  | 512 (configurable)                                 | ~¼ the params; still fits 8 heads × 64 head_dim                                         |
| Output          | ConvHead upsamples patches→224² (xy, z)| Linear patch→3 at 16×16                            | Teacher targets are cached at patch resolution; no upsampling needed                    |
| Output init     | default                               | zero-initialized                                   | Head contributes zero until distillation loss drives it — same trick as the ray_embed   |

