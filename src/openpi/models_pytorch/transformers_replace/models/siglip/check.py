import inspect

import transformers


def check_whether_transformers_replace_is_installed_correctly():
    if transformers.__version__ != "4.53.2":
        return False

    try:
        from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings
    except Exception:
        return False

    try:
        source = inspect.getsource(SiglipVisionEmbeddings.forward)
    except OSError:
        return False

    return "_pending_ray_emb" in source and "pending_ray_emb" in source
