import os
import torch
import numpy as np
import json
from transformers import GPT2LMHeadModel

# 1) choose your model
model_name = "gpt2"  # the 117M-param checkpoint
out_dir    = "weights"
os.makedirs(out_dir, exist_ok=True)

# 2) load the HF model
model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float32)
sd    = model.state_dict()

# manifest dict to collect mappings
manifest = {}

# helper to write a Float32Array to .bin and record in manifest
def save(name: str, arr: np.ndarray):
    path = os.path.join(out_dir, f"{name}.bin")
    arr.astype(np.float32).tofile(path)
    manifest[name] = f"/{out_dir[7:]}/{name}.bin" # subscript gets rid of warning later on
    print(f"→ {path}  shape={arr.shape}")

# 3) embeddings
save("wte", sd["transformer.wte.weight"].cpu().numpy())   # [vocab, emb]
save("wpe", sd["transformer.wpe.weight"].cpu().numpy())   # [pos,   emb]

# 4) for each layer, split out Q/K/V from the combined c_attn
for i in range(model.config.n_layer):
    prefix = f"transformer.h.{i}"

    # c_attn: shape [3*emb, emb]
    c_attn_w = sd[f"{prefix}.attn.c_attn.weight"].cpu()
    c_attn_b = sd[f"{prefix}.attn.c_attn.bias"].cpu()
    qw, kw, vw = c_attn_w.chunk(3, dim=-1)
    qb, kb, vb = c_attn_b.chunk(3, dim=-1)
    save(f"c_attn_q_w_{i}", qw.numpy())
    save(f"c_attn_k_w_{i}", kw.numpy())
    save(f"c_attn_v_w_{i}", vw.numpy())
    save(f"c_attn_q_b_{i}", qb.numpy())
    save(f"c_attn_k_b_{i}", kb.numpy())
    save(f"c_attn_v_b_{i}", vb.numpy())

    # output projection
    save(f"c_proj_w_{i}", sd[f"{prefix}.attn.c_proj.weight"].cpu().numpy())
    save(f"c_proj_b_{i}", sd[f"{prefix}.attn.c_proj.bias"].cpu().numpy())

    # MLP
    save(f"mlp_fc_w_{i}",   sd[f"{prefix}.mlp.c_fc.weight"].cpu().numpy())
    save(f"mlp_fc_b_{i}",   sd[f"{prefix}.mlp.c_fc.bias"].cpu().numpy())
    save(f"mlp_proj_w_{i}", sd[f"{prefix}.mlp.c_proj.weight"].cpu().numpy())
    save(f"mlp_proj_b_{i}", sd[f"{prefix}.mlp.c_proj.bias"].cpu().numpy())

    # layer norms
    save(f"ln_1_g_{i}", sd[f"{prefix}.ln_1.weight"].cpu().numpy())
    save(f"ln_1_b_{i}", sd[f"{prefix}.ln_1.bias"].cpu().numpy())
    save(f"ln_2_g_{i}", sd[f"{prefix}.ln_2.weight"].cpu().numpy())
    save(f"ln_2_b_{i}", sd[f"{prefix}.ln_2.bias"].cpu().numpy())

# 5) final layer norm + lm head
save("ln_f_g",   sd["transformer.ln_f.weight"].cpu().numpy())
save("ln_f_b",   sd["transformer.ln_f.bias"].cpu().numpy())
save("lm_head_w", sd["lm_head.weight"].cpu().numpy())
# if there’s a separate bias
if "lm_head.bias" in sd:
    save("lm_head_b", sd["lm_head.bias"].cpu().numpy())

# 6) write out manifest.json
manifest_path = os.path.join(out_dir, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"→ {manifest_path}  ({len(manifest)} entries)")
