"""V-JEPA 2.1 teacher encoder.

Loads Meta's V-JEPA 2.1 weights (HF-converted safetensors, weights only) into the
native `transformers` VJEPA2 encoder and re-implements the two 2.1-specific
additions on top of the native modules:
  * a learned modality embedding added to video tokens after patch embedding
  * `norms_block` (deep-supervision layer norms); the last one is the output norm
No remote code is executed.
"""
from __future__ import annotations
import json, torch, torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import VJEPA2Config
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Encoder

CKPTS = {"vitb": "apiantonio/vjepa2.1-vit-base-384", "vitl": "apiantonio/vjepa2.1-vit-large-384"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class VJEPA21Encoder(nn.Module):
    def __init__(self, size="vitb", crop=256, frames=16):
        super().__init__()
        repo = CKPTS[size]
        c = json.load(open(hf_hub_download(repo, "config.json")))
        cfg = VJEPA2Config(hidden_size=c["hidden_size"], num_hidden_layers=c["num_hidden_layers"],
                           num_attention_heads=c["num_attention_heads"], crop_size=crop, frames_per_clip=frames,
                           tubelet_size=c["tubelet_size"], patch_size=c["patch_size"], mlp_ratio=c["mlp_ratio"],
                           layer_norm_eps=c["layer_norm_eps"])
        self.enc = VJEPA2Encoder(cfg)
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        self.video_mod_embed = nn.Parameter(enc.pop("embeddings.video_mod_embed"))
        n_norm = max(int(k.split(".")[1]) for k in enc if k.startswith("norms_block")) + 1
        self.out_norm = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])
        self.out_norm.weight.data = enc.pop(f"norms_block.{n_norm-1}.weight")
        self.out_norm.bias.data = enc.pop(f"norms_block.{n_norm-1}.bias")
        drop = [k for k in enc if k.startswith("norms_block") or "patch_embeddings_img" in k or "img_mod_embed" in k]
        for k in drop: enc.pop(k)
        res = self.enc.load_state_dict(enc, strict=False)
        assert set(res.missing_keys) == {"layernorm.weight", "layernorm.bias"}, res
        assert not res.unexpected_keys, res
        self.hidden = c["hidden_size"]; self.tubelet = c["tubelet_size"]; self.patch = c["patch_size"]
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, video_uint8: torch.Tensor):
        """video_uint8: [B,T,H,W,3] uint8 -> tokens [B, T/2 * (H/16)*(W/16), D]"""
        x = video_uint8.permute(0, 1, 4, 2, 3).float() / 255.0
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        h = self.enc.embeddings(x) + self.video_mod_embed.to(x.device)
        for layer in self.enc.layer:
            h = layer(h, None)[0]
        return self.out_norm(h)

    @torch.no_grad()
    def per_step_latents(self, video_uint8: torch.Tensor, grid=2):
        """Returns [B, T/2, D*(1+grid*grid)]: per temporal slot, mean token + grid-pooled spatial tokens."""
        B, T, H, W, _ = video_uint8.shape
        tok = self.forward(video_uint8)
        th, tw = H // self.patch, W // self.patch
        tok = tok.view(B, T // self.tubelet, th, tw, self.hidden)
        mean = tok.mean(dim=(2, 3))
        pooled = torch.nn.functional.adaptive_avg_pool2d(tok.permute(0, 1, 4, 2, 3).flatten(0, 1), grid)
        pooled = pooled.view(B, T // self.tubelet, self.hidden, grid * grid).permute(0, 1, 3, 2).flatten(2)
        return torch.cat([mean, pooled], dim=-1)
