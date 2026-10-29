# Pretrained Weight Loader
# Downloads OpenAI's GPT-2 (124M) checkpoint from Hugging Face and maps it onto our from-scratch GPTModel.

import logging

import requests
import torch

from core.config import PRETRAINED_GPT2_PATH, PRETRAINED_GPT2_URL


def download_gpt2_checkpoint(path=PRETRAINED_GPT2_PATH, url=PRETRAINED_GPT2_URL):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading pretrained GPT-2 weights from {url} ...")
    tmp_path = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp_path.rename(path)
    logging.info(f"Saved pretrained weights to {path}")
    return path


def _assign(param, value):
    if param.shape != value.shape:
        raise ValueError(f"Shape mismatch: expected {tuple(param.shape)}, got {tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value)


def load_gpt2_weights(model, hf_state_dict):
    """Copy a Hugging Face GPT-2 state dict into GPTModel.

    Hugging Face stores GPT-2 linear layers as Conv1D ([in, out]), so every
    weight matrix is transposed to match nn.Linear ([out, in]). The fused
    c_attn projection is split into separate query/key/value matrices.
    """
    sd = {k.replace("transformer.", "", 1): v for k, v in hf_state_dict.items()}

    _assign(model.tok_emb.weight, sd["wte.weight"])
    _assign(model.pos_emb.weight, sd["wpe.weight"])

    for i, block in enumerate(model.trf_blocks):
        p = f"h.{i}."
        q_w, k_w, v_w = sd[p + "attn.c_attn.weight"].chunk(3, dim=-1)
        q_b, k_b, v_b = sd[p + "attn.c_attn.bias"].chunk(3, dim=-1)
        _assign(block.att.W_query.weight, q_w.T)
        _assign(block.att.W_key.weight, k_w.T)
        _assign(block.att.W_value.weight, v_w.T)
        _assign(block.att.W_query.bias, q_b)
        _assign(block.att.W_key.bias, k_b)
        _assign(block.att.W_value.bias, v_b)
        _assign(block.att.out_proj.weight, sd[p + "attn.c_proj.weight"].T)
        _assign(block.att.out_proj.bias, sd[p + "attn.c_proj.bias"])

        _assign(block.ff.layers[0].weight, sd[p + "mlp.c_fc.weight"].T)
        _assign(block.ff.layers[0].bias, sd[p + "mlp.c_fc.bias"])
        _assign(block.ff.layers[2].weight, sd[p + "mlp.c_proj.weight"].T)
        _assign(block.ff.layers[2].bias, sd[p + "mlp.c_proj.bias"])

        _assign(block.norm1.weight, sd[p + "ln_1.weight"])
        _assign(block.norm1.bias, sd[p + "ln_1.bias"])
        _assign(block.norm2.weight, sd[p + "ln_2.weight"])
        _assign(block.norm2.bias, sd[p + "ln_2.bias"])

    _assign(model.final_norm.weight, sd["ln_f.weight"])
    _assign(model.final_norm.bias, sd["ln_f.bias"])
    # GPT-2 ties the output head to the token embedding
    _assign(model.out_head.weight, sd["wte.weight"])
    return model


def load_pretrained_gpt2(model):
    path = download_gpt2_checkpoint()
    hf_state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return load_gpt2_weights(model, hf_state_dict)
