import pytest
import torch

from core.architecture import GPTModel
from core.pretrained import load_gpt2_weights

TINY_CONFIG = {"vocab_size": 100, "context_length": 8, "emb_dim": 16, "n_heads": 2,
               "n_layers": 2, "drop_rate": 0.0, "qkv_bias": True}


def fake_hf_state_dict(cfg, prefix="transformer."):
    d, v, n = cfg["emb_dim"], cfg["vocab_size"], cfg["context_length"]
    sd = {"wte.weight": torch.randn(v, d), "wpe.weight": torch.randn(n, d),
          "ln_f.weight": torch.randn(d), "ln_f.bias": torch.randn(d)}
    for i in range(cfg["n_layers"]):
        p = f"h.{i}."
        sd.update({
            p + "attn.c_attn.weight": torch.randn(d, 3 * d), p + "attn.c_attn.bias": torch.randn(3 * d),
            p + "attn.c_proj.weight": torch.randn(d, d), p + "attn.c_proj.bias": torch.randn(d),
            p + "mlp.c_fc.weight": torch.randn(d, 4 * d), p + "mlp.c_fc.bias": torch.randn(4 * d),
            p + "mlp.c_proj.weight": torch.randn(4 * d, d), p + "mlp.c_proj.bias": torch.randn(d),
            p + "ln_1.weight": torch.randn(d), p + "ln_1.bias": torch.randn(d),
            p + "ln_2.weight": torch.randn(d), p + "ln_2.bias": torch.randn(d),
        })
    return {prefix + k: t for k, t in sd.items()}


def test_weights_are_transposed_and_split():
    hf = fake_hf_state_dict(TINY_CONFIG)
    model = load_gpt2_weights(GPTModel(TINY_CONFIG), hf)
    d = TINY_CONFIG["emb_dim"]

    c_attn = hf["transformer.h.0.attn.c_attn.weight"]
    assert torch.equal(model.trf_blocks[0].att.W_query.weight, c_attn[:, :d].T)
    assert torch.equal(model.trf_blocks[0].att.W_key.weight, c_attn[:, d:2 * d].T)
    assert torch.equal(model.trf_blocks[0].att.W_value.weight, c_attn[:, 2 * d:].T)
    assert torch.equal(model.trf_blocks[1].ff.layers[0].weight, hf["transformer.h.1.mlp.c_fc.weight"].T)
    assert torch.equal(model.tok_emb.weight, hf["transformer.wte.weight"])
    assert torch.equal(model.final_norm.bias, hf["transformer.ln_f.bias"])


def test_keys_without_prefix_also_load():
    model = load_gpt2_weights(GPTModel(TINY_CONFIG), fake_hf_state_dict(TINY_CONFIG, prefix=""))
    assert model.pos_emb.weight.shape == (TINY_CONFIG["context_length"], TINY_CONFIG["emb_dim"])


def test_shape_mismatch_raises():
    wrong = dict(TINY_CONFIG, emb_dim=32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_gpt2_weights(GPTModel(TINY_CONFIG), fake_hf_state_dict(wrong))
