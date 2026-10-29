# Text Preprocessing
# Shared by training and inference so a transcript is always encoded exactly the same way.

import torch

from core.config import MAX_LENGTH, PAD_TOKEN_ID


def encode_transcript(tokenizer, text, max_length=MAX_LENGTH):
    """Return (padded token ids, number of real tokens)."""
    ids = tokenizer.encode(text)[:max_length]
    length = max(len(ids), 1)  # an empty transcript still reads position 0
    return ids + [PAD_TOKEN_ID] * (max_length - len(ids)), length


def select_last_token(logits, lengths):
    """Pick the logits at the last *real* token of each sequence.

    With causal attention this is the first position that has seen the whole
    transcript, and unlike the final padded position it is not diluted by padding.
    logits: [batch, seq_len, num_classes], lengths: [batch]
    """
    index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    return logits.gather(1, index).squeeze(1)


def to_tensors(tokenizer, texts, device):
    encoded = [encode_transcript(tokenizer, t) for t in texts]
    input_ids = torch.tensor([ids for ids, _ in encoded], device=device)
    lengths = torch.tensor([length for _, length in encoded], device=device)
    return input_ids, lengths
