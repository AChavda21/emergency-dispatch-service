import tiktoken
import torch

from core.config import MAX_LENGTH, PAD_TOKEN_ID
from core.preprocessing import encode_transcript, select_last_token, to_tensors

tokenizer = tiktoken.get_encoding("gpt2")


def test_short_transcript_is_padded_and_reports_real_length():
    ids, length = encode_transcript(tokenizer, "Send help now!")
    real = tokenizer.encode("Send help now!")
    assert len(ids) == MAX_LENGTH
    assert length == len(real)
    assert ids[:length] == real
    assert all(t == PAD_TOKEN_ID for t in ids[length:])


def test_long_transcript_is_truncated():
    ids, length = encode_transcript(tokenizer, "fire " * 200)
    assert len(ids) == MAX_LENGTH
    assert length == MAX_LENGTH


def test_empty_transcript_reads_first_position():
    ids, length = encode_transcript(tokenizer, "")
    assert ids == [PAD_TOKEN_ID] * MAX_LENGTH
    assert length == 1


def test_select_last_token_picks_last_real_position():
    logits = torch.arange(2 * 4 * 3, dtype=torch.float).view(2, 4, 3)
    lengths = torch.tensor([2, 4])
    picked = select_last_token(logits, lengths)
    assert torch.equal(picked[0], logits[0, 1])
    assert torch.equal(picked[1], logits[1, 3])


def test_to_tensors_shapes():
    input_ids, lengths = to_tensors(tokenizer, ["a", "a much longer transcript"], torch.device("cpu"))
    assert input_ids.shape == (2, MAX_LENGTH)
    assert lengths.tolist() == [1, len(tokenizer.encode("a much longer transcript"))]
