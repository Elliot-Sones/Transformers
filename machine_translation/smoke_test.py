import torch

from machine_translation.mini_transformer import (
    Seq2SeqConfig,
    Seq2Seq,
    prepare_decoder_inputs_and_labels,
)


def _quick_checks():
    V, B, S, T = 17, 2, 5, 6
    cfg = Seq2SeqConfig(src_vocab_size=V, tgt_vocab_size=V)
    m = Seq2Seq(cfg)

    # Random but valid inputs
    src = torch.randint(3, V, (B, S))
    src_mask = torch.ones(B, S, dtype=torch.long)

    tgt = torch.randint(3, V, (B, T))
    # Force BOS at t=0, EOS somewhere
    tgt[:, 0] = cfg.bos_token_id
    tgt[:, -2] = cfg.eos_token_id
    tgt_mask = (tgt != cfg.pad_token_id).long()

    dec_in, dec_mask, labels = prepare_decoder_inputs_and_labels(tgt, cfg.pad_token_id)
    loss, logits = m(src, src_mask, dec_in, dec_mask, labels)
    assert torch.isfinite(loss), "Loss should be finite"
    assert logits.shape == (B, T - 1, V)

    # All-keys-masked on source -> no NaNs
    src_mask_zero = torch.zeros_like(src_mask)
    _, logits2 = m(src, src_mask_zero, dec_in, dec_mask, labels)
    assert torch.isfinite(logits2).all(), "NaN in logits with all-masked src"

    # Generation (greedy + sampled)
    g = m.greedy_generate(src, src_mask, max_new_tokens=8)
    assert g.ndim == 2 and g.shape[0] == B
    gk = m.greedy_generate(src, src_mask, max_new_tokens=8, temperature=0.8, top_k=5)
    assert gk.shape[0] == B

    # Decode helper strips BOS and pads after EOS
    gd = m.decode_tokens(g.clone())
    assert gd.shape[0] == B


if __name__ == "__main__":
    _quick_checks()
    print("OK")

