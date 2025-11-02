# Machine Translation 


Seq2Seq wrapper:
Owns Encoder, Decoder, and a tied output head.
Forward computes cross-entropy over tgt_input_ids[1:] given teacher forcing on tgt_input_ids[:-1].
Ignores PAD in loss; supports label smoothing.


Combining both the encoder and decoder 

+ the extra things to make it work


## Steps and connections

- Run `python3 setup_data.py` to produce `archive/train.csv`, `archive/test.csv`.
- Implement `data.py` to load CSVs, tokenize, pad, build masks.
- Implement `model.py` `Seq2Seq` wiring Encoder+Decoder with loss.
- Implement `train.py` loop, scheduler, checkpointing, BLEU eval.
- Implement `inference.py` greedy/beam search using saved checkpoints.