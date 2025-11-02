# Decoder 

In this section we will be implementing the decoder. 

<img src="assets/definitions/decoder.png" width=60% >

The **decoder** section also has their own multipleencoder layers to build context of the language (french languge in my example).

Althouhg the encoder also adds in another layer after, that performs multi-head attention ([visit encoder read me for def of attention](/encoder_transformer/README.md)) over the output of the encoder stack.



Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.









# Decoder: 
[Simple video](https://www.youtube.com/watch?v=wjZofJX0v4M)

 **Transformers** are a type of deep neural network architecture that takes a sequence of data (such as text), understands how each element in the sequence relates to the others and predicts what comes next.








### How it works

In large language models (LLMs), a transformer understands how words in a sentence relate to each other so it can capture meaning and generate the next word.

> **"The cat sat ..."**

The model does this by separating the sentence into small sections called **tokens** (sections that can be a full word but not always can be section of a word). Then we embed each input token into a vector (a list of numbers) that captures its meaning.


$$
E =
\begin{bmatrix}
0.2 & 0.4 & 0.1 \\\\
0.6 & 0.1 & 0.8 \\\\
0.9 & 0.7 & 0.3
\end{bmatrix}

The = 
\begin{bmatrix}
0.2 & 0.4 & 0.1
\end{bmatrix}

cat= 
\begin{bmatrix}
0.6 & 0.1 & 0.8 
\end{bmatrix}

sat= 
\begin{bmatrix}
0.9 & 0.7 & 0.3
\end{bmatrix}
$$


#### Transformer Blocks



The transformer blocks, decides which token are most relevant to each other (Self-Attention layer) and then refines and transform the information (Feedfoward Neural Network).

<img src="assets/definitions/transformer.png" width="50%" alt="Transformers definition">

##### Self Attention


Each token creates three versions of itself:  
- **Query (Q):** What it’s looking for  
- **Key (K):** What it offers  
- **Value (V):** Its meaning  

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Then, every word compares its **Query** with every other **Key**:

$$
\text{Scores} = QK^T
$$

Higher scores mean the words are more related. Finally, each word mixes the information from all others using these weights:

$$
\text{Output} = \text{Attention} \times V
$$


#### Feedforward Neural Network + Normalize

After **Self-Attention**, each word’s vector now contains context.  

The **Feedforward Layer** helps the model process that information more deeply using a small two-layer network:

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

Then, the model adds the original input back to the output (a **residual connection**) and **normalizes** it for stability:

$$
\text{Output} = \text{LayerNorm}(x + \text{FFN}(x))
$$



### Language Modeling Head
After the transformer finishes processing,  
each word has a final **vector** that captures its full meaning and context.  
Now the model needs to turn those vectors into actual **predicted words**.

Each word is now just a list of numbers (a vector) — for example:

| Word | Vector (example) |
|:-----|:----------------:|
| The  | [0.23, -0.11, 0.77, 0.52, ...] |
| king | [0.45, 0.84, -0.31, 0.09, ...] |

These vectors are the **input** to the language modeling head.

---

#### ⚙️ What happens next

The model multiplies these vectors by a large **vocabulary matrix**  
(one row for every possible word it can predict).  
This gives a **score** for each possible next word.

Then it uses **softmax** to turn those scores into **probabilities**  
that add up to 1.

---

#### 🗣️ Output Example

If the model just saw the phrase “The”, it might predict:

| Next Word | Probability |
|:-----------|:-------------:|
| king | 0.65 |
| cat  | 0.18 |
| apple | 0.06 |
| sat | 0.02 |

The word with the **highest probability** ("king") is chosen as the next word.





<img src="assets/definitions/transformers.jpeg" width="50%" alt="Transformers definition">




# Applying:

**Input** : Tiny ShakespeaRe

**Train**
- Run: `python3 training.py`
- Device auto-detects `mps` on Apple Silicon; falls back to `cpu`.
- The script now saves a checkpoint at `assets/checkpoints/gpt-YYYYmmdd-HHMMSS.pt` and a convenient copy at `assets/checkpoints/latest.pt`.

**Generate**
- After training, sample text with:
  - `python3 sample.py --prompt "ROMEO:" --max_new_tokens 300`
- If needed, specify device: `--device cpu` or `--device mps`.
- To use a specific checkpoint: `--ckpt assets/checkpoints/gpt-20241017-153800.pt`.

Notes
- Prompts should use characters seen during training (Tiny Shakespeare) for best results.
- Checkpoint includes model hyperparameters and the character vocabulary, so generation does not require the training data.

**Resume + Checkpoints**
- Training now saves periodic checkpoints every `--save_interval` steps (default = `eval_interval`).
- Latest checkpoint path: `assets/checkpoints/latest.pt`.
- Resume training:
  - `python3 training.py --resume` (auto picks `latest.pt` if present)
  - Or specify: `python3 training.py --resume --ckpt assets/checkpoints/gpt-YYYYmmdd-HHMMSS-step3000.pt`
- Safe interrupt: Press Ctrl+C; the script saves a checkpoint at the next safe point and exits.


## Steps and connections

- Implement `decoder.py` with embeddings, masked self-attn, cross-attn, FFN blocks.
- Inputs: `tgt_input_ids`, `encoder_hidden_states`, masks → logits.
- Tie decoder token embedding with LM head for efficiency.
- Use causal mask internally; mask PAD tokens in self-attention too.
- Connect in `machine_translation/model.py` as the generation component.






Decoder (French:
Stack of N decoder blocks, each block has:
Masked self-attention on target tokens (causal mask).
Cross-attention over encoder_hidden_states (K,V from encoder; Q from decoder).

**Goal:** Build a decoder (GPT) from scratch

Extra resources to help out: