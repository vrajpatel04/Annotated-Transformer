# AnnotatedTransformer.ipynb Explained

This guide follows `AnnotatedTransformer.ipynb` in notebook order. It focuses on the code cells, because those are the parts you usually need to explain aloud.

The notebook builds a Transformer for sequence-to-sequence work:

1. Convert source and target token IDs into embeddings.
2. Add positional information.
3. Encode the source sequence with stacked self-attention layers.
4. Decode the target sequence with masked self-attention plus source attention.
5. Project decoder vectors into vocabulary probabilities.
6. Train with masks, label smoothing, Adam, and a custom learning-rate schedule.

## Mental Model

For translation, imagine:

```text
German input:  <s> Ein Mann lacht </s>
English target: <s> A man laughs </s>
```

During training, the model sees the German sentence and the partial English prefix:

```text
decoder input:  <s> A man laughs
expected output: A man laughs </s>
```

The decoder must predict the next token at each position. The target mask prevents it from seeing future words.

Common tensor shapes:

```text
batch of token IDs:      [batch, seq_len]
embeddings:             [batch, seq_len, d_model]
attention scores:       [batch, heads, query_len, key_len]
model output:           [batch, seq_len, d_model]
vocab log-probabilities:[batch, seq_len, vocab_size]
```

## Cell 5: Imports And Global Flag

```python
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torch.utils.data import DataLoader, Dataset
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import Counter
from datasets import load_dataset
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
```

Line by line:

- `os`, `exists`: file paths, environment variables, and checking whether cached files exist.
- `torch`, `nn`: PyTorch tensors and neural-network modules.
- `log_softmax`: converts logits into log-probabilities for training.
- `pad`: pads variable-length token sequences to a fixed length.
- `math`: square roots, logs, sine/cosine calculations.
- `copy`: deep-copies modules so each Transformer layer has separate weights.
- `time`: measures training speed.
- `LambdaLR`: custom learning-rate schedule.
- `pandas`, `altair`: build visualization data and charts.
- `DataLoader`, `Dataset`: mini-batch loading.
- `spacy`: German and English tokenization.
- `GPUtil`, distributed imports: GPU monitoring and multi-GPU training.
- `Counter`: count tokens while building vocabularies.
- `load_dataset`: loads the Multi30k translation dataset.
- `warnings.filterwarnings("ignore")`: hides warning messages in the notebook.
- `RUN_EXAMPLES = True`: controls whether demo cells execute.

Example: if `RUN_EXAMPLES = False`, functions wrapped with `show_example(...)` or `execute_example(...)` will not run.

## Cell 6: Notebook Helpers

```python
def is_interactive_notebook():
    return __name__ == "__main__"
```

Returns `True` when the notebook is being run directly.

```python
def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)
```

Runs a function and returns its output only in interactive notebook mode.

Example:

```python
show_example(example_mask)
```

will display the mask chart when examples are enabled.

```python
def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)
```

Same idea, but used when the function prints or trains instead of returning a chart.

```python
class DummyOptimizer(torch.optim.Optimizer):
```

A fake optimizer for evaluation. During validation, the notebook wants to reuse `run_epoch`, but it does not want to update weights.

```python
self.param_groups = [{"lr": 0}]
```

Makes the dummy object look enough like a real PyTorch optimizer that logging code can access `optimizer.param_groups[0]["lr"]`.

```python
def step(self): None
def zero_grad(self, set_to_none=False): None
```

Do nothing. Evaluation needs no gradient update.

```python
class DummyScheduler:
    def step(self):
        None
```

Fake learning-rate scheduler for evaluation.

## Cell 13: EncoderDecoder

```python
class EncoderDecoder(nn.Module):
```

Defines the full sequence-to-sequence model wrapper.

```python
def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
```

The model is assembled from five pieces:

- `encoder`: processes the source sentence.
- `decoder`: generates target-side hidden states.
- `src_embed`: source token embedding plus positional encoding.
- `tgt_embed`: target token embedding plus positional encoding.
- `generator`: converts decoder output vectors to vocabulary probabilities.

```python
self.encoder = encoder
self.decoder = decoder
self.src_embed = src_embed
self.tgt_embed = tgt_embed
self.generator = generator
```

Stores each piece as part of the PyTorch module.

```python
def forward(self, src, tgt, src_mask, tgt_mask):
    return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
```

Full forward pass:

1. Encode `src`.
2. Feed encoder output, source mask, target tokens, and target mask to the decoder.

Example shape:

```text
src:      [32, 72]
tgt:      [32, 71]
encoded:  [32, 72, 512]
decoded:  [32, 71, 512]
```

```python
def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)
```

Embeds the source token IDs, then passes them through the encoder stack.

```python
def decode(self, memory, src_mask, tgt, tgt_mask):
    return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

Embeds target token IDs and decodes them using `memory`, the encoder output.

## Cell 14: Generator

```python
class Generator(nn.Module):
```

Final output layer.

```python
self.proj = nn.Linear(d_model, vocab)
```

Maps each decoder vector of size `d_model` to one score per vocabulary token.

Example:

```text
decoder vector: [512]
vocab size:     10000
output logits:  [10000]
```

```python
return log_softmax(self.proj(x), dim=-1)
```

Converts logits to log-probabilities along the vocabulary dimension.

## Cell 18: clones

```python
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

Creates `N` independent copies of a layer. This is important: every encoder layer has the same structure, but different learned weights.

Example:

```python
clones(EncoderLayer(...), 6)
```

returns six encoder layers.

## Cell 19: Encoder Stack

```python
class Encoder(nn.Module):
```

The full encoder is a stack of repeated `EncoderLayer`s.

```python
self.layers = clones(layer, N)
self.norm = LayerNorm(layer.size)
```

Creates `N` layers and a final layer normalization.

```python
for layer in self.layers:
    x = layer(x, mask)
return self.norm(x)
```

Passes the sequence through each encoder layer. Each layer updates every token's representation using self-attention and feed-forward processing.

## Cell 21: LayerNorm

```python
self.a_2 = nn.Parameter(torch.ones(features))
self.b_2 = nn.Parameter(torch.zeros(features))
```

Learnable scale and bias. These let the model adjust the normalized values.

```python
mean = x.mean(-1, keepdim=True)
std = x.std(-1, keepdim=True)
```

Computes mean and standard deviation over the last dimension, usually `d_model`.

```python
return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

Normalizes each token vector.

Tiny example:

```text
x token vector = [2, 4, 6]
mean = 4
std ~= 2
normalized ~= [-1, 0, 1]
```

## Cell 23: SublayerConnection

```python
class SublayerConnection(nn.Module):
```

Combines three operations:

1. Layer normalization.
2. A sublayer such as attention or feed-forward.
3. Dropout and residual addition.

```python
return x + self.dropout(sublayer(self.norm(x)))
```

This is a residual connection. The model keeps the original `x` and adds the transformed version.

Example:

```text
new representation = old representation + attention_result
```

Residual connections help gradients flow through deep networks.

## Cell 25: EncoderLayer

```python
self.self_attn = self_attn
self.feed_forward = feed_forward
self.sublayer = clones(SublayerConnection(size, dropout), 2)
```

Each encoder layer has:

- one self-attention sublayer.
- one feed-forward sublayer.
- two residual/norm wrappers.

```python
x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
```

Self-attention uses the same tensor as query, key, and value. In plain language: every source token looks at every other source token.

```python
return self.sublayer[1](x, self.feed_forward)
```

Applies a feed-forward network independently to each token position.

## Cell 27: Decoder Stack

```python
self.layers = clones(layer, N)
self.norm = LayerNorm(layer.size)
```

Creates `N` decoder layers and a final normalization.

```python
for layer in self.layers:
    x = layer(x, memory, src_mask, tgt_mask)
return self.norm(x)
```

Each decoder layer uses:

- target self-attention.
- source attention over encoder output.
- feed-forward processing.

## Cell 29: DecoderLayer

```python
self.sublayer = clones(SublayerConnection(size, dropout), 3)
```

The decoder has three sublayers, so it needs three residual/norm wrappers.

```python
m = memory
```

`memory` is the encoded source sentence.

```python
x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
```

Masked target self-attention. Each target position can only look left, not right.

Example: while predicting word 3, the decoder can see words 1 and 2, but not word 4.

```python
x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
```

Source attention. Target-side queries attend to source-side keys and values.

Example: while generating `"man"`, the decoder may attend strongly to German `"Mann"`.

```python
return self.sublayer[2](x, self.feed_forward)
```

Applies feed-forward transformation.

## Cell 31: subsequent_mask

```python
attn_shape = (1, size, size)
```

Builds a square attention mask for a sequence of length `size`.

```python
subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.bool)
```

Creates an upper-triangular matrix where future positions are `True`.

For `size = 4`, before inversion:

```text
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```

```python
return subsequent_mask == 0
```

Returns `True` where attention is allowed:

```text
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

## Cell 33: Mask Visualization

This cell turns the mask into a `pandas` table and plots it with Altair.

```python
subsequent_mask(20)[0][x, y]
```

Reads one allowed/disallowed attention value for row `x`, column `y`.

```python
alt.Chart(LS_data).mark_rect()
```

Draws the mask as a heatmap.

## Cell 36: Scaled Dot-Product Attention

```python
d_k = query.size(-1)
```

Gets the dimension of each query/key vector.

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
```

Computes similarity between each query and each key, then scales it.

Example:

```text
query: "it"
keys:  ["the", "animal", "sat"]
scores say which source words "it" should care about.
```

Scaling by `sqrt(d_k)` prevents dot products from becoming too large.

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

For masked positions, inserts a huge negative number. After softmax, those positions get probability almost zero.

```python
p_attn = scores.softmax(dim=-1)
```

Converts scores into attention weights that sum to 1.

```python
return torch.matmul(p_attn, value), p_attn
```

Weighted average of value vectors plus the attention weights themselves.

## Cell 40: MultiHeadedAttention

```python
assert d_model % h == 0
self.d_k = d_model // h
self.h = h
```

Splits the model dimension evenly across heads.

Example:

```text
d_model = 512, h = 8
d_k = 64 per head
```

```python
self.linears = clones(nn.Linear(d_model, d_model), 4)
```

Creates four linear layers:

1. query projection.
2. key projection.
3. value projection.
4. final output projection.

```python
if mask is not None:
    mask = mask.unsqueeze(1)
```

Adds a head dimension so the same mask can be broadcast across all heads.

```python
lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
```

Projects then reshapes from:

```text
[batch, seq_len, 512]
```

to:

```text
[batch, 8, seq_len, 64]
```

```python
x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
```

Runs scaled dot-product attention for every head.

```python
x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
```

Combines heads back into one `d_model` vector per token.

```python
return self.linears[-1](x)
```

Applies final projection.

## Cell 43: PositionwiseFeedForward

```python
self.w_1 = nn.Linear(d_model, d_ff)
self.w_2 = nn.Linear(d_ff, d_model)
```

Two-layer MLP applied to every position independently.

Example with defaults:

```text
512 -> 2048 -> 512
```

```python
return self.w_2(self.dropout(self.w_1(x).relu()))
```

Expands, applies ReLU, applies dropout, then compresses back.

## Cell 45: Embeddings

```python
self.lut = nn.Embedding(vocab, d_model)
```

Lookup table mapping token IDs to vectors.

Example:

```text
token ID 42 -> vector of length 512
```

```python
return self.lut(x) * math.sqrt(self.d_model)
```

Scales embeddings by `sqrt(d_model)`, as in the Transformer paper.

## Cell 47: PositionalEncoding

Transformers have no recurrence or convolution, so position must be added explicitly.

```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len).unsqueeze(1)
```

Creates a table with one row per possible position.

```python
div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
```

Creates frequencies for sine/cosine waves.

```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

Even dimensions get sine waves, odd dimensions get cosine waves.

```python
pe = pe.unsqueeze(0)
self.register_buffer("pe", pe)
```

Adds a batch dimension and stores it with the model without making it trainable.

```python
x = x + self.pe[:, : x.size(1)].requires_grad_(False)
return self.dropout(x)
```

Adds positional vectors to token embeddings and applies dropout.

Example:

```text
"bank" at position 2 gets a different final vector than "bank" at position 8.
```

## Cell 49: Positional Encoding Visualization

Creates a positional encoding with `d_model = 20`, applies it to zeros, and plots selected dimensions. Because the input is zero, the output is purely positional encoding.

## Cell 52: make_model

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
```

Factory function for the whole Transformer.

```python
c = copy.deepcopy
attn = MultiHeadedAttention(h, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
position = PositionalEncoding(d_model, dropout)
```

Creates prototype modules.

```python
Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
```

Builds the encoder stack from `N` cloned encoder layers.

```python
Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
```

Builds the decoder stack. It uses one attention module for decoder self-attention and another for source attention.

```python
nn.Sequential(Embeddings(d_model, src_vocab), c(position))
```

Source embedding pipeline: token embedding, then positional encoding.

```python
Generator(d_model, tgt_vocab)
```

Output projection to target vocabulary.

```python
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

Initializes matrix-shaped parameters with Xavier uniform initialization.

## Cell 54: Untrained Inference Test

```python
test_model = make_model(11, 11, 2)
test_model.eval()
```

Creates a small 2-layer Transformer with source and target vocab size 11.

```python
src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
src_mask = torch.ones(1, 1, 10)
```

Creates one source sequence and a mask that allows all source positions.

```python
memory = test_model.encode(src, src_mask)
ys = torch.zeros(1, 1).type_as(src)
```

Encodes the source and starts decoding with a single start token, `0`.

```python
for i in range(9):
```

Generates 9 more tokens.

```python
out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
prob = test_model.generator(out[:, -1])
_, next_word = torch.max(prob, dim=1)
```

Decodes the current prefix, looks only at the final position, and chooses the most likely next token.

Because the model is untrained, this prediction is random-looking.

## Cell 59: Batch And Masks

```python
self.src = src
self.src_mask = (src != pad).unsqueeze(-2)
```

Stores source tokens and creates a source padding mask.

Example:

```text
src = [5, 6, 2, 2], pad = 2
src_mask = [1, 1, 0, 0]
```

```python
self.tgt = tgt[:, :-1]
self.tgt_y = tgt[:, 1:]
```

Splits target into decoder input and expected output.

Example:

```text
tgt    = [<s>, A, man, </s>]
self.tgt   = [<s>, A, man]
self.tgt_y = [A, man, </s>]
```

```python
self.tgt_mask = self.make_std_mask(self.tgt, pad)
```

Combines padding mask and future-token mask.

```python
self.ntokens = (self.tgt_y != pad).data.sum()
```

Counts non-padding target tokens. Used to normalize the loss.

```python
tgt_mask = (tgt != pad).unsqueeze(-2)
tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
```

Allows attention only to non-pad positions and previous/current positions.

## Cell 62: TrainState

```python
class TrainState:
```

Tracks progress during training:

- `step`: batches processed in current epoch.
- `accum_step`: optimizer updates after gradient accumulation.
- `samples`: examples processed.
- `tokens`: non-padding tokens processed.

Note: this class uses class attributes rather than `@dataclass` instance fields. It works for the notebook, but a dataclass would be cleaner in production code.

## Cell 63: run_epoch

```python
start = time.time()
total_tokens = 0
total_loss = 0
tokens = 0
n_accum = 0
```

Initializes timers and counters.

```python
for i, batch in enumerate(data_iter):
```

Loops through batches.

```python
out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
```

Runs the Transformer.

```python
loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
```

Computes normalized loss. `loss_node` keeps the graph for backpropagation.

```python
if mode == "train" or mode == "train+log":
    loss_node.backward()
```

Backpropagates during training modes only.

```python
if i % accum_iter == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

Performs an optimizer update every `accum_iter` batches.

```python
scheduler.step()
```

Updates learning rate.

```python
total_loss += loss
total_tokens += batch.ntokens
tokens += batch.ntokens
```

Accumulates loss and token counts.

```python
if i % 40 == 1 and ...
```

Logs progress every 40 batches.

```python
return total_loss / total_tokens, train_state
```

Returns average loss per non-padding target token.

## Cell 69: Learning-Rate Schedule

```python
if step == 0:
    step = 1
```

Avoids dividing by zero.

```python
return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
```

This is the Transformer learning-rate schedule:

- during warmup, learning rate increases linearly.
- after warmup, learning rate decays proportional to `1 / sqrt(step)`.

Example intuition:

```text
early training: take gradually larger steps
later training: take gradually smaller steps
```

## Cell 70: Learning-Rate Visualization

Builds three dummy optimizer/scheduler pairs and records learning rates for 20,000 steps. Then it plots the schedules with Altair.

The model being optimized is irrelevant; this cell only visualizes the schedule.

## Cell 73: LabelSmoothing

```python
self.criterion = nn.KLDivLoss(reduction="sum")
```

Uses KL divergence between predicted log-probabilities and a smoothed target distribution.

```python
self.confidence = 1.0 - smoothing
```

If smoothing is `0.1`, the correct class receives confidence `0.9`.

```python
true_dist = x.data.clone()
true_dist.fill_(self.smoothing / (self.size - 2))
```

Starts by giving every non-special class a small probability.

Why `size - 2`? It excludes the padding class and the true class from the smoothing mass calculation.

```python
true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
```

Places the high probability on the correct class.

```python
true_dist[:, self.padding_idx] = 0
```

Padding should never be predicted as meaningful.

```python
mask = torch.nonzero(target.data == self.padding_idx)
if mask.dim() > 0:
    true_dist.index_fill_(0, mask.squeeze(), 0.0)
```

If the target itself is padding, set the whole target distribution to zero so it contributes no loss.

## Cell 75: Label-Smoothing Example

Creates a vocabulary of size 5 with padding index 0 and smoothing 0.4.

If the target is class `2`, the target distribution is roughly:

```text
class 0 pad: 0.0
class 1:     0.133
class 2:     0.6
class 3:     0.133
class 4:     0.133
```

The chart visualizes these smoothed distributions.

## Cell 77: Penalization Visualization

```python
def loss(x, crit):
    d = x + 3 * 1
    predict = torch.tensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
```

Builds fake prediction distributions where class 1 becomes more confident as `x` grows.

```python
return crit(torch.log(predict), torch.tensor([1])).data
```

Measures label-smoothing loss when the true class is 1.

Main lesson: with label smoothing, being infinitely confident is not always rewarded. The target distribution expects some probability mass on other classes.

## Cell 80: Synthetic Copy Data

```python
data = torch.randint(1, V, size=(batch_size, 10))
data[:, 0] = 1
```

Creates random sequences of length 10. First token is forced to 1.

```python
src = data.clone().detach()
tgt = data.clone().detach()
yield Batch(src, tgt, 0)
```

Source and target are identical. The task is copying.

Example:

```text
src = [1, 4, 9, 2]
tgt = [1, 4, 9, 2]
```

This checks whether the architecture can learn before trying translation.

## Cell 82: SimpleLossCompute

```python
x = self.generator(x)
```

Converts model hidden states to vocabulary log-probabilities.

```python
x.contiguous().view(-1, x.size(-1))
y.contiguous().view(-1)
```

Flattens batch and sequence dimensions.

Example:

```text
x: [batch=2, seq=3, vocab=11] -> [6, 11]
y: [batch=2, seq=3]           -> [6]
```

```python
sloss = self.criterion(...) / norm
```

Computes average loss per target token.

```python
return sloss.data * norm, sloss
```

Returns:

- detached total loss for logging.
- graph-connected normalized loss for backpropagation.

## Cell 85: greedy_decode

```python
memory = model.encode(src, src_mask)
```

Encodes the source once.

```python
ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
```

Starts target generation with a start symbol.

```python
for i in range(max_len - 1):
```

Generates one token at a time.

```python
out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
```

Decodes using the tokens generated so far.

```python
prob = model.generator(out[:, -1])
_, next_word = torch.max(prob, dim=1)
```

Uses only the last time step and picks the highest-probability token.

```python
ys = torch.cat([...], dim=1)
```

Appends the chosen token to the generated sequence.

Greedy decoding is simple but not always best. Beam search is usually better for translation.

## Cell 86: Toy Copy-Task Training

```python
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
```

Creates a tiny Transformer for an 11-token vocabulary.

```python
optimizer = torch.optim.Adam(...)
lr_scheduler = LambdaLR(...)
```

Uses the Transformer optimizer setup.

```python
for epoch in range(20):
```

Trains for 20 epochs.

```python
run_epoch(data_gen(...), model, SimpleLossCompute(...), optimizer, lr_scheduler, mode="train")
```

Runs training batches from synthetic copy data.

```python
run_epoch(..., DummyOptimizer(), DummyScheduler(), mode="eval")
```

Evaluates without updating weights.

```python
print(greedy_decode(...))
```

After training, the model should output something close to the source sequence.

## Cell 89: Loading spaCy Tokenizers

```python
spacy.load("de_core_news_sm")
spacy.load("en_core_web_sm")
```

Loads German and English tokenizer models.

```python
except IOError:
    os.system("python -m spacy download ...")
```

Downloads the model if it is missing.

The tokenizers split raw text into tokens:

```text
"Ein Mann lacht." -> ["Ein", "Mann", "lacht", "."]
```

## Cell 90: tokenize

```python
return [tok.text for tok in tokenizer.tokenizer(text)]
```

Runs spaCy tokenization and returns plain token strings.

## Cell 91: Vocabulary

```python
class Vocab:
```

Small replacement for `torchtext.vocab`.

```python
self.stoi = stoi
self.itos = itos
self.default_idx = default_idx
```

Stores:

- `stoi`: string to index.
- `itos`: index to string.
- `default_idx`: token ID for unknown words.

```python
def __getitem__(self, token):
    return self.stoi.get(token, self.default_idx)
```

Allows `vocab["dog"]`.

```python
def __call__(self, tokens):
    return [self.stoi.get(t, self.default_idx) for t in tokens]
```

Allows `vocab(["A", "man"])`.

```python
specials = ["<s>", "</s>", "<blank>", "<unk>"]
```

Special token IDs:

```text
0 = start of sentence
1 = end of sentence
2 = padding
3 = unknown
```

```python
counter_de.update(tokenize_de(ex["de"]))
counter_en.update(tokenize_en(ex["en"]))
```

Counts German and English tokens across train, validation, and test splits.

```python
for w, _ in counter_de.most_common():
    if w not in stoi_de:
        stoi_de[w] = len(stoi_de)
```

Adds words to vocabulary in frequency order.

```python
torch.save((vocab_src, vocab_tgt), "vocab.pt")
```

Caches vocabularies so they do not need to be rebuilt every run.

## Cell 94: collate_batch

`collate_batch` converts a list of raw string examples into padded tensors.

```python
bs_id = torch.tensor([0], device=device)
eos_id = torch.tensor([1], device=device)
```

Start and end token IDs.

```python
for (_src, _tgt) in batch:
```

Loops through raw sentence pairs.

```python
processed_src = torch.cat([bs_id, torch.tensor(src_vocab(src_pipeline(_src))), eos_id], 0)
```

Tokenizes source text, converts tokens to IDs, and wraps with `<s>` and `</s>`.

Example:

```text
"Ein Mann" -> [0, 812, 44, 1]
```

```python
pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id)
```

Pads the sequence to `max_padding`.

Example:

```text
[0, 812, 44, 1] -> [0, 812, 44, 1, 2, 2, 2, ...]
```

```python
src = torch.stack(src_list)
tgt = torch.stack(tgt_list)
return (src, tgt)
```

Stacks all examples into batch tensors.

## Cell 95: Multi30kDataset And Dataloaders

```python
class Multi30kDataset(Dataset):
```

Wraps the Hugging Face Multi30k split as a PyTorch dataset.

```python
self.data = list(ds)
```

Loads dataset rows into a Python list.

```python
return (self.data[idx]["de"], self.data[idx]["en"])
```

Each item is a German-English sentence pair.

```python
def create_dataloaders(...):
```

Builds train and validation dataloaders.

```python
train_sampler = DistributedSampler(train_dataset) if is_distributed else None
```

In distributed mode, each GPU receives a different slice of the data.

```python
DataLoader(..., batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn)
```

Creates loaders that produce already-tokenized, padded tensors.

## Cell 97: train_worker

This is one complete training process. In single-GPU or CPU mode, there is one worker. In distributed mode, there is one per GPU.

```python
device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")
```

Chooses GPU if available, otherwise CPU.

```python
pad_idx = vocab_tgt["<blank>"]
d_model = 512
model = make_model(len(vocab_src), len(vocab_tgt), N=6)
model = model.to(device)
```

Creates a full 6-layer Transformer and moves it to the device.

```python
if is_distributed:
    dist.init_process_group(...)
    model = DDP(model, device_ids=[gpu])
    module = model.module
```

Sets up distributed training if requested.

```python
criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
```

Uses label smoothing for translation.

```python
train_dataloader, valid_dataloader = create_dataloaders(...)
```

Builds data pipelines.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9)
```

Uses Adam with paper-style beta and epsilon values.

```python
lr_scheduler = LambdaLR(... rate(step, d_model, factor=1, warmup=config["warmup"]))
```

Uses the custom Transformer learning-rate schedule.

```python
for epoch in range(config["num_epochs"]):
```

Main training loop.

```python
model.train()
run_epoch(..., mode="train+log", accum_iter=config["accum_iter"])
```

Trains one epoch.

```python
torch.save(module.state_dict(), file_path)
```

Saves a checkpoint after each epoch on the main process.

```python
model.eval()
run_epoch(..., DummyOptimizer(), DummyScheduler(), mode="eval")
```

Runs validation.

```python
torch.save(module.state_dict(), "%sfinal.pt" % config["file_prefix"])
```

Saves the final model weights.

## Cell 98: Training Entry Points

```python
def train_distributed_model(...):
```

Detects number of GPUs and starts one training worker per GPU.

```python
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12356"
```

Distributed training processes coordinate through this address and port.

```python
mp.spawn(train_worker, nprocs=ngpus, args=(...))
```

Launches workers.

```python
def train_model(...):
    if config["distributed"]:
        train_distributed_model(...)
    else:
        train_worker(...)
```

Chooses distributed or single-process training.

```python
def load_trained_model():
```

Defines training config, trains if needed, then loads final weights.

```python
config = {
    "batch_size": 32,
    "distributed": False,
    "num_epochs": 8,
    ...
}
```

These are smaller notebook-friendly settings, not necessarily paper-scale settings.

## Cell 105: Shared Embeddings Sketch

```python
if False:
```

This code is intentionally disabled.

It shows the idea of tying source embeddings, target embeddings, and output projection weights. Weight tying can reduce parameters and sometimes improve performance.

Note: this snippet appears illustrative and would need correction before running because it references names like `tgt_embeddings` and `generator.lut` that do not match the implemented classes.

## Cell 108: Model Averaging Sketch

```python
def average(model, models):
```

Intended to average parameters from multiple checkpoints.

Conceptually:

```text
final_weight = average(weight_from_checkpoint_1, weight_from_checkpoint_2, ...)
```

Note: as written, `m.params()` should likely be `m.parameters()`, and the `torch.sum(*ps[1:])` expression is not correct for summing a tuple of tensors. Treat this as a sketch, not production-ready code.

## Cell 113: Checking Model Outputs

```python
results = [()] * n_examples
```

Preallocates output slots.

```python
b = next(iter(valid_dataloader))
rb = Batch(b[0], b[1], pad_idx)
```

Gets one validation batch and wraps it in the `Batch` helper to create masks.

```python
src_tokens = [vocab_src.get_itos()[int(x)] for x in rb.src[0] if x != pad_idx]
```

Converts source token IDs back to readable text.

```python
tgt_tokens = [vocab_tgt.get_itos()[int(x)] for x in rb.tgt[0] if x != pad_idx]
```

Converts target token IDs back to readable text.

```python
model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
```

Generates a translation.

```python
model_txt = " ".join([...]).split(eos_string, 1)[0] + eos_string
```

Converts predicted IDs to text and stops at the first end-of-sentence token.

```python
results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
```

Stores everything needed for later attention visualization.

## Cell 115: Attention Heatmap Helpers

```python
def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
```

Converts an attention matrix into a table.

```python
for r in range(m.shape[0])
for c in range(m.shape[1])
if r < max_row and c < max_col
```

Loops over visible rows and columns.

```python
float(m[r, c])
```

Attention value from row token to column token.

```python
def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
```

Builds one heatmap for one attention head.

```python
attn[0, head].data
```

Selects batch item 0 and one head.

```python
alt.Chart(data=df).mark_rect()
```

Draws the attention matrix as colored rectangles.

Interpretation:

```text
row token attends to column token
brighter/darker color means stronger/weaker attention depending on scale
```

## Cell 116: Accessing Attention From Layers

```python
def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn
```

Gets encoder self-attention weights for one layer.

```python
def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn
```

Gets decoder self-attention weights.

```python
def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn
```

Gets decoder-to-source attention weights.

```python
n_heads = attn.shape[1]
charts = [attn_map(... h ...) for h in range(n_heads)]
```

Creates one chart per attention head.

```python
assert n_heads == 8
```

Assumes the model uses 8 heads.

```python
return alt.vconcat(charts[0] | charts[2] | charts[4] | charts[6])
```

Shows a subset of heads in one combined chart.

## Cells 118, 120, 122: Attention Visualizations

### Encoder Self-Attention

```python
visualize_layer(model, layer, get_encoder, len(example[1]), example[1], example[1])
```

Rows and columns are both source tokens. This shows which source words attend to which other source words.

Example question it can answer:

```text
Does "Mann" attend to "Ein" or to the verb?
```

### Decoder Self-Attention

```python
visualize_layer(model, layer, get_decoder_self, len(example[1]), example[1], example[1])
```

Shows target-side masked self-attention. In principle, future target positions should be masked.

Note: the code passes `example[1]` source tokens as row and column labels. For a perfect label match, decoder self-attention should use target/output tokens.

### Decoder Source Attention

```python
visualize_layer(model, layer, get_decoder_src, max(len(example[1]), len(example[2])), example[1], example[2])
```

Shows how decoder tokens attend to source tokens. This is the closest visualization to word alignment in translation.

## One Complete Flow Example

Assume:

```text
source sentence IDs: [0, 10, 20, 1, 2, 2]
target sentence IDs: [0, 30, 40, 1, 2, 2]
```

`Batch` creates:

```text
src:      [0, 10, 20, 1, 2, 2]
src_mask: [1, 1, 1, 1, 0, 0]

tgt input:  [0, 30, 40, 1, 2]
tgt_y:      [30, 40, 1, 2, 2]
tgt_mask:   padding mask AND future mask
```

`EncoderDecoder.forward` does:

```text
src IDs
-> source embeddings
-> positional encoding
-> encoder self-attention stack
-> memory

tgt input IDs
-> target embeddings
-> positional encoding
-> decoder masked self-attention
-> decoder source attention over memory
-> feed-forward stack
-> decoder output vectors
```

`SimpleLossCompute` does:

```text
decoder vectors
-> Generator
-> log probabilities over vocabulary
-> KL divergence against smoothed target distribution
```

`greedy_decode` does:

```text
start with [<s>]
predict next token
append it
repeat until max length
```

## The Most Important Concepts To Explain

1. Attention is weighted lookup.
   The query asks a question, keys are compared with it, and values are averaged according to the attention weights.

2. Multi-head attention repeats this lookup several ways in parallel.
   Different heads can learn different relationships, such as local phrase structure, long-distance dependencies, or source-target alignment.

3. Masks protect the model from invalid information.
   Source masks hide padding. Target masks hide padding and future target tokens.

4. Residual connections preserve old information.
   Each sublayer learns an update, not a complete replacement.

5. Positional encoding gives order to a model that otherwise sees tokens as a set.

6. Training shifts the target.
   Decoder input is the target prefix. Decoder output is trained to predict the next token.

7. Greedy decoding is autoregressive.
   At inference time, the model feeds its own previous predictions back into the decoder.

