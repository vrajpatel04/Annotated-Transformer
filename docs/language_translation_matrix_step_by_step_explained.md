# Language Translation With Matrices: Step By Step

This guide explains Transformer translation using small matrices.

It follows the same idea as `AnnotatedTransformer.ipynb`, but uses tiny dimensions so every matrix is visible.

Real notebook size:

```text
d_model = 512
heads = 8
d_k = 64
```

Tiny example size:

```text
batch_size = 1
d_model = 4
heads = 2
d_k = 2
```

We will translate:

```text
German source:
Ein Mann spielt Gitarre

English target:
A man plays guitar
```

With start and end tokens:

```text
source:
<s> Ein Mann spielt Gitarre </s>

target:
<s> A man plays guitar </s>
```

## 1. Token IDs Are The First Matrix-Like Object

The model does not receive strings. It receives integer IDs.

Toy source vocabulary:

```text
<s>      -> 0
</s>     -> 1
<blank>  -> 2
<unk>    -> 3
Ein      -> 4
Mann     -> 5
spielt   -> 6
Gitarre  -> 7
```

Toy target vocabulary:

```text
<s>      -> 0
</s>     -> 1
<blank>  -> 2
<unk>    -> 3
A        -> 4
man      -> 5
plays    -> 6
guitar   -> 7
```

So:

```text
src_ids =
[
  [0, 4, 5, 6, 7, 1]
]

shape = [1, 6]
```

```text
tgt_full_ids =
[
  [0, 4, 5, 6, 7, 1]
]

shape = [1, 6]
```

The batch dimension is the first dimension:

```text
1 sentence
6 token positions
```

## 2. Training Splits Target Into Input And Output

The notebook does this:

```python
self.tgt = tgt[:, :-1]
self.tgt_y = tgt[:, 1:]
```

So the full target:

```text
[<s>, A, man, plays, guitar, </s>]
```

becomes:

```text
tgt input to decoder =
[
  [<s>, A, man, plays, guitar]
]

IDs =
[
  [0, 4, 5, 6, 7]
]

shape = [1, 5]
```

```text
tgt_y expected output =
[
  [A, man, plays, guitar, </s>]
]

IDs =
[
  [4, 5, 6, 7, 1]
]

shape = [1, 5]
```

Meaning:

```text
decoder input position 0: <s>       -> predict A
decoder input position 1: A         -> predict man
decoder input position 2: man       -> predict plays
decoder input position 3: plays     -> predict guitar
decoder input position 4: guitar    -> predict </s>
```

## 3. Embedding Matrix

An embedding layer is a lookup table.

If source vocabulary size is 8 and `d_model = 4`:

```text
source embedding table E_src shape = [8, 4]
```

Example:

```text
E_src =

token      id   vector
<s>        0    [0.10, 0.00, 0.20, 0.10]
</s>       1    [0.00, 0.10, 0.10, 0.40]
<blank>    2    [0.00, 0.00, 0.00, 0.00]
<unk>      3    [0.05, 0.05, 0.05, 0.05]
Ein        4    [0.40, 0.30, 0.10, 0.00]
Mann       5    [0.20, 0.70, 0.30, 0.10]
spielt     6    [0.80, 0.10, 0.60, 0.20]
Gitarre    7    [0.30, 0.20, 0.90, 0.70]
```

Lookup:

```text
src_ids = [0, 4, 5, 6, 7, 1]
```

becomes:

```text
X_src =
[
  [0.10, 0.00, 0.20, 0.10],   # <s>
  [0.40, 0.30, 0.10, 0.00],   # Ein
  [0.20, 0.70, 0.30, 0.10],   # Mann
  [0.80, 0.10, 0.60, 0.20],   # spielt
  [0.30, 0.20, 0.90, 0.70],   # Gitarre
  [0.00, 0.10, 0.10, 0.40]    # </s>
]

shape = [6, 4]
```

With batch dimension:

```text
X_src shape = [1, 6, 4]
```

In the real notebook:

```text
[1, 6, 4] becomes [batch, src_len, 512]
```

## 4. Positional Encoding Matrix

The model adds position information.

Use this tiny positional matrix:

```text
P_src =
[
  [0.00, 0.00, 0.00, 0.00],   # position 0
  [0.01, 0.02, 0.03, 0.04],   # position 1
  [0.02, 0.04, 0.06, 0.08],   # position 2
  [0.03, 0.06, 0.09, 0.12],   # position 3
  [0.04, 0.08, 0.12, 0.16],   # position 4
  [0.05, 0.10, 0.15, 0.20]    # position 5
]

shape = [6, 4]
```

Add embedding plus position:

```text
X_src_pos = X_src + P_src
```

Result:

```text
X_src_pos =
[
  [0.10, 0.00, 0.20, 0.10],   # <s>
  [0.41, 0.32, 0.13, 0.04],   # Ein
  [0.22, 0.74, 0.36, 0.18],   # Mann
  [0.83, 0.16, 0.69, 0.32],   # spielt
  [0.34, 0.28, 1.02, 0.86],   # Gitarre
  [0.05, 0.20, 0.25, 0.60]    # </s>
]

shape = [6, 4]
```

This is what enters the encoder.

## 5. Encoder Self-Attention As Matrix Multiplication

In one attention head:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Where:

```text
X shape   = [src_len, d_model] = [6, 4]
W_Q shape = [d_model, d_k]     = [4, 2]
W_K shape = [d_model, d_k]     = [4, 2]
W_V shape = [d_model, d_k]     = [4, 2]
```

So:

```text
Q shape = [6, 2]
K shape = [6, 2]
V shape = [6, 2]
```

For a simple example, suppose after multiplying by learned projection matrices we get:

```text
Q_src =
[
  [0.10, 0.10],   # <s>
  [0.40, 0.20],   # Ein
  [0.30, 0.80],   # Mann
  [0.90, 0.10],   # spielt
  [0.20, 0.50],   # Gitarre
  [0.00, 0.20]    # </s>
]
```

```text
K_src =
[
  [0.10, 0.10],   # <s>
  [0.40, 0.20],   # Ein
  [0.30, 0.80],   # Mann
  [0.90, 0.10],   # spielt
  [0.20, 0.50],   # Gitarre
  [0.00, 0.20]    # </s>
]
```

```text
V_src =
[
  [0.20, 0.10],   # <s>
  [0.10, 0.00],   # Ein
  [0.30, 0.10],   # Mann
  [0.60, 0.20],   # spielt
  [0.90, 0.70],   # Gitarre
  [0.10, 0.40]    # </s>
]
```

Attention scores:

```text
Scores = Q_src K_src^T / sqrt(d_k)
```

Shape:

```text
Q_src       [6, 2]
K_src^T     [2, 6]
Scores      [6, 6]
```

Every row is a token asking a question.
Every column is a token being looked at.

```text
Scores =

             <s>   Ein  Mann spielt Gitarre </s>
<s>         0.01  0.04  0.08   0.07   0.05   0.02
Ein         0.04  0.20  0.28   0.38   0.18   0.04
Mann        0.08  0.28  0.73   0.35   0.46   0.16
spielt      0.07  0.38  0.35   0.82   0.23   0.02
Gitarre     0.05  0.18  0.46   0.23   0.29   0.10
</s>        0.02  0.04  0.16   0.02   0.10   0.04
```

The real code divides by `sqrt(d_k)`:

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
```

Then softmax converts each row into attention probabilities:

```text
A_src = softmax(Scores)

shape = [6, 6]
```

Example attention probability matrix:

```text
A_src =

             <s>   Ein  Mann spielt Gitarre </s>
<s>         0.16  0.16  0.17   0.17   0.17   0.16
Ein         0.15  0.17  0.18   0.19   0.17   0.15
Mann        0.13  0.16  0.25   0.17   0.19   0.14
spielt      0.13  0.18  0.18   0.29   0.16   0.12
Gitarre     0.14  0.16  0.21   0.16   0.18   0.15
</s>        0.16  0.16  0.18   0.16   0.17   0.16
```

Now multiply probabilities by values:

```text
Z_src = A_src V_src
```

Shape:

```text
A_src  [6, 6]
V_src  [6, 2]
Z_src  [6, 2]
```

For the `spielt` row:

```text
Z_spielt =
  0.13 * V_<s>
+ 0.18 * V_Ein
+ 0.18 * V_Mann
+ 0.29 * V_spielt
+ 0.16 * V_Gitarre
+ 0.12 * V_</s>
```

This is a weighted mixture of source token vectors.

Meaning:

```text
The new "spielt" representation contains information from the full source sentence,
with more weight on tokens the model found useful.
```

## 6. Multi-Head Matrix Shape

One head gave:

```text
Z_head_1 shape = [6, 2]
```

With 2 heads:

```text
Z_head_1 shape = [6, 2]
Z_head_2 shape = [6, 2]
```

Concatenate:

```text
Z_concat shape = [6, 4]
```

Then apply output projection:

```text
Encoder attention output = Z_concat W_O
```

Shape:

```text
Z_concat [6, 4]
W_O      [4, 4]
output   [6, 4]
```

This matches the original `d_model`, so the layer can continue.

In the real notebook:

```text
before split: [batch, src_len, 512]
after split:  [batch, heads, src_len, 64]
after concat: [batch, src_len, 512]
```

## 7. Encoder Memory Matrix

After self-attention, residual connections, layer norm, and feed-forward layers, the encoder returns:

```text
memory
```

For our tiny example:

```text
memory =
[
  [0.18, 0.08, 0.22, 0.16],   # contextual <s>
  [0.31, 0.28, 0.18, 0.11],   # contextual Ein
  [0.29, 0.62, 0.34, 0.19],   # contextual Mann
  [0.76, 0.20, 0.58, 0.27],   # contextual spielt
  [0.38, 0.30, 0.88, 0.69],   # contextual Gitarre
  [0.09, 0.18, 0.20, 0.48]    # contextual </s>
]

shape = [6, 4]
```

With batch dimension:

```text
memory shape = [1, 6, 4]
```

Important:

```text
memory is not the English translation.
memory is a matrix of contextual German source vectors.
```

## 8. Decoder Input Matrix

During training, decoder input is:

```text
<s> A man plays guitar
```

IDs:

```text
tgt = [0, 4, 5, 6, 7]
```

Target embedding lookup plus positional encoding gives:

```text
X_tgt_pos =
[
  [0.12, 0.00, 0.18, 0.05],   # <s>
  [0.52, 0.22, 0.11, 0.08],   # A
  [0.24, 0.66, 0.31, 0.17],   # man
  [0.71, 0.19, 0.55, 0.29],   # plays
  [0.33, 0.24, 0.84, 0.73]    # guitar
]

shape = [5, 4]
```

With batch dimension:

```text
X_tgt_pos shape = [1, 5, 4]
```

## 9. Target Future Mask Matrix

The decoder must not see future target words.

For target input:

```text
<s> A man plays guitar
```

the future mask is:

```text
tgt_mask =

          <s>  A  man plays guitar
<s>        1   0   0    0      0
A          1   1   0    0      0
man        1   1   1    0      0
plays      1   1   1    1      0
guitar     1   1   1    1      1

shape = [5, 5]
```

Meaning:

```text
row = current target position
column = target position it may attend to
```

For the `man` row:

```text
can see:    <s>, A, man
cannot see: plays, guitar
```

In code:

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

The masked positions become huge negative numbers before softmax.

After softmax, their probability becomes almost zero.

## 10. Decoder Masked Self-Attention Matrix

The decoder first attends over the target prefix.

Formula:

```text
Q_tgt = X_tgt W_Q
K_tgt = X_tgt W_K
V_tgt = X_tgt W_V
```

Shapes:

```text
X_tgt [5, 4]
W_Q   [4, 2]
W_K   [4, 2]
W_V   [4, 2]

Q_tgt [5, 2]
K_tgt [5, 2]
V_tgt [5, 2]
```

Scores:

```text
Scores_tgt = Q_tgt K_tgt^T / sqrt(d_k)

shape = [5, 5]
```

Before mask:

```text
Scores_tgt =

          <s>   A   man plays guitar
<s>      0.10 0.20 0.30  0.40   0.50
A        0.20 0.40 0.50  0.70   0.80
man      0.30 0.50 0.90  0.60   0.70
plays    0.40 0.70 0.60  1.00   0.90
guitar   0.50 0.80 0.70  0.90   1.10
```

After applying future mask:

```text
Masked Scores_tgt =

          <s>   A   man plays guitar
<s>      0.10 -inf -inf -inf   -inf
A        0.20 0.40 -inf -inf   -inf
man      0.30 0.50 0.90 -inf   -inf
plays    0.40 0.70 0.60 1.00   -inf
guitar   0.50 0.80 0.70 0.90   1.10
```

After softmax:

```text
A_tgt =

          <s>    A    man  plays guitar
<s>      1.00  0.00  0.00  0.00   0.00
A        0.45  0.55  0.00  0.00   0.00
man      0.25  0.31  0.44  0.00   0.00
plays    0.20  0.27  0.24  0.29   0.00
guitar   0.15  0.20  0.18  0.22   0.25
```

Then:

```text
Z_tgt = A_tgt V_tgt

shape = [5, 2]
```

Meaning:

```text
Each English position now contains information from only allowed previous English positions.
```

## 11. Cross-Attention Is The Translation Matrix

Now comes the most important part.

In decoder source attention:

```python
self.src_attn(x, m, m, src_mask)
```

That means:

```text
Q comes from decoder target-side vectors
K comes from encoder memory
V comes from encoder memory
```

Matrix formulas:

```text
Q_cross = decoder_state W_Q_cross
K_cross = memory W_K_cross
V_cross = memory W_V_cross
```

Shapes:

```text
decoder_state [tgt_len, d_model] = [5, 4]
memory        [src_len, d_model] = [6, 4]

Q_cross       [5, 2]
K_cross       [6, 2]
V_cross       [6, 2]
```

Cross-attention scores:

```text
Scores_cross = Q_cross K_cross^T / sqrt(d_k)
```

Shape:

```text
Q_cross        [5, 2]
K_cross^T      [2, 6]
Scores_cross   [5, 6]
```

This shape is important:

```text
[target positions, source positions]
```

Rows are English-side positions.
Columns are German-side positions.

```text
Scores_cross =

              <s>   Ein  Mann spielt Gitarre </s>
<s>          0.10  0.45  0.20   0.15   0.10   0.05
A            0.05  1.20  0.30   0.10   0.05   0.02
man          0.03  0.25  1.30   0.20   0.10   0.02
plays        0.00  0.10  0.20   1.40   0.15   0.00
guitar       0.00  0.05  0.10   0.20   1.50   0.05
```

After softmax row by row:

```text
A_cross =

              <s>   Ein  Mann spielt Gitarre </s>
<s>          0.15  0.22  0.17   0.16   0.15   0.14
A            0.12  0.36  0.15   0.12   0.12   0.11
man          0.11  0.14  0.42   0.13   0.12   0.11
plays        0.11  0.12  0.13   0.43   0.12   0.11
guitar       0.10  0.11  0.11   0.12   0.45   0.11
```

This is the easiest matrix to connect to translation.

It says:

```text
English A      attends most to German Ein
English man    attends most to German Mann
English plays  attends most to German spielt
English guitar attends most to German Gitarre
```

Cross-attention output:

```text
Z_cross = A_cross V_cross
```

Shape:

```text
A_cross  [5, 6]
V_cross  [6, 2]
Z_cross  [5, 2]
```

Each English position receives a weighted mixture of German source information.

## 12. One Cross-Attention Row In Detail

Focus on the target position that should predict:

```text
plays
```

The cross-attention row is:

```text
plays row =

<s>      0.11
Ein      0.12
Mann     0.13
spielt   0.43
Gitarre  0.12
</s>     0.11
```

Suppose:

```text
V_cross =

<s>      [0.18, 0.08]
Ein      [0.31, 0.28]
Mann     [0.29, 0.62]
spielt   [0.76, 0.20]
Gitarre  [0.38, 0.88]
</s>     [0.09, 0.18]
```

Then:

```text
Z_plays =
  0.11 * [0.18, 0.08]
+ 0.12 * [0.31, 0.28]
+ 0.13 * [0.29, 0.62]
+ 0.43 * [0.76, 0.20]
+ 0.12 * [0.38, 0.88]
+ 0.11 * [0.09, 0.18]
```

First dimension:

```text
0.11*0.18 + 0.12*0.31 + 0.13*0.29 + 0.43*0.76 + 0.12*0.38 + 0.11*0.09
= 0.477
```

Second dimension:

```text
0.11*0.08 + 0.12*0.28 + 0.13*0.62 + 0.43*0.20 + 0.12*0.88 + 0.11*0.18
= 0.334
```

So:

```text
Z_plays = [0.477, 0.334]
```

This vector now carries a lot of information from `spielt`, because the weight for `spielt` was highest.

## 13. Feed-Forward Matrix

After attention, the decoder applies a feed-forward network at every target position.

Conceptually:

```text
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

Shape in tiny example:

```text
x      [5, 4]
W_1    [4, 8]
hidden [5, 8]
W_2    [8, 4]
output [5, 4]
```

Shape in the real notebook:

```text
x      [batch, tgt_len, 512]
W_1    [512, 2048]
hidden [batch, tgt_len, 2048]
W_2    [2048, 512]
output [batch, tgt_len, 512]
```

The feed-forward network does not mix token positions.

Attention mixes positions.
Feed-forward transforms each position's vector.

## 14. Decoder Output Matrix

After all decoder layers:

```text
decoder_output =
[
  [0.21, 0.10, 0.30, 0.12],   # position <s>, predicts A
  [0.40, 0.30, 0.22, 0.14],   # position A, predicts man
  [0.55, 0.18, 0.61, 0.25],   # position man, predicts plays
  [0.35, 0.22, 0.82, 0.70],   # position plays, predicts guitar
  [0.10, 0.08, 0.15, 0.90]    # position guitar, predicts </s>
]

shape = [5, 4]
```

With batch:

```text
decoder_output shape = [1, 5, 4]
```

Each row will be used to predict the next target token.

## 15. Generator Matrix: Decoder Vector To Vocabulary Scores

The generator is:

```python
self.proj = nn.Linear(d_model, vocab)
return log_softmax(self.proj(x), dim=-1)
```

Matrix form:

```text
logits = decoder_output W_vocab + b_vocab
```

Tiny target vocabulary size is 8:

```text
W_vocab shape = [4, 8]
b_vocab shape = [8]
```

So:

```text
decoder_output [5, 4]
W_vocab        [4, 8]
logits         [5, 8]
```

One row per target position.
One column per possible English token.

Columns:

```text
0:<s>  1:</s>  2:<blank>  3:<unk>  4:A  5:man  6:plays  7:guitar
```

Example logits:

```text
logits =

              <s>  </s> blank unk    A   man plays guitar
pos 0 <s>    -2.0  -1.8  -9.0 -3.0  3.2  1.0  0.2   0.1
pos 1 A      -2.0  -1.5  -9.0 -3.0  0.4  3.6  0.7   0.2
pos 2 man    -2.2  -1.4  -9.0 -3.0  0.2  0.8  4.1   0.5
pos 3 plays  -2.1  -1.2  -9.0 -3.0  0.1  0.3  0.8   3.9
pos 4 guitar -2.0   4.2  -9.0 -3.0  0.1  0.2  0.2   0.4
```

After softmax:

```text
probabilities =

position 0 predicts A
position 1 predicts man
position 2 predicts plays
position 3 predicts guitar
position 4 predicts </s>
```

This matches:

```text
tgt_y = [A, man, plays, guitar, </s>]
```

## 16. Loss Matrix View

The target answers are:

```text
tgt_y =
[4, 5, 6, 7, 1]
```

The generator output has shape:

```text
[5, 8]
```

The training loss looks at the correct column for each row:

```text
row 0 correct column = 4, token A
row 1 correct column = 5, token man
row 2 correct column = 6, token plays
row 3 correct column = 7, token guitar
row 4 correct column = 1, token </s>
```

If the probabilities are:

```text
P(A at row 0)        = 0.80
P(man at row 1)      = 0.84
P(plays at row 2)    = 0.88
P(guitar at row 3)   = 0.82
P(</s> at row 4)     = 0.90
```

the loss is low.

If the probabilities are:

```text
P(A at row 0)        = 0.05
P(man at row 1)      = 0.02
P(plays at row 2)    = 0.01
P(guitar at row 3)   = 0.03
P(</s> at row 4)     = 0.04
```

the loss is high.

Backpropagation updates the matrices:

```text
embedding matrices
W_Q, W_K, W_V, W_O matrices
feed-forward matrices
generator W_vocab matrix
```

## 17. Inference Matrix Flow

During inference, target is unknown.

Source:

```text
src = [<s>, Ein, Mann, spielt, Gitarre, </s>]
```

Step 1:

```text
encode source once:
memory shape = [6, 4]
```

Start:

```text
ys = [<s>]
```

Decoder matrix:

```text
X_ys shape = [1, 4]
```

Cross-attention:

```text
Q_cross shape      = [1, 2]
K_memory shape     = [6, 2]
Scores_cross shape = [1, 6]
A_cross shape      = [1, 6]
```

Generator:

```text
decoder_output shape = [1, 4]
logits shape         = [1, 8]
```

Choose highest probability:

```text
A
```

Now:

```text
ys = [<s>, A]
```

Step 2:

```text
X_ys shape = [2, 4]
Scores_self shape = [2, 2]
Scores_cross shape = [2, 6]
logits shape = [2, 8]
```

The notebook uses only the last output row:

```python
prob = model.generator(out[:, -1])
```

That means:

```text
Use the decoder vector for the newest position only.
```

Choose:

```text
man
```

Now:

```text
ys = [<s>, A, man]
```

Step 3:

```text
X_ys shape = [3, 4]
Scores_self shape = [3, 3]
Scores_cross shape = [3, 6]
logits shape from last row = [1, 8]
```

Choose:

```text
plays
```

The same pattern continues until:

```text
ys = [<s>, A, man, plays, guitar, </s>]
```

## 18. The Most Important Matrix Shapes

Training:

```text
src IDs              [batch, src_len]
tgt input IDs        [batch, tgt_len]
tgt_y IDs            [batch, tgt_len]

src embeddings       [batch, src_len, d_model]
tgt embeddings       [batch, tgt_len, d_model]

encoder memory       [batch, src_len, d_model]
decoder output       [batch, tgt_len, d_model]
generator output     [batch, tgt_len, tgt_vocab_size]
```

Encoder self-attention:

```text
Q_src                [batch, heads, src_len, d_k]
K_src                [batch, heads, src_len, d_k]
V_src                [batch, heads, src_len, d_k]
scores              [batch, heads, src_len, src_len]
attention           [batch, heads, src_len, src_len]
output              [batch, heads, src_len, d_k]
```

Decoder masked self-attention:

```text
Q_tgt                [batch, heads, tgt_len, d_k]
K_tgt                [batch, heads, tgt_len, d_k]
V_tgt                [batch, heads, tgt_len, d_k]
scores              [batch, heads, tgt_len, tgt_len]
attention           [batch, heads, tgt_len, tgt_len]
output              [batch, heads, tgt_len, d_k]
```

Decoder cross-attention:

```text
Q_cross              [batch, heads, tgt_len, d_k]
K_cross              [batch, heads, src_len, d_k]
V_cross              [batch, heads, src_len, d_k]
scores              [batch, heads, tgt_len, src_len]
attention           [batch, heads, tgt_len, src_len]
output              [batch, heads, tgt_len, d_k]
```

The most translation-looking matrix is:

```text
decoder cross-attention scores/attention:
[target length, source length]
```

Because it connects:

```text
target words -> source words
```

## 19. One-Screen Summary

For translation:

```text
source IDs
  -> source embedding matrix lookup
  -> add positional matrix
  -> encoder self-attention:
       Q_src K_src^T gives [src_len, src_len]
  -> memory matrix

target prefix IDs
  -> target embedding matrix lookup
  -> add positional matrix
  -> decoder masked self-attention:
       Q_tgt K_tgt^T gives [tgt_len, tgt_len]
  -> decoder cross-attention:
       Q_decoder K_memory^T gives [tgt_len, src_len]
  -> feed-forward
  -> generator:
       decoder_output W_vocab gives [tgt_len, vocab_size]
  -> choose next token
```

The key matrix for translation is:

```text
A_cross = softmax(Q_decoder K_memory^T / sqrt(d_k))
```

Then:

```text
Z_cross = A_cross V_memory
```

That is how the decoder pulls the right source-language information into each target-language position.

