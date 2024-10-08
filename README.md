# Transformer Implementation in PyTorch

This repository contains a Transformer model implemented in PyTorch, leveraging the "English-Chinese Basic Sentences" dataset from the Kurohashi-Kawahara Laboratory at Kyoto University. The model is designed to perform machine translation between English and Chinese using an encoder-decoder architecture. Below, we provide an overview of the model structure and key components, along with code snippets for better understanding.

## Dataset
The "English-Chinese Basic Sentences" dataset is used as the primary training and evaluation data. Each sentence pair contains a basic English sentence and its corresponding translation in Chinese, making it suitable for training language models for translation tasks.

## Model Architecture
The implemented Transformer follows the traditional encoder-decoder architecture with separate multi-head attention layers for encoding and decoding. The encoder and decoder structures are built to capture complex word relationships and contextual information from input sequences. A high-level summary of the architecture is given below:

### Encoder
The encoder is composed of multiple stacked layers, each designed to extract sequential and contextual information from the input sentences. Each layer consists of the following sub-components:

1. **Token Embedding Layer**  
   Converts each token into a fixed-size embedding vector, capturing the semantic representation of words.
   
   ```python
   self.embedding = nn.Embedding(vocab_size, embed_dim)
   ```

2. **Positional Encoding**  
   Since Transformers lack inherent sequential order awareness, a positional encoding is added to the embedding vector to capture token positions within the sentence.

   ```python
   self.positional_encoding = PositionalEncoding(embed_dim)
   ```

3. **Multi-Head Self-Attention**  
   This layer computes relationships between tokens, allowing the model to focus on relevant parts of the input sentence.

   ```python
   attn_output, _ = self.self_attention(query, key, value)
   ```

4. **Residual Connection and Layer Normalization**  
   Applies skip connections to prevent vanishing gradients and layer normalization for stabilizing training.

   ```python
   self.norm1 = nn.LayerNorm(embed_dim)
   ```

5. **Feed-Forward Network**  
   A simple fully connected layer with a ReLU activation to introduce non-linearity.

   ```python
   self.feed_forward = nn.Sequential(
       nn.Linear(embed_dim, ff_dim),
       nn.ReLU(),
       nn.Linear(ff_dim, embed_dim)
   )
   ```

6. **Repeat Above Layers**  
   The above structure is repeated multiple times (6 layers) to refine the embeddings.

### Decoder
The decoder is structured similarly to the encoder but includes an additional attention layer to incorporate information from the encoder outputs. The components include:

1. **Token Embedding Layer**  
   Similar to the encoder, the decoder starts by converting input tokens to embeddings.

2. **Positional Encoding**  
   Positional encodings are added to maintain the order of tokens in the target sequence.

3. **Masked Multi-Head Self-Attention**  
   Prevents the decoder from attending to future positions by masking future tokens.

   ```python
   attn_output, _ = self.self_attention(query, key, value, mask=mask)
   ```

4. **Residual Connection and Layer Normalization**  

5. **Source-Target Multi-Head Attention**  
   Attends to the encoder outputs to gather context for each word in the target sequence.

   ```python
   attn_output, _ = self.cross_attention(query, encoder_outputs, encoder_outputs)
   ```

6. **Residual Connection and Layer Normalization**  

7. **Feed-Forward Network**  

8. **Repeat Above Layers**  
   The decoder also stacks multiple such layers (6 layers in total).

9. **Output Linear Layer**  
   The final linear layer maps the transformed embeddings back to vocabulary size.

   ```python
   self.output_layer = nn.Linear(embed_dim, vocab_size)
   ```

10. **Softmax**  
    Applies softmax to convert logits to probability distributions over the vocabulary.

## How to Run
To train and evaluate the model, follow these steps:

1. **Install Dependencies**
   Ensure you have the required libraries installed:

   ```bash
   pip install torch numpy
   ```

2. **Prepare the Dataset**
   Download the "English-Chinese Basic Sentences" dataset from the official source. Preprocess the dataset to create input-output pairs for training.

3. **Train the Model**
   Run the training script with the appropriate hyperparameters:

   ```bash
   python train.py --epochs 20 --batch_size 64 --learning_rate 1e-4
   ```

4. **Evaluate the Model**
   Use the evaluation script to measure the translation performance on the test set:

   ```bash
   python evaluate.py --model_path saved_model.pth --test_data test_data.txt
   ```

## Results
The model achieves promising results on the given dataset, showing that the encoder-decoder structure successfully captures the linguistic nuances of English and Chinese translations.

## References
1. [Vaswani et al., 2017. "Attention is All You Need"](https://arxiv.org/abs/1706.03762)
2. Kurohashi-Kawahara Laboratory: ["English-Chinese Basic Sentences"](https://nlp.ist.i.kyoto-u.ac.jp/EN/)
