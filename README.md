# Transformer Implementation in PyTorch

This repository contains a Transformer model implemented in PyTorch, leveraging the "English-Chinese Basic Sentences" dataset from the Kurohashi-Kawahara Laboratory at Kyoto University. 

## Dataset
The "English-Chinese Basic Sentences" dataset is used as the primary training and evaluation data. Each sentence pair contains a basic English sentence and its corresponding translation in Chinese, making it suitable for training language models for translation tasks.

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
