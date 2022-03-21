# kbart

K-BART is a encoder-decoder model. Encoder is injected with a Knowledge graph. Encoder is using DistilRoBERTa, decoder is using GPT2
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install transformers==4.11.2
pip install spacy
python -m spacy download en
pip install rouge_score
pip install sacrebleu
pip install --upgrade gdown
pip install --upgrade datasets
```
## Dataset 
Dataset is already in ```datasets``` folder. Dataset consists of *train, val, test*

Here, we use ```medical_train```, ```medical_val```, ```medical_test``` to train model. There are 23801 samples, 2975 samples and 2976 samples, repectively. 
About other datasets such as ```*_half```, ```*_mini``` smaller than the main dataset, they are extracted from the main datasets and used to test model whether it works or not after coding. 

## Train

### Model with BiLSTM 
BiLSTM takes the last_hidden_states of DistilROBERTa, then it is put into GPT2 to compute cross-attention like an encoder-decoder model. GPT2 is frozen.
```python
python /content/k-distilroberta-gpt2/KDR_GPT2_BiLSTM.py \
        --pretrained_model_path ./models/KDRB_GPT2_model_BiLSTM.bin \
        --batch_size 4 \
        --epochs_num 15 \
        --max_length 210\
        --dropout 0.2 \
        --output_model_path ./models/KDRB_GPT2_model_BiLSTM.bin \
        --vocab_path ./roberta/vocab.json \
        --train_path ./datasets/medical_train.tsv \
        --dev_path ./datasets/medical_val.tsv \
        --test_path ./datasets/medical_test.tsv \
        --log_path ./logs \
        --kg_path ./brain/kgs/Medical.spo \
        --last_logging /content/drive/MyDrive/CEREBRO/K-BART/K-BERT/K-distilBERT-GPT2/model/logs/log_epoch_5.json
```

### Model without BiLSTM 
There is no BiLSTM layer in this model. GPT2 is frozen.
```python
python /content/k-distilroberta-gpt2/KDR_GPT2_Linear.py \
        --pretrained_model_path ./models/KDRB_GPT2_model_BiLSTM.bin \
        --batch_size 4 \
        --epochs_num 15 \
        --max_length 210\
        --dropout 0.2 \
        --output_model_path ./models/KDRB_GPT2_model_BiLSTM.bin \
        --vocab_path ./roberta/vocab.json \
        --train_path ./datasets/medical_train.tsv \
        --dev_path ./datasets/medical_val.tsv \
        --test_path ./datasets/medical_test.tsv \
        --log_path ./logs \
        --kg_path ./brain/kgs/Medical.spo \
        --last_logging /content/drive/MyDrive/CEREBRO/K-BART/K-BERT/K-distilBERT-GPT2/model/logs/log_epoch_5.json
```

Option of ```KDR_GPT2_*.py```
```
useage:     --pretrained_model_path - Path to the pre-trained model parameters. (str or None)
            --vocab_path - Path to the vocabulary file.
            --max_length - The max length of generated text
            --train_path - Path to the training dataset.
            --dev_path - Path to the validating dataset.
            --test_path - Path to the testing dataset.
            --epochs_num - The number of training epoches.
            --batch_size - Batch size of the training process.
            --kg_path - The name of knowledge graph, "HowNet", "CnDbpedia" or "Medical".
            --log_path - Path to the ```logs``` folder that saves the training progress information
            --output_model_path - Path to the output model.
            --last_logging - Path to ```.json``` logging file of previous epoch that is save in ```logs``` folder. (str or None)
```