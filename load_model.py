from email.policy import strict
from statistics import mode
import sys
from turtle import forward
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from multiprocessing import Process, Pool
import numpy as np
from constants import *
from config import *
from seed import *
from vocab import *
from modeling_distilbert import *
# from knowledge import *
from knowgraph import KnowledgeGraph
from optimizers import *
from model_saver import *
import traceback

path = ""
def parsers(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="D:\CEREBRO\\transformers\src\\transformers\models\distilbert\models\\03032022_cls_epoch5_batch8_data998\classifier_model.bin", type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="D:\CEREBRO\\transformers\src\\transformers\models\distilbert\models\classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="D:\CEREBRO\\transformers\src\\transformers\models\distilbert\models\\vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", default="D:\CEREBRO\\transformers\src\\transformers\models\distilbert\datasets\\train.tsv",type=str,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", default="D:\CEREBRO\\transformers\src\\transformers\models\distilbert\datasets\dev.tsv",type=str,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", default="D:\CEREBRO\\transformers\src\\transformers\models\distilbert\datasets\\test.tsv",type=str,
                        help="Path of the testset.")
    # parser.add_argument("--config_path", default="./models/google_config.json", type=str,
    #                     help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    # parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
    #                                                "cnn", "gatedcnn", "attn", \
    #                                                "rcnn", "crnn", "gpt", "bilstm"], \
    #                                                default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    # parser.add_argument("--subword_type", choices=["none", "char"], default="none",
    #                     help="Subword feature type.")
    # parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
    #                     help="Path of the subword vocabulary file.")
    # parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
    #                     help="Subencoder type.")
    # parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    # parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
    #                     help="Specify the tokenizer." 
    #                          "Original Google BERT uses bert tokenizer on Chinese corpus."
    #                          "Char tokenizer segments sentences into characters."
    #                          "Word tokenizer supports online word segmentation based on jieba segmentor."
    #                          "Space tokenizer segments sentences into words according to space."
    #                          )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", default="CnDbpedia",type=str, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()
    return args
class BertClassification(nn.Module): 
    def __init__(self,args, transformer):
        super(BertClassification, self).__init__()
        self.transformer = transformer
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim = -1)
        self.criterion = nn.NLLLoss()

    def forward(self, src, label, attention_mask, pos=None, vm = None):
        """
        Input: 
            - src:  is input_ids with dimension: [bs, seq_length]
            - label:    labels (0/1) with dimension: [bs]
            - attention_mask: is attention mask corresponding with input_ids with dimension: [bs, seq_length]

        Output: 
        """
        last_hidden_state = self.transformer(src, attention_mask, pos, vm)[0] #torch.tensor(bs, seq_length, dim)
        
        #Targer: 
        if self.pooling == "mean": 
            last_hidden_state = torch.mean(last_hidden_state, dim = 1)
        elif self.pooling == "max": 
            last_hidden_state = torch.max(last_hidden_state, dim = 1)[0]
        elif self.pooling == "last": 
            last_hidden_state = last_hidden_state[:, -1, :]
        else: 
            last_hidden_state = last_hidden_state[:, 0, :]
        last_hidden_state = torch.tanh(self.output_layer_1(last_hidden_state))
        logits = self.output_layer_2(last_hidden_state)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits
def main():
    args = parsers()
    
    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set) 

    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab
    # print("args: ", args)

    # #Build 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration = DistilBertConfig(dropout= args.dropout, max_position_embeddings=args.seq_length, output_hidden_states=True)
    model = DistilBertModel(configuration) 
    model = model.from_pretrained('distilbert-base-uncased')
    model = BertClassification(args, model)
    if args.pretrained_model_path is not None: 
        #initializw with pretrained model 
        print("Pretrained from {} is loaded.".format(args.pretrained_model_path))
        model.load_state_dict(torch.load(args.pretrained_model_path), strict = False)
    model = model.to(device)
    print("model: \n", model)

if __name__ == "__main__":
    main()