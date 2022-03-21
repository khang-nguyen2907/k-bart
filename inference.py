from distutils.log import info
from statistics import mode
import sys
import os
from turtle import forward
from unittest import skip
from unittest.util import _MAX_LENGTH
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from multiprocessing import Process, Pool
import numpy as np
from constant_roberta import *
from config import *
from seed import *
from knowledge_roberta import KnowledgeGraph
from optimizers import *
from model_saver import *
import traceback
import datasets
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, RobertaConfig, RobertaModel, RobertaTokenizer, EncoderDecoderModel
from tqdm import tqdm
import time
import numpy as np

def add_knowledge_worker(params, decoder_tokenizer):

    p_id, sentences, columns, kg, vocab, args = params

    sentences_num = len(sentences)
    dataset = []
    check_len_token = set()
    check_len_pos = set()
    check_len_vm = set()
    check_len_id = set()
    check_len_mask = set()
    
    count = 0
    for line_id, line in enumerate(sentences):
        count += 1
        # if line_id % sentences_num == 0:
        if count == 5000:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            count = 0
            sys.stdout.flush()
        line = line.strip().split('\t')
        try:
            if len(line) == 2:
                label = str(line[columns["answer"]])
                text = str(line[columns["question"]])
                # print("label: {0} || text: {1} ".format(label,text))
   
                tokens, pos, vm= kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length_decoder)
                if  len(tokens)==0 and len(pos)==0 and len(vm)==0:
                    continue
                else: 
                    tokens = tokens[0]
                    check_len_token.add(len(tokens))
                    if len(check_len_token) > 1: 
                        print("len: {0} || text: {1}".format(len(tokens), text))
                    pos = pos[0]
                    check_len_pos.add(len(pos))
                    vm = vm[0].astype("bool")
                    check_len_vm.add(vm.shape)

                    token_ids = [vocab.get(t) for t in tokens] #input_ids
                    check_len_id.add(len(token_ids))
                    mask = [1 if t != PAD_TOKEN else 0 for t in tokens] #attention_mask
                    check_len_mask.add(len(mask))

                    outputs = decoder_tokenizer(label, padding="max_length", truncation=True, max_length = args.seq_length_decoder)
                    decoder_input_ids = outputs.input_ids
                    labels = outputs.input_ids.copy()
                    decoder_attn_mask = outputs.attention_mask
                    labelled = [-100 if mask == 0 else token for mask, token in zip(decoder_attn_mask, labels)]

                    dataset.append((token_ids, labelled, decoder_input_ids, decoder_attn_mask, mask, pos, vm))
            
            elif len(line) == 3:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm))
            
            elif len(line) == 4:  # for dbqa
                qid=int(line[columns["qid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = CLS_TOKEN + text_a + SEP_TOKEN + text_b + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length_decoder)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1
                
                dataset.append((token_ids, label, mask, pos, vm, qid))
            else:
                pass
            
        except Exception:
            print("TRACEBACK: \n", traceback.format_exc())
            print("Error line: ", line)
            print("len line: ",len(line))
            return
            # print("Error line: ", line)
    print("check length: ", check_len_token)
    print("check_len_pos: ", check_len_pos)
    print("check_len_id: ", check_len_id)
    print("check_len_mask: ", check_len_mask)
    print("check_len_vm: ", check_len_vm)
    return dataset

def read_dataset(args, path, columns, kg, vocab,decoder_tokenizer,workers_num=1):
    print("Loading sentences from {}".format(path))
    sentences = []
    with open(path, mode='r', encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            sentences.append(line)
    sentence_num = len(sentences)

    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append((i, sentences[i*sentence_per_block: (i+1)*sentence_per_block], columns, kg, vocab, args))
        pool = Pool(workers_num)
        res = pool.map(add_knowledge_worker, params)
        pool.close()
        pool.join()
        dataset = [sample for block in res for sample in block]
    else:
        params = (0, sentences, columns, kg, vocab, args)
        # print("sentences: \n", sentences)
        # print("columns: \n", columns)
        # print("KG: \n", kg if kg is not None else None)
        # print("Vocab: \n", vocab if vocab is not None else None)
        dataset = add_knowledge_worker(params, decoder_tokenizer)

    return dataset
def parsers(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--vocab_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/roberta/vocab.json", type=str,
                        help="Path of the vocabulary file.") 
    parser.add_argument("--test_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_test.tsv",type=str,
                        help="Path of the testset.")
    parser.add_argument("--logging_folder", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/logs",type=str,
                        help="Path of the testset.")
    parser.add_argument("--anyway", default=True,type=bool,
                        help="Path of the testset.")
    

    # Model options.
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size.")
    parser.add_argument("--seq_length_encoder", type=int, default=512,
                        help="Sequence length of encoder.")
    parser.add_argument("--seq_length_decoder", type=int, default=512,
                        help="Sequence length of decoder.")
    parser.add_argument("--max_length", type=int, default = 256, 
                        help= "max length.")
    parser.add_argument("--min_length", type = int, default=50, 
                        help="Min length.")
    
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.3,
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
    parser.add_argument("--kg_path", default="/content/k-distilroberta-gpt2/brain/kgs/Medical.spo",type=str, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    

    args = parser.parse_args()
    return args

def main():
    args = parsers()
    
    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    # labels_set = set()
    if not args.anyway:
        columns = {}
        with open(args.test_path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                try:
                    line = line.strip().split("\t")
                    if line_id == 0:
                        for i, column_name in enumerate(line):
                            columns[column_name] = i
                        continue
                except:
                    pass    

    f = open(args.vocab_path)
    vocab = json.load(f)
    f.close()
    args.vocab = vocab
    print("Vocabulary Size: ", len(vocab))

    #Model
    #make sure GPT2 appends EOS in begin and end 
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs 
    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

    encoder_tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    decoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('distilroberta-base', 'gpt2')
    model.encoder.config.max_position_embeddings = args.seq_length_encoder
    model.encoder.config.dropout = args.dropout
    model.encoder.config.output_hidden_states = True 
    model.decoder.config.add_cross_attention = True

    #cache is currently not supported by EncoderDecoder framework 
    model.decoder.config.use_cache = False

    #CLS token will work as BOS token 
    encoder_tokenizer.bos_token = encoder_tokenizer.cls_token

    #SEP token will work as EOS token
    encoder_tokenizer.eos_token = encoder_tokenizer.sep_token

    #set pad_token_id to unk_token_id -> be careful here as unk_token_id  == eos_token_id == bos_token_id
    decoder_tokenizer.pad_token = decoder_tokenizer.unk_token

    #set decoding params 
    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    model.config.eos_token_id = decoder_tokenizer.eos_token_id
    model.config.max_length = args.max_length
    model.config.min_length = args.min_length
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0 
    model.config.num_beams = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.pretrained_model_path is not None: 
        print("CONTINUE TO TRAIN THE MODEL FROM: {}".format(args.pretrained_model_path))
        model.load_state_dict(torch.load(args.pretrained_model_path), strict = False)
    model = model.to(device)

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(txt_file="/content/k-distilroberta-gpt2/brain/kgs/Medical.spo", predicate=True)

    def batch_loader(batch_size, input_ids, label_ids, decoder_input_ids, decoder_attn_mask, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size, :]
            decoder_input_ids_batch = decoder_input_ids[i*batch_size: (i+1)*batch_size, :]
            decoder_attn_mask_batch = decoder_attn_mask[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vms_batch = vms[i*batch_size: (i+1)*batch_size]
            yield input_ids_batch, label_ids_batch, decoder_input_ids_batch, decoder_attn_mask_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:, :]
            decoder_input_ids_batch = decoder_input_ids[instances_num//batch_size*batch_size:, :]
            decoder_attn_mask_batch = decoder_attn_mask[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vms_batch = vms[instances_num//batch_size*batch_size:]
            yield input_ids_batch, label_ids_batch, decoder_input_ids_batch, decoder_attn_mask_batch, mask_ids_batch, pos_ids_batch, vms_batch

    rouge = datasets.load_metric("rouge")

    if args.anyway: 
        input_sentence = str(input("Please enter a question (^_^): \n"))
        tokens, pos, vm= kg.add_knowledge_with_vm([input_sentence], add_pad=True, max_length=args.seq_length_decoder)
    
        tokens = tokens[0]
        pos = torch.LongTensor(pos[0]).unsqueeze(0)
        vm = torch.LongTensor(vm[0].astype("bool")).unsqueeze(0)
        token_ids = [vocab.get(t) for t in tokens] #input_ids
        token_ids = torch.LongTensor(token_ids)
        mask = [1 if t != PAD_TOKEN else 0 for t in tokens] #attention_mask
        mask = torch.LongTensor(mask)
        model.eval()
        outputs = model.generate(
            token_ids, 
            do_sample = True, 
            min_length = args.min_length,
            max_length = args.max_length, 
            top_k = 50, 
            top_p = 0.95, 
        )
        answer = decoder_tokenizer.decode(outputs[0], skip_special_tokens= True)
        print("ANSWER: \n" + 100 * '-')
        print(answer)
    else:
        dataset = read_dataset(args,args.test_path,columns, kg, vocab, decoder_tokenizer,workers_num=args.workers_num)
        input_ids = torch.LongTensor([example[0] for example in dataset])
        label_ids = torch.LongTensor([example[1] for example in dataset])
        decoder_input_ids = torch.LongTensor([example[2] for example in dataset])
        decoder_attn_mask = torch.LongTensor([example[3] for example in dataset])
        mask_ids = torch.LongTensor([example[4] for example in dataset])
        pos_ids = torch.LongTensor([example[5] for example in dataset])
        vms = [example[6] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]

        print("The number of evaluation instances: ", instances_num)
        model.eval()
        for i,(input_ids_batch, label_ids_batch, decoder_input_ids_batch, decoder_attn_mask_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, decoder_input_ids, decoder_attn_mask, mask_ids, pos_ids, vms)):
            inff = {}
            vms_batch = torch.LongTensor(vms_batch)
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            decoder_input_ids_batch = decoder_input_ids_batch.to(device)
            decoder_attn_mask_batch = decoder_attn_mask_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)
    
            pred_ids = model.generate(
                input_ids_batch,
                do_sample = True, 
                min_length = args.min_length, 
                max_length = args.max_length, 
                top_k = 50, 
                top_p = 0.95
                )
            pred_str = decoder_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_ids_batch[label_ids_batch==-100] = decoder_tokenizer.eos_token_id
            label_str = decoder_tokenizer.batch_decode(label_ids_batch, skip_special_tokens=True)
            rouge_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rouge2"])["rouge2"].mid
            rouge2_precision = round(rouge_output.precision, 4) 
            rouge2_recall = round(rouge_output.recall, 4)
            rouge2_fmeasure = round(rouge_output.fmeasure, 4)
            result_eval = {
                "rouge2_precision": rouge2_precision, 
                "rouge2_recall": rouge2_recall, 
                "rouge2_fmeasure": rouge2_fmeasure
            }
            inff["metric"] = result_eval
            for id, (pred, lab) in enumerate(zip(pred_str,label_str)): 
                inff[id] = {
                    "labels": lab, 
                    "prediction": pred
                }
            print("Testing result is saved in {}".format(args.logging_folder))
            with open(os.path.join(args.logging_folder, "test_"+str(i)+".json"), mode="w") as outfile: 
                json.dump(inff, outfile)

    
    