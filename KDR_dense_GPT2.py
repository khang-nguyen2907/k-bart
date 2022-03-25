from distutils.log import info
from statistics import mode
import sys
import os
from turtle import forward
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
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, RobertaConfig, RobertaModel, RobertaTokenizer, EncoderDecoderModel, EncoderDecoderConfig, AutoConfig
from tqdm import tqdm
import time
from override import *

def add_knowledge_worker(params, decoder_tokenizer):

    p_id, sentences, columns, kg, vocab, args = params

    sentences_num = len(sentences)
    dataset = []
    # check_len_token = set()
    # check_len_pos = set()
    # check_len_vm = set()
    # check_len_id = set()
    # check_len_mask = set()
    
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
                    # check_len_token.add(len(tokens))
                    # if len(check_len_token) > 1: 
                    #     print("len: {0} || text: {1}".format(len(tokens), text))
                    pos = pos[0]
                    # check_len_pos.add(len(pos))
                    vm = vm[0].astype("bool")
                    # check_len_vm.add(vm.shape)

                    token_ids = [vocab.get(t) for t in tokens] #input_ids
                    # check_len_id.add(len(token_ids))
                    mask = [1 if t != PAD_TOKEN else 0 for t in tokens] #attention_mask
                    # check_len_mask.add(len(mask))

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
    # print("check length: ", check_len_token)
    # print("check_len_pos: ", check_len_pos)
    # print("check_len_id: ", check_len_id)
    # print("check_len_mask: ", check_len_mask)
    # print("check_len_vm: ", check_len_vm)
    return dataset

import numpy as np
def parsers(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/roberta/KDRB_GPT2_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/roberta/vocab.json", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_train.tsv",type=str,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_val.tsv",type=str,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_test.tsv",type=str,
                        help="Path of the testset.")
    parser.add_argument("--log_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/logs",type=str,
                        help="Path of the testset.")
    parser.add_argument("--last_logging", default=None,type=str,
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
    parser.add_argument("--type_vocab_size", type = int, default=3, 
                        help="type_vocab_size of encoder config.")
    
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
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()
    return args


def main():
    args = parsers()
    
    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    # labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
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
    print(200 * '-')
    print("Vocabulary Size: ", len(vocab))
    print(200 * '-')

    #Model
    #MODEL
    ##Tokenizer
    #make sure GPT2 appends EOS in begin and end 
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs 
    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

    encoder_tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    decoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #CLS token <s> will work as BOS token <s>
    encoder_tokenizer.bos_token = encoder_tokenizer.cls_token

    #SEP token </s> will work as EOS token </s>
    encoder_tokenizer.eos_token = encoder_tokenizer.sep_token

    #set pad_token_id to unk_token_id -> be careful here as unk_token_id  == eos_token_id == bos_token_id
    decoder_tokenizer.pad_token = decoder_tokenizer.unk_token
    
    ## Config
    encoder_config = AutoConfig.from_pretrained('distilroberta-base')
    decoder_config = AutoConfig.from_pretrained('gpt2')
    encoder_config.max_position_embeddings = args.seq_length_encoder
    encoder_config.hidden_dropout_prob = args.dropout
    encoder_config.type_vocab_size = args.type_vocab_size
    encoder_config.output_hidden_states = True 
    decoder_config.add_cross_attention = True
    decoder_config.use_cache = False #cache is currently not supported by EncoderDecoder framework 
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    ## Model
    model = KDRB_dense_GPT2(config=config, encoder = None, decoder = None)
    for parameter in model.decoder.parameters(): 
        parameter.requires_grad = False
    for parameter in model.decoder.lm_head.parameters(): 
        parameter.requires_grad = True
    rand_weight = torch.rand(model.decoder.lm_head.weight.shape)
    model.decoder.lm_head.weight = torch.nn.parameter.Parameter(rand_weight)
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
    if args.pretrained_model_path != "None": 
        print(200*"-")
        print("CONTINUE TO TRAIN THE MODEL FROM: {}".format(args.pretrained_model_path))
        print(200*'-')
        model.load_state_dict(torch.load(args.pretrained_model_path), strict = False)
    else: 
        print(200*"-")
        print("THE MODEL IS INITIALIZED FROM SCRATCH")
        print(200*'-')
    model = model.to(device)
    start_epoch = 1

    # Build knowledge graph.
    if args.kg_path == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_path]
    kg = KnowledgeGraph(txt_file=args.kg_path, predicate=True)

    last_epoch = 0
    best_result = 9999.0

    print("Best result before training: ", best_result)
    if args.last_logging != "None": 
        print(200*"-")
        print("LOADING LOGGING INFORMATION FROM {}".format(args.last_logging))
        last_logger = open(args.last_logging)
        logger_info = json.load(last_logger)
        last_epoch = logger_info['epoch']
        print("Previous epoch: ", last_epoch)
        last_loss = logger_info['total_loss']
        print("Previous loss: ", last_loss)
        best_result = last_loss
        start_epoch += last_epoch
        print("Previous best result: ", best_result)
        print("start_epoch: {0} || last_epoch: {1}".format(start_epoch, last_epoch + args.epochs_num + 1))
        print(200*'-')
    

    def read_dataset(path, decoder_tokenizer,workers_num=1):

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
    def compute_metrics(pred):
        labels_ids = pred.label_ids 
        pred_ids = pred.predictions 

        pred_str = decoder_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = decoder_tokenizer.eos_token_id
        label_str = decoder_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4), 
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    def evaluate(args, is_test):
        
        if is_test:
            dataset = read_dataset(args.test_path, decoder_tokenizer,workers_num=args.workers_num)
        else:
            dataset = read_dataset(args.dev_path, decoder_tokenizer,workers_num=args.workers_num)
        input_ids = torch.LongTensor([example[0] for example in dataset])
        label_ids = torch.LongTensor([example[1] for example in dataset])
        decoder_input_ids = torch.LongTensor([example[2] for example in dataset])
        decoder_attn_mask = torch.LongTensor([example[3] for example in dataset])
        mask_ids = torch.LongTensor([example[4] for example in dataset])
        pos_ids = torch.LongTensor([example[5] for example in dataset])
        vms = [example[6] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]

        if is_test:
            print("The number of evaluation instances: ", instances_num)
        
        model.eval()
        for i, (input_ids_batch, label_ids_batch, decoder_input_ids_batch, decoder_attn_mask_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, decoder_input_ids, decoder_attn_mask, mask_ids, pos_ids, vms)):

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
            rouge1_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rouge1"])["rouge1"].mid
            rouge2_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rouge2"])["rouge2"].mid
            rougeL_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rougeL"])["rougeL"].mid


            rouge1_precision = round(rouge1_output.precision, 4) 
            rouge1_recall = round(rouge1_output.recall, 4)
            rouge1_fmeasure = round(rouge1_output.fmeasure, 4)
            
            rouge2_precision = round(rouge2_output.precision, 4) 
            rouge2_recall = round(rouge2_output.recall, 4)
            rouge2_fmeasure = round(rouge2_output.fmeasure, 4)
            
            rougeL_precision = round(rougeL_output.precision, 4) 
            rougeL_recall = round(rougeL_output.recall, 4)
            rougeL_fmeasure = round(rougeL_output.fmeasure, 4)
            metrics_result = {
                "rouge1_precision" : rouge1_precision, 
                "rouge1_recall": rouge1_recall, 
                "rouge1_fmeasure": rouge1_fmeasure, 
                "rouge2_precision": rouge2_precision, 
                "rouge2_recall": rouge2_recall, 
                "rouge2_fmeasure": rouge2_fmeasure, 
                "rougeL_precision": rougeL_precision, 
                "rougeL_recall": rougeL_recall, 
                "rougeL_fmeasure": rougeL_fmeasure
            }
            if is_test: 
                print("report TEST: rouge1_precision: {0} \trouge1_recall: {1} \trouge1_fmeasure: {2}".format(rouge1_precision, rouge1_recall, rouge1_fmeasure))
                print("report TEST: rouge2_precision: {0} \trouge2_recall: {1} \trouge2_fmeasure: {2}".format(rouge2_precision, rouge2_recall, rouge2_fmeasure))
                print("report TEST: rougeL_precision: {0} \trougeL_recall: {1} \trougeL_fmeasure: {2}".format(rougeL_precision, rougeL_recall, rougeL_fmeasure))
                return metrics_result
            else: 
                print("report VAL: rouge1_precision: {0} \trouge1_recall: {1} \trouge1_fmeasure: {2}".format(rouge1_precision, rouge1_recall, rouge1_fmeasure))
                print("report VAL: rouge2_precision: {0} \trouge2_recall: {1} \trouge2_fmeasure: {2}".format(rouge2_precision, rouge2_recall, rouge2_fmeasure))
                print("report VAL: rougeL_precision: {0} \trougeL_recall: {1} \trougeL_fmeasure: {2}".format(rougeL_precision, rougeL_recall, rougeL_fmeasure))
                return metrics_result
        

    # Training phase.
    print(200 * '-')
    print("Start training.")
    trainset = read_dataset(args.train_path, decoder_tokenizer,workers_num=args.workers_num)
    print("Shuffling dataset")
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    print("input_ids")
    input_ids = torch.LongTensor([example[0] for example in trainset])
    print(input_ids.size())

    print("\nlabel_ids")
    label_ids = torch.LongTensor([example[1] for example in trainset])
    print(label_ids.size())

    print("\ndecoder_input_ids")
    decoder_input_ids = torch.LongTensor([example[2] for example in trainset])
    print(decoder_input_ids.size())

    print("\ndecoder_attn_mask")
    decoder_attn_mask = torch.LongTensor([example[3] for example in trainset])
    print(decoder_attn_mask.size())

    print("\nmask_ids")
    mask_ids = torch.LongTensor([example[4] for example in trainset])
    print(mask_ids.size())

    print("\npos_ids")
    pos_ids = torch.LongTensor([example[5] for example in trainset])
    print(pos_ids.size())

    print("\nvms")
    vms = [example[6] for example in trainset]
    print(len(vms))
    print(200 * '-')

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)
    report_step = int(instances_num//batch_size)
    args.report_steps = report_step
    print("report step: ",report_step)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)
    #addition part
    scaler = torch.cuda.amp.GradScaler()
    #
    total_loss = 0.
    result = 0.0     

    for epoch in range(start_epoch, last_epoch + args.epochs_num+1):
        print('\n')
        print(200 * '-')
        print("[EPOCH {}]".format(epoch))
        t1 = time.time()
        info = {}
        total_losses = []
        losses = []
        model.train()
        for i, (input_ids_batch, label_ids_batch, decoder_input_ids_batch, decoder_attn_mask_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(tqdm(batch_loader(batch_size, input_ids, label_ids, decoder_input_ids, decoder_attn_mask, mask_ids, pos_ids, vms))):
            model.zero_grad() #sets gradients of all model parameters to zero

            vms_batch = torch.LongTensor(vms_batch)
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            decoder_input_ids_batch = decoder_input_ids_batch.to(device)
            decoder_attn_mask_batch = decoder_attn_mask_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids_batch,
                    attention_mask=vms_batch,
                    decoder_input_ids=decoder_input_ids_batch,
                    decoder_attention_mask=decoder_attn_mask_batch,
                    encoder_outputs=None,
                    past_key_values=None,
                    inputs_embeds=None,
                    decoder_inputs_embeds=None,
                    labels=label_ids_batch,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                    position_ids = pos_ids_batch, 
                    token_type_ids = mask_ids_batch
                )
                loss = outputs[0]
            
            # outputs = model(
            #         input_ids=input_ids_batch,
            #         attention_mask=vms_batch,
            #         decoder_input_ids=decoder_input_ids_batch,
            #         decoder_attention_mask=decoder_attn_mask_batch,
            #         encoder_outputs=None,
            #         past_key_values=None,
            #         inputs_embeds=None,
            #         decoder_inputs_embeds=None,
            #         labels=label_ids_batch,
            #         use_cache=None,
            #         output_attentions=None,
            #         output_hidden_states=None,
            #         return_dict=None,
            #         position_ids = pos_ids_batch
            #     )
            # loss = outputs[0]

            losses.append(loss.item())
            if torch.cuda.device_count() >1: 
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i+1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_losses.append(total_loss / args.report_steps)
                sys.stdout.flush()
                total_loss = 0.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

        print("\nStart evaluation on dev dataset")
        result = evaluate(args, False)

        print("\nStart evaluation on test dataset.")
        rt = evaluate(args, True)
        
        t2 = time.time()
        info['epoch'] = int(epoch)
        info['total_loss'] = float(total_losses[-1])
        info['loss'] = losses
        info['val'] = result
        info['test'] = rt
        info['time'] = t2-t1
        path_log = os.path.join(args.log_path, "log_epoch_"+str(epoch)+".json")
        with open(path_log, mode = "w") as outfile: 
            json.dump(info, outfile)

        ttl = float(total_losses[-1])
        if ttl < best_result: 
            best_result = ttl
            save_model(model, args.output_model_path)

    #Evaluation phase 
    print("\nFinal evaluation on the test dataset")

    if torch.cuda.device_count()>1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else: 
        model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)
    print("\nTraining progress completed.")

if __name__ == "__main__": 
    main()
