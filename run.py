from override import *
from transformers import AutoTokenizer
from transformers import RobertaModel, GPT2Model, RobertaTokenizer, GPT2Tokenizer, EncoderDecoderModel
en_tok = AutoTokenizer.from_pretrained('distilroberta-base')
de_tok = AutoTokenizer.from_pretrained('gpt2')
model = EncoderDecoderModel.from_encoder_decoder_pretrained('distilroberta-base', 'gpt2')
model.decoder.config.use_cache = False

#CLS token will work as BOS token 
en_tok.bos_token = en_tok.cls_token

#SEP token will work as EOS token
en_tok.eos_token = en_tok.sep_token

#set pad_token_id to unk_token_id -> be careful here as unk_token_id  == eos_token_id == bos_token_id
de_tok.pad_token = de_tok.unk_token

#set decoding params 
model.config.decoder_start_token_id = de_tok.bos_token_id
model.config.eos_token_id = de_tok.eos_token_id
model.config.max_length = 256
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.length_penalty = 2.0 
model.config.num_beams = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

text = """can i treat pain from mouth uclers using advil? (ibuprofen). i've been experiencing pain from mouth ulcers for a few days now. and i was just wondering if it was possible to treat the pain with advil and if it would make a difference if i used it or not."""
labels = """you can take advil for mouth ulcers (aphthous stomatitis or canker sores). since this medication is anti-inflammatory this can help relieve some of the discomfort until they naturally go away."""

inputs = en_tok(text,padding="max_length", truncation=True, max_length=512)
input_ids = inputs.input_ids
attn_mask = inputs.attention_mask

outputs = de_tok(labels, padding="max_length", truncation=True, max_length=512)
decoder_input_ids = outputs.input_ids
decoder_attn_mask = outputs.attention_mask
label = outputs.input_ids.copy()

labelled = [-100 if mask == 0 else token for mask, token in zip(decoder_attn_mask, label)]

input_ids_batch = torch.LongTensor(input_ids).unsqueeze(0)
attn_mask_batch = torch.LongTensor(attn_mask).unsqueeze(0)
decoder_input_ids_batch = torch.LongTensor(decoder_input_ids).unsqueeze(0)
decoder_attn_mask_batch = torch.LongTensor(decoder_attn_mask).unsqueeze(0)
labelled_batch = torch.LongTensor(labelled).unsqueeze(0)

bilstm_gpt2 = GPT2LM_BiLSTM.from_pretrained('gpt2')
output_bilstm = bilstm_gpt2(
        input_ids=decoder_input_ids_batch,
        attention_mask=decoder_attn_mask_batch,
        labels = labelled_batch
)
# print(output_bilstm[0])
print(model.config)
# gpt2_lmmodel = GPT2LMHeadModel.from_pretrained('gpt2')
# gpt2_lmmodel_freeze = GPT2LM_Linear.from_pretrained('gpt2')
# gpt2_lmmodel_bi = GPT2LM_BiLSTM.from_pretrained('gpt2')
# total_params = sum(p.numel() for p in gpt2_lmmodel.parameters() if p.requires_grad)
# total_params_freeze = sum(p.numel() for p in gpt2_lmmodel_freeze.parameters() if p.requires_grad)
# total_params_bi =  sum(p.numel() for p in gpt2_lmmodel_bi.parameters() if p.requires_grad)

# print('\n')
# print(total_params)
# print(total_params_freeze)
# print(total_params_bi)