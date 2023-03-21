from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from bio_embeddings.embed import ProtTransBertBFDEmbedder
# from bio_embeddings.embed import ProtTransBertBFDEmbedder
# from transformers import T5Tokenizer, T5EncoderModel
# from transformers import BertTokenizer, BertEncoderModel
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, BertTokenizerFast, EvalPrediction
import torch
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
import os, sys

model_name = 'Rostlab/prot_bert_bfd'

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
# print(vars(tokenizer))
# tokenizer = tokenizer.to(device)
# model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
# tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
# model = AutoModel.from_pretrained("Rostlab/prot_bert") #.to(device)
# fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)

# define dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequence, labels, tokenizer_name=model_name, max_length=1024):
        self.labels = labels
        self.seqs = sequence
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length, add_special_tokens=True)
  #       print(seq_ids)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        # print(label)
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample

def chunk_list(in_list, batch_size):
    out_list = [in_list[i:i+batch_size] for i in range(0, len(in_list), batch_size)]
    return out_list

def tokenize(seqs, batch_size):
    # code taken from https://github.com/agemagician/ProtTrans
    seqs_normal = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]
    seqs_batched = chunk_list(seqs_normal, batch_size)
    # ids = tokenizer.batch_encode_plus(seqs_normal, add_special_tokens=True, padding="longest")
    # input_ids = torch.tensor(ids['input_ids']).to(device)
    # attention_mask = torch.tensor(ids['attention_mask']).to(device)
    # print(type(input_ids), input_ids.shape)
    # print(type(attention_mask), attention_mask.shape)
    # input_ids = torch.split(input_ids, batch_size, 0)
    # attention_mask = torch.split(attention_mask, batch_size, 0)
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(seqs_batched))):
            embedding_rpr = fe(seqs_batched[i])
            # embedding_rpr = fe(input_ids=input_ids[i],attention_mask=attention_mask[i])
            # embeddings += torch.split(embedding_rpr.last_hidden_state, 1, 0)
            embeddings += embedding_rpr
#             print(sys.getsizeof(embedding_rpr))
    return embeddings

def seq_to_df(seq_records, arg_dict, max_length):
    df = pd.DataFrame()
    sequence = []
    label = []
    for i in tqdm(range(len(seq_records))):
        sequence.append(str(seq_records[i]._seq))
        #translate
        info = seq_records[i].id.split('|')
        if (info[0] == "sp"):
            label.append(14)
        else:
            label.append(arg_dict[info[3]])
    #now sequence and label are filled
    # embeddings = tokenize(sequence, batch_size)
    # print(embeddings.shape)
    df['input'] = ["".join(seq.split()) for seq in sequence]
    df['input'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input']]
    sequence = [ list(seq)[:max_length-2] for seq in df['input']]
    df['label'] = [l for l in label]
    return sequence, label, df

def collate_batch(batch):
    # https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
    text_list, classes = [], []
    print(type(batch))
    for i in range(len(batch)):
        sample = batch[i]
        embed = fe([sample["Sequence"]])
        text_list.append(embed)
        classes.append(sample["Class"])
    text = torch.tensor(text_list)
    print(classes)
    classes = torch.tensor(classes)
    print(text.shape, classes.shape)
    return text, classes



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def model_init(num_labels=14):
  return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=14)

# https://github.com/sacdallago/bio_embeddings/blob/develop/notebooks/embed_fasta_sequences.ipynb

#load dataset
arg_dict = {'aminoglycoside': 0,
            'macrolide-lincosamide-streptogramin': 1,
            'polymyxin': 2,
            'fosfomycin': 3,
            'trimethoprim': 4,
            'bacitracin': 5,
            'quinolone': 6,
            'multidrug': 7,
            'chloramphenicol': 8,
            'tetracycline': 9,
            'rifampin': 10,
            'beta_lactam': 11,
            'sulfonamide': 12,
            'glycopeptide': 13,
            'nonarg': 14
            }
train_data = SeqIO.parse("./data/train.fasta", "fasta")
val_data = SeqIO.parse("./data/val.fasta", "fasta")
# print(type(train_data))
train_data = list(train_data)
val_data = list(val_data)
print(type(train_data[0]))
print(vars(train_data[0]))

# embedder = ProtTransBertBFDEmbedder()

seq_tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

batch_size = 16
max_length = 1576

train_seqs, train_labels, train_df = seq_to_df(train_data, arg_dict, max_length)
val_seqs, val_labels, val_df = seq_to_df(val_data, arg_dict, max_length)

train_dataset = ProteinDataset(sequence=train_df['input'], labels=train_df['label'])
val_dataset = ProteinDataset(sequence=val_df['input'], labels=val_df['label'])

torch.save(train_dataset, "./traindata.pt")
torch.save(val_dataset, "./valdata.pt")



# train_seqs_encodings = seq_tokenizer(train_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)

# train_dataloader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = collate_batch)

# for (idx, batch) in enumerate(train_dataloader):    # Print the 'text' data of the batch
#     print(batch)
#     print(idx, 'Text data: ', batch['Sequence'])    # Print the 'class' data of batch
#     print(idx, 'Class data: ', batch['Class'], '\n')
#     break

#load model


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=10,   # batch size for evaluation
    warmup_steps=1000,               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after eachh epoch
    gradient_accumulation_steps=64,  # total number of steps before back propagation
    fp16=True,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
    run_name="ProBert-BFD-ARG",       # experiment name
    seed=3                           # Seed for experiment reproducibility 3x3
)

trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,    # evaluation metrics
)

trainer.train()

trainer.save_model('models/')
#fine-tune