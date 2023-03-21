from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from bio_embeddings.embed import ProtTransBertBFDEmbedder
# from bio_embeddings.embed import ProtTransBertBFDEmbedder
# from transformers import T5Tokenizer, T5EncoderModel
# from transformers import BertTokenizer, BertEncoderModel
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import re

from tqdm import tqdm
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
# print(vars(tokenizer))
# tokenizer = tokenizer.to(device)
# model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert") #.to(device)
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)

# define dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequence, labels):
        self.labels = labels
        self.sequence = sequence

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sequence = self.sequence[idx]
        # print(label)
        sample = {"Sequence": sequence, "Class": label}
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
            print(sys.getsizeof(embedding_rpr))
    return embeddings

def seq_to_df(seq_records, arg_dict, batch_size):
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
    data = {'sequence': sequence,
            'label': label}
    df = pd.DataFrame(data)
    return df

def collate_batch(batch):
    # https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
    text_list, classes = [], []
    print(batch)
    for (_text, _class) in batch:
        embed = fe([_text])
        text_list.append(embed)
        classes.append(_class)
    text = torch.tensor(text_list)
    print(classes)
    classes = torch.tensor(classes)
    print(text.shape, classes.shape)
    return text, classes

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

batch_size = 16

train_df = seq_to_df(train_data, arg_dict, batch_size)
val_df = seq_to_df(val_data, arg_dict, batch_size)

train_dataset = ProteinDataset(sequence=train_df['sequence'], labels=train_df['label'])
val_dataset = ProteinDataset(sequence=val_df['sequence'], labels=val_df['label'])

torch.save(train_dataset, "./traindata.pt")
torch.save(val_dataset, "./valdata.pt")

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = collate_batch)

for (idx, batch) in enumerate(train_dataloader):    # Print the 'text' data of the batch
    print(idx, 'Text data: ', batch['Sequence'])    # Print the 'class' data of batch
    print(idx, 'Class data: ', batch['Class'], '\n')
    break

#load model

#fine-tune