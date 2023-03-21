from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from bio_embeddings.embed import ProtTransBertBFDEmbedder
# from bio_embeddings.embed import ProtTransBertBFDEmbedder
from transformers import T5Tokenizer, T5EncoderModel
import torch

from tqdm import tqdm

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
        sample = {"Sequence": sequence, "Class": label}
        return sample

def seq_to_df(seq_records, arg_dict):
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
    # embeddings = embedder.embed_many(sequence)
    # embeddings = [ProtTransBertBFDEmbedder.reduce_per_protein(e) for e in list(embeddings)]
    data = {'sequence': sequence,
            'label': label}
    df = pd.DataFrame(data)
    return df



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

train_df = seq_to_df(train_data, arg_dict)
val_df = seq_to_df(val_data, arg_dict)

# for i in range(len(train_data)):
    

# dataloader = 

#load model

#fine-tune