from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from bio_embeddings.embed import ProtTransBertBFDEmbedder

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

def seq_to_df(seq_records):
    df = pd.DataFrame(columns=['sequence', 'label'])
    for i in range(len(seq_records)):
# https://github.com/sacdallago/bio_embeddings/blob/develop/notebooks/embed_fasta_sequences.ipynb

#load dataset
train_data = SeqIO.parse("./data/train.fasta", "fasta")
val_data = SeqIO.parse("./data/val.fasta", "fasta")
# print(type(train_data))
train_data = list(train_data)
print(type(train_data[0]))
print(vars(train_data[0]))

train_df = pd.DataFrame()
val_df = pd.DataFrame()

# for i in range(len(train_data)):
    

# dataloader = 

#load model

#fine-tune