from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader

# define dataset class
class ProteinDataset(Dataset):
    def __init__(self, txt, labels):
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = {"Text": text, "Class": label}
        return sample



#load dataset
train_data = SeqIO.parse("./data/train.fasta", "fasta")
val_data = SeqIO.parse("./data/val.fasta", "fasta")
# print(type(train_data))
train_data = list(train_data)
print(type(train_data[0]))
print(vars(train_data[0]))
# for i in range(len(train_data)):
    

# dataloader = 

#load model

#fine-tune