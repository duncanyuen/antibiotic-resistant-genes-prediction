from Bio import SeqIO

#load dataset
train_data = SeqIO.parse("./data/train.fasta", "fasta")
val_data = SeqIO.parse("./data/val.fasta", "fasta")
print(type(train_data))
train_data = list(train_data)
print(type(train_data[0]))

#load model

#fine-tune