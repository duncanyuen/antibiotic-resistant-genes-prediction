from Bio import SeqIO

train_data = list(SeqIO.parse("./data/train.fasta", "fasta"))
val_data = list(SeqIO.parse("./data/val.fasta", "fasta"))

max_len = 0
for i in range(len(train_data)):
    if len(train_data[i]._seq) > max_len:
        max_len = len(train_data[i]._seq)

for i in range(len(val_data)):
    if len(val_data[i]._seq) > max_len:
        max_len = len(val_data[i]._seq)

print(max_len)