from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, BertTokenizerFast, EvalPrediction
from transformers import BertForSequenceClassification, BertConfig
import torch
import re
import numpy as np
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import os, sys
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

        sample = {}
        sample['input_ids'] = self.seqs[idx]
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample

def chunk_list(in_list, batch_size):
    out_list = [in_list[i:i+batch_size] for i in range(0, len(in_list), batch_size)]
    return out_list

def tokenize(seqs, batch_size):
    # code taken from https://github.com/agemagician/ProtTrans
    seqs_normal = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]
    seqs_batched = chunk_list(seqs_normal, batch_size)
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(seqs_batched))):
            embedding_rpr = fe(seqs_batched[i])
            embeddings += embedding_rpr
    return embeddings

def seq_to_df(seq_dir, arg_dict, max_length):
    df = pd.DataFrame()
    sequence = []
    label = []
    for file in tqdm(os.listdir(seq_dir)):
        filename = os.fsdecode(file)
        info = filename.split('|')
        embedding = torch.load(os.path.join(seq_dir, filename))
        embedding = list(embedding['mean_representations'].values())[0]
        sequence.append( embedding)
        if (info[0] == "sp"):
            label.append(14)
        else:
            label.append(arg_dict[info[3]])
    #now sequence and label are filled
    print(len(sequence), len(label))
    df['input'] = sequence
    df['label'] = label
    return sequence, label, df

def collate_batch(batch):
    # https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
    text_list, classes = [], []
    print(type(batch))
    for i in range(len(batch)):
        sample = batch[i]
        text_list.append(sample["Sequence"])
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
  return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=13)

def sample_weight(data_folder, num_class):
    # https://discuss.pytorch.org/t/class-imbalance-with-weightedrandomsampler/81758/3
    class_sample_count = np.array([len([i for i in data_folder.labels if i == t]) for t in range(0, num_class)])
    weight = 1 / class_sample_count
    samples_weight = np.array([weight[t] for t in data_folder.labels])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight,
                                    len(samples_weight))
    return sampler

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

seq_tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

max_length = 1280
num_class = len(arg_dict)

train_seqs, train_labels, train_df = seq_to_df("./embeddings/train/", arg_dict, max_length)
val_seqs, val_labels, val_df = seq_to_df("./embeddings/val/", arg_dict, max_length)

train_dataset = ProteinDataset(sequence=train_df['input'], labels=train_df['label'], max_length=max_length)
val_dataset = ProteinDataset(sequence=val_df['input'], labels=val_df['label'], max_length=max_length)

train_sampler = sample_weight(train_dataset, num_class=num_class)

torch.save(train_dataset, "./traindata.pt")
torch.save(val_dataset, "./valdata.pt")

# https://mccormickml.com/2019/07/22/BERT-fine-tuning/

batch_size = 32

train_dataloader = DataLoader(
            train_dataset,
            sampler = train_sampler,
            batch_size = batch_size,
        )

validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )

#load model

model = nn.Sequential(
    nn.Linear(max_length, max_length//2),
    nn.Tanh(),
    nn.Dropout(0.5),
    nn.Linear(max_length//2, num_class)
)

model = model.to(device)

optimizer = optim.Adam(model.parameters(),
                  lr = 2e-5,
                )

loss_fn = nn.CrossEntropyLoss()

num_epochs = 64

total_steps = len(train_dataloader) * num_epochs

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 20054 #乖

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.

# For each epoch...
for i in range(num_epochs):
    model.train()
    steps = 0
    total_train_loss = 0.0
    dataloader = tqdm(enumerate(train_dataloader))
    for step, batch in dataloader: #per batch
        b_input_ids = batch["input_ids"].to(device)
        b_input_ids = b_input_ids.to(torch.float32)
        b_labels = batch["labels"].to(device)

        optimizer.zero_grad()

        result = model(b_input_ids)

        loss = loss_fn(result, b_labels)
        total_train_loss += loss.item()

        loss.backward()

        dataloader.set_description(f"training loss: {float(loss):0.3f}")

        optimizer.step()


    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("Running Validation...")

    model.eval()

    total_val_accuracy = 0
    total_val_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:

        b_input_ids = batch["input_ids"].to(device)
        b_input_ids = b_input_ids.to(torch.float32)
        b_labels = batch["labels"].to(device)
        
        with torch.no_grad():        

            result = model(b_input_ids)

        loss = loss_fn(result, b_labels)
        total_val_loss += loss.item()

        _, result = torch.max(result, 1)
        result = result.to('cpu').numpy()
        labels = b_labels.to('cpu').numpy()

        total_val_accuracy += f1_score(result, labels, average='micro')
        

    avg_val_accuracy = total_val_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_val_loss / len(validation_dataloader)
    
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

print("")
print("Training complete!")


output_dir = './model_save/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

state = {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'seed': seed_val,
            }

torch.save(state, "./model_save/linear_esm_model.pt")
