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

model_name = 'Rostlab/prot_bert_bfd'

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

        # seq = " ".join("".join(self.seqs[idx].split()))
        # seq = re.sub(r"[UZOB]", "X", seq)
        # seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length, add_special_tokens=True)
        # sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample = {}
        # sample['input_ids'] = torch.tensor(self.seqs[idx])
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
        # print(embedding)
        # length = embedding.shape[0]
        # padding = nn.ZeroPad2d(((max_length-length), 0, 0, 0))
        # embedding = padding(embedding)
        sequence.append( embedding)
        if (info[0] == "sp"):
            label.append(14)
        else:
            label.append(arg_dict[info[3]])
    #now sequence and label are filled
    print(len(sequence), len(label))
    df['input'] = sequence
    # df['input'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input']]
    # sequence = [ list(seq)[:max_length-2] for seq in df['input']]
    df['label'] = label
    return sequence, label, df

def collate_batch(batch):
    # https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
    text_list, classes = [], []
    print(type(batch))
    for i in range(len(batch)):
        sample = batch[i]
        # embed = fe([sample["Sequence"]])
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
# train_data = SeqIO.parse("./data/train.fasta", "fasta")
# val_data = SeqIO.parse("./data/val.fasta", "fasta")
# train_data = list(train_data)
# val_data = list(val_data)
# print(type(train_data[0]))
# print(vars(train_data[0]))

## embedder = ProtTransBertBFDEmbedder()

# train_dir = "./embeddings/train/"
# directory = os.fsencode(train_dir)
# directory = train_dir
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     info = filename.split('|')
#   for file in tqdm(os.listdir(seq_dir)):  embedding = torch.load(os.path.join(directory, filename))
#     print(info)
#     print(embedding)
#     print(embedding['representations'][33].shape)
#     print(embedding['mean_representations'][33].shape)
#     sys.exit(0)

seq_tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

max_length = 1280
num_class = len(arg_dict)

train_seqs, train_labels, train_df = seq_to_df("./embeddings/train/", arg_dict, max_length)
val_seqs, val_labels, val_df = seq_to_df("./embeddings/val/", arg_dict, max_length)

# print(val_labels)

# sys.exit(0)

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
#            shuffle=True
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
    nn.Dropout(0.7),
    nn.Linear(max_length//2, num_class)
)

model = model.to(device)

# print(model)

optimizer = optim.Adam(model.parameters(),
                  lr = 2e-5,
                )

# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 32

total_steps = len(train_dataloader) * num_epochs

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

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
        # print(b_input_ids)
        b_input_ids = b_input_ids.to(torch.float32)
        # b_input_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        optimizer.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # In PyTorch, calling `model` will in turn call the model's `forward` 
        # function and pass down the arguments. The `forward` function is 
        # documented here: 
        # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
        # The results are returned in a results object, documented here:
        # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
        # Specifically, we'll get the loss (because we provided labels) and the
        # "logits"--the model outputs prior to activation.
        result = model(b_input_ids)

        loss = loss_fn(result, b_labels)
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        dataloader.set_description(f"training loss: {float(loss):0.3f}")

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
  #       lr_scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
#     print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_val_accuracy = 0
    total_val_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch["input_ids"].to(device)
        # print(b_input_ids)
        b_input_ids = b_input_ids.to(torch.float32)
        # b_input_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids)

        # Get the loss and "logits" output by the model. The "logits" are the 
        # output values prior to applying an activation function like the 
        # softmax.
        loss = loss_fn(result, b_labels)
        # Accumulate the validation loss.
        total_val_loss += loss.item()

        # Move logits and labels to CPU
        _, result = torch.max(result, 1)
        result = result.to('cpu').numpy()
        labels = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        # print(result, labels)
        total_val_accuracy += f1_score(result, labels, average='micro')
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_val_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_val_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    # validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
  #  print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
   # training_stats.append(
    #    {
     #       'epoch': epoch_i + 1,
      #      'Training Loss': avg_train_loss,
       #     'Valid. Loss': avg_val_loss,
        #    'Valid. Accur.': avg_val_accuracy,
         #   'Training Time': training_time,
          #  'Validation Time': validation_time
#        }
 #   )

print("")
print("Training complete!")

# print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
# model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
# model_to_save.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

state = {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'seed': seed_val,
            }

torch.save(state, "./model_save/linear_esm_model.pt")

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))