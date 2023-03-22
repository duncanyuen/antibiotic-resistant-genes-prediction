import torch
import torch.nn as nn
import sys, os
from tqdm.auto import tqdm

CHECKPOINT_DIR = "./model_save/"
max_length = 1280
num_class = 15

model = nn.Sequential(
    nn.Linear(max_length, max_length//2),
    nn.Tanh(),
    nn.Dropout(0.3),
    nn.Linear(max_length//2, num_class)
)



state = torch.load(os.path.join(CHECKPOINT_DIR, 'linear_esm_model.pt'))
# print(state)
model.load_state_dict(state['model'])

seq_dir = "./embeddings/test/"

file_name = "submission.csv"

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

try:
    os.remove("./submission.csv")
except FileNotFoundError:
    print("File does not exist")


f = open(file_name, "a")
f.write("id,label\n")

for file in tqdm(os.listdir(seq_dir)):
    filename = os.fsdecode(file)
    embedding = torch.load(os.path.join(seq_dir, filename))
    embedding = list(embedding['mean_representations'].values())[0]
    result = model(embedding)
    print(result)
    idx = result.data.cpu().numpy().argmax()
    # label = list(labels.keys())[list(labels.values()).index(int(idx))]
    id = filename.split(".")
    f.write(str(id[0]) + "," + str(idx) + '\n')
    # sys.exit(0)
f.close()