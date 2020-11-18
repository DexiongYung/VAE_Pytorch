from model.AutoEncoder import VariationalAutoEncoder
import numpy as np
import argparse
import torch
import json

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Name to test', type=str, default='Dyln')
parser.add_argument('--model_name',
                    help='JSON config and weight name', type=str, default='new_eps')
args = parser.parse_args()

NAME = args.name
MODEL_NAME = args.model_name

DEVICE = "cpu"

json_file = json.load(open(f'json/{MODEL_NAME}.json', 'r'))
t_args = argparse.Namespace()
t_args.__dict__.update(json_file)
args = parser.parse_args(namespace=t_args)

SOS_IDX = args.sos_idx
EOS_IDX = args.eos_idx
VOCAB = args.vocab
INVERTED_VOCAB = {v: k for k, v in VOCAB.items()}

model = VariationalAutoEncoder(DEVICE, args)
model.load(f'weight/{MODEL_NAME}.path.tar')

length_tensor = torch.LongTensor([len(NAME)])
input_tensor = torch.LongTensor(
    list(map(VOCAB.get, INVERTED_VOCAB[SOS_IDX] + NAME))).unsqueeze(0)

_, probs_tensor, _, _ = model.forward(
    input_tensor, length_tensor, is_teacher_force=True)
arg_max = torch.argmax(probs_tensor, dim=2)
arg_max_list = arg_max.tolist()
output = [list(map(INVERTED_VOCAB.get, arg_max_list[i]))
          for i in range(len(arg_max_list))]

probs = []

for i in range(probs_tensor.shape[1] - 1):
    idx = input_tensor[0, i+1].item()
    prob = probs_tensor[0, i, idx].item()
    probs.append(prob)

probs.append(probs_tensor[0, probs_tensor.shape[1]-1, EOS_IDX].item())

print(np.min(probs))

for name in output:
    print(''.join(name))
