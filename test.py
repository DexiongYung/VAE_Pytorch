from model.AutoEncoder import AutoEncoder
import numpy as np
import argparse
import torch
import json

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Name to test', type=str, default='Dylan')
parser.add_argument('--model_name',
                    help='JSON config and weight name', type=str, default='large_params')
args = parser.parse_args()

NAME = args.name
MODEL_NAME = args.model_name

DEVICE = "cpu"

json_file = json.load(open(f'json/{MODEL_NAME}.json', 'r'))
t_args = argparse.Namespace()
t_args.__dict__.update(json_file)
args = parser.parse_args(namespace=t_args)

SOS_IDX = args.sos_idx
PAD_IDX = args.pad_idx
EOS_IDX = args.eos_idx
VOCAB = args.vocab
INVERTED_VOCAB = {v: k for k, v in VOCAB.items()}

model = AutoEncoder(args.vocab, args.pad_idx,
                    args.max_name_length, DEVICE, args)
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

for name in output:
    print(''.join(name))
