import argparse
import torch

parser = argparse.ArgumentParser(
                    prog='Train colorization nn',
                    description='This script runs training of colorization neural network',
                    )

parser.add_argument('--name', help='defines name of experiment')           # positional argument
parser.add_argument('--epochs', type=int, help='how much training epochs need')      # option that takes a value
parser.add_argument('-bs', '--batch_size', type=int, help='how much training examples will be used per training iteration')      # option that takes a value
parser.add_argument('-m', '--modify', help='defines if modification will be used',
                    action='store_true')
parser.add_argument('--contunue', help='defines if train will be started from last save',
                    action='store_true')
parser.add_argument('--device', default='cuda:0', help='defines which c.u. will used; note, that multy-gpu not allowed')

if __name__=='__main__':
    opt = parser.parse_args()

    

