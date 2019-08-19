from SANPolyA import *
import re
import os, sys, copy, getopt, re, argparse
import random
import pandas as pd 
import numpy as np


def dataProcessing(sequence):

    alphabet = np.array(['A', 'G', 'T', 'C'])

    line = list(sequence.strip('\n'));
    
    seq = np.array(line, dtype = '|U1').reshape(-1,1);
    seq_data = []

    for i in range(len(seq)):
        if seq[i] == 'A':
            seq_data.append([1,0,0,0])
        if seq[i] == 'T':
            seq_data.append([0,1,0,0])
        if seq[i] == 'C':
            seq_data.append([0,0,1,0])
        if seq[i] == 'G':
            seq_data.append([0,0,0,1])
 
    return np.array(seq_data).reshape(1,206,4) #(n, 41, 4), (n,)

def main():

    parser = argparse.ArgumentParser(description="identification of PAS signals")
    parser.add_argument("--h5File", type=str, help="the model weights file", required=True)
    parser.add_argument("--sequence", type=str, help="input 206bp long sequence", required=True)
    args = parser.parse_args()

    Path = os.path.abspath(args.h5File)
    sequence = args.sequence
    sequence = re.sub('\s', '', sequence);


    if not os.path.exists(Path):
        print("The model not exist! Error\n")
        sys.exit()

    # print(len(sequence))
    # print(sequence)

    seq = dataProcessing(sequence)
    keras_Model = SANPolyA();
    keras_Model.load_weights(Path)
    pred = keras_Model.predict(seq)

    print('probablity of positive PAS signal sample is ' + str(pred[0][0]))
    
if __name__ == "__main__":
    main()