import subprocess
import sys
import os


    # config['embedding_file'] = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    # config['seq_len'] = 100
    # config['bs'] = 64
    # config['dropout'] = 0.25
    # config['internal_dim'] = 100
    # config['lr'] = 0.01
    # config['func'] = lstm_gru_attn #function is used as a variable

    
cmdlist = [
    "python Models.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding glove --lr 0.01 --top_k_checkpoints 5".split(" "),    
    "python Models.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding paragram --lr 0.01 --top_k_checkpoints 5".split(" "),    
    "python Models.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding wiki --lr 0.01 --top_k_checkpoints 5".split(" "),    
]
for cmd in cmdlist:
    logSignature = '_'.join(cmd[2:]).replace("--","_")
    F = open(os.path.join("..\\logs", "%s.out" %logSignature ),"wb")
    failure = subprocess.call(cmd, stdout=F)
