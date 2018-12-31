import subprocess
import sys
import os



        # config['embedding'] = 'wiki'
        # config['seq_len'] = 100
        # config['bs'] = 512
        # config['dropout'] = 0.25
        # config['internal_dim'] = 100
        # config['lr'] = 0.01
        # config['func'] = lstm_gru_attn #function is used as a variable
        # config['top_k_checkpoints'] = 3 #for checkpoint ensemble-averaging (maximum of epochs)
        # config['number_to_x'] = False
        # config['block_math'] = False
        # config['swear_word_map'] = False
        # config['n_t_replace'] = False
        # config['lemmatize'] = False
        # config['two_split'] = False

    
cmdlist = [
    #"python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding glove --lr 0.01 --top_k_checkpoints 5".split(" "), 
    #"python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding glove --lr 0.01 --top_k_checkpoints 5 --number_to_x".split(" "),   
    "python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding glove --lr 0.01 --top_k_checkpoints 5 --block_math".split(" "),   
    "python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding glove --lr 0.01 --top_k_checkpoints 5 --swear_word_map".split(" "),   
    "python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding glove --lr 0.01 --top_k_checkpoints 5 --n_t_replace".split(" "),   
    "python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding glove --lr 0.01 --top_k_checkpoints 5 --two_split".split(" "),       
    
    
    # "python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding glove --lr 0.01 --top_k_checkpoints 5".split(" "),   
    #"python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding paragram --lr 0.01 --top_k_checkpoints 5".split(" "),    
    #"python Train.py --internal_dim 100 --seq_len 100 --dropout 0.25 --embedding wiki --lr 0.01 --top_k_checkpoints 5".split(" "),    
]
for cmd in cmdlist:
    logSignature = '_'.join(cmd[2:]).replace("--","_")
    F = open(os.path.join("..\\logs", "%s.out" %logSignature ),"wb")
    failure = subprocess.call(cmd, stdout=F)
    #failure = subprocess.call(cmd)
