from collections import OrderedDict

def prototype_state():
    state = {}

    state['train_file'] = 'tmp/informfood_train.pkl'
    state['valid_file'] = 'tmp/informfood_dev.pkl'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['eos_sym'] = 0	# end of system action
    state['fill_sym'] = -1

    state['seq_len_in'] = 280
    state['seq_len_out'] = 10

    # ----- ACTIV ---- 
    state['active'] = 'lambda x: T.tanh(x)'
    
    # ----- SIZES ----
    state['prefix'] = 'model_'
    state['word_dim'] = 700
    state['emb_dim'] = 256
    state['h_dim'] = 256

    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 100
    state['cost_threshold'] = 1.003

     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  
    # Batch size
    state['bs'] = 500
    # Sort by length groups of  
    state['sort_k_batches'] = 1
   
    # Maximum number of iterations
    state['max_iters'] = 10
    # Modify this in the prototype
    state['save_dir'] = './model'
    
    # ----- TRAINING PROCESS -----
    # Frequency of training error reports (in number of batches)
    state['train_freq'] = 1
    # Validation frequency
    state['valid_freq'] = 50
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1


    return state

