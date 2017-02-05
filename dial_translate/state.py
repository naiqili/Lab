from collections import OrderedDict

def prototype_state():
    state = {}

    state['train_file'] = 'tmp/train_data_coded.pkl'
    state['valid_file'] = 'tmp/dev_data_coded.pkl'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['eos_sym'] = 0	# end of system action

    # ----- ACTIV ---- 
    state['activation'] = 'lambda x: T.tanh(x)'
    
    # ----- SIZES ----
    state['prefix'] = 'model_'
    state['word_dim'] = 700
    state['emb_dim'] = 256
    state['h_dim'] = 512
    state['rnnh_dim'] = 512

    state['margin'] = 50
    state['noise_cnt'] = 50
    state['acttype_cnt'] = 35
    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 5
    state['cost_threshold'] = 1.003

     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  
    # Maximum sequence length / trim batches
    state['seqlen'] = 280
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
    state['valid_freq'] = 20
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1


    return state

def simple_state():
    state = {}

    state['train_file'] = 'tmp/simple_train_data_coded.pkl'
    state['valid_file'] = 'tmp/simple_dev_data_coded.pkl'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['eos_sym'] = 0	# end of system action

    # ----- ACTIV ---- 
    state['activation'] = 'lambda x: T.tanh(x)'
    
    # ----- SIZES ----
    state['prefix'] = 'model_'
    state['word_dim'] = 700
    state['emb_dim'] = 100
    state['rnnh_dim'] = 256
    state['h_dim'] = 256

    state['margin'] = 50
    state['noise_cnt'] = 50
    state['acttype_cnt'] = 28
    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 5
    state['cost_threshold'] = 1.003

     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  
    # Maximum sequence length / trim batches
    state['seqlen'] = 280
    # Batch size
    state['bs'] = 2000
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
    state['valid_freq'] = 1
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1


    return state
