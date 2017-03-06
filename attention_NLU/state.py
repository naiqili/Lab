from collections import OrderedDict

def title_state():
    state = {}

    state['train_file'] = 'data/train_data.pkl'
    state['valid_file'] = 'data/dev_data.pkl'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['eos_sym'] = 0	# end of system action
    state['fill_sym'] = -1

    state['seq_len_in'] = 100
    state['title_seq_len_out'] = 15
    state['who_seq_len_out'] = 15
    state['loc_seq_len_out'] = 15

    # ----- ACTIV ---- 
    state['active'] = 'lambda x: T.tanh(x)'
    
    # ----- SIZES ----
    state['prefix'] = 'title_'
    state['word_dim'] = 2390
    state['emb_dim'] = 256
    state['h_dim'] = 256

    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 80
    state['cost_threshold'] = 1.003

     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'rmsprop'  
    # Batch size
    state['bs'] = 20
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
    state['valid_freq'] = 1000
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1

    return state



def where_state():
    state = {}

    state['train_file'] = 'data/train_data.pkl'
    state['valid_file'] = 'data/dev_data.pkl'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['eos_sym'] = 0	# end of system action
    state['fill_sym'] = -1

    state['seq_len_in'] = 100
    state['title_seq_len_out'] = 15
    state['who_seq_len_out'] = 15
    state['loc_seq_len_out'] = 15

    # ----- ACTIV ---- 
    state['active'] = 'lambda x: T.tanh(x)'
    
    # ----- SIZES ----
    state['prefix'] = 'title_'
    state['word_dim'] = 2390
    state['emb_dim'] = 256
    state['h_dim'] = 256

    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 20
    state['cost_threshold'] = 1.003

     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  
    # Batch size
    state['bs'] = 1000
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
    
def who_state():
    state = {}

    state['train_file'] = 'data/train_data.pkl'
    state['valid_file'] = 'data/dev_data.pkl'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['eos_sym'] = 0	# end of system action
    state['fill_sym'] = -1

    state['seq_len_in'] = 100
    state['title_seq_len_out'] = 15
    state['who_seq_len_out'] = 15
    state['loc_seq_len_out'] = 15

    # ----- ACTIV ---- 
    state['active'] = 'lambda x: T.tanh(x)'
    
    # ----- SIZES ----
    state['prefix'] = 'title_'
    state['word_dim'] = 2390
    state['emb_dim'] = 256
    state['h_dim'] = 256

    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 20
    state['cost_threshold'] = 1.003

     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  
    # Batch size
    state['bs'] = 1000
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
    
    
def whenst_hour_state():
    state = {}

    state['train_file'] = 'data/train_data.pkl'
    state['valid_file'] = 'data/dev_data.pkl'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['eos_sym'] = 0	# end of system action
    state['fill_sym'] = -1

    state['seq_len_in'] = 100
    state['title_seq_len_out'] = 15
    state['who_seq_len_out'] = 15
    state['loc_seq_len_out'] = 15
    state['out_dim'] = 25

    # ----- ACTIV ---- 
    state['active'] = 'lambda x: T.tanh(x)'
    
    # ----- SIZES ----
    state['prefix'] = 'title_'
    state['word_dim'] = 2390
    state['emb_dim'] = 256
    state['h_dim'] = 256
    state['out_dim'] = 25
    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 20
    state['cost_threshold'] = 1.003

     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  
    # Batch size
    state['bs'] = 20
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
    state['valid_freq'] = 3000
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1


    return state
    
def whened_hour_state():
    state = {}

    state['train_file'] = 'data/train_data.pkl'
    state['valid_file'] = 'data/dev_data.pkl'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['eos_sym'] = 0	# end of system action
    state['fill_sym'] = -1

    state['seq_len_in'] = 100
    state['title_seq_len_out'] = 15
    state['who_seq_len_out'] = 15
    state['loc_seq_len_out'] = 15
    state['out_dim'] = 25

    # ----- ACTIV ---- 
    state['active'] = 'lambda x: T.tanh(x)'
    
    # ----- SIZES ----
    state['prefix'] = 'title_'
    state['word_dim'] = 2390
    state['emb_dim'] = 256
    state['h_dim'] = 256
    state['out_dim'] = 25
    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 20
    state['cost_threshold'] = 1.003

     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  
    # Batch size
    state['bs'] = 20
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
    state['valid_freq'] = 3000
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1


    return state
