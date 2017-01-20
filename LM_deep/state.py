from collections import OrderedDict

def prototype_state():
    state = {}

    state['train_file'] = 'tmp/train_data.txt'
    state['valid_file'] = 'tmp/dev_data.txt'

    state['triple_step_type'] = 'gated'
    state['output_dim'] = 5
    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    state['oov'] = '<oov>' # Not used
    
    # These are end-of-sequence marks
    state['end_sym_system'] = '</s>'
    state['end_sym_turn'] = '</t>'
    
    state['eos_sym'] = 0	# end of system action

    # ----- ACTIV ---- 
    state['activation'] = 'lambda x: T.tanh(x)'
    
    state['decoder_bias_type'] = 'all' # first, or selective 

    state['sent_step_type'] = 'gated'
    # state['triple_step_type'] = 'gated' 

    # ----- SIZES ----
    state['prefix'] = 'model_'
    state['worddim'] = 700
    state['embdim'] = 80
    state['hdim'] = 100

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 10
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
    state['valid_freq'] = 100
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1

    return state

def layer2_gru_state():
    state = {}

    state['train_file'] = 'tmp/train_data.txt'
    state['valid_file'] = 'tmp/dev_data.txt'

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    state['oov'] = '<oov>' # Not used
    
    # These are end-of-sequence marks
    state['eos_sym'] = 0	# end of system action

    # ----- ACTIV ---- 
    state['activation'] = 'lambda x: T.tanh(x)'
    
    # state['triple_step_type'] = 'gated' 

    # ----- SIZES ----
    state['prefix'] = 'model_'
    state['word_dim'] = 700
    state['emb_dim'] = 50
    state['h1_dim'] = 128
    state['h2_dim'] = 128

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 10
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


def lstm_state():
    state = {}

    state['train_file'] = 'tmp/train_data.txt'
    state['valid_file'] = 'tmp/dev_data.txt'

    state['triple_step_type'] = 'gated'
    state['output_dim'] = 5
    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    state['oov'] = '<oov>' # Not used
    
    # These are end-of-sequence marks
    state['end_sym_system'] = '</s>'
    state['end_sym_turn'] = '</t>'
    
    state['eos_sym'] = 0	# end of system action

    # ----- ACTIV ---- 
    state['activation'] = 'lambda x: T.tanh(x)'
    
    state['decoder_bias_type'] = 'all' # first, or selective 

    state['sent_step_type'] = 'gated'
    # state['triple_step_type'] = 'gated' 

    # ----- SIZES ----
    state['prefix'] = 'model_'
    state['worddim'] = 700
    state['embdim'] = 20
    state['hdim'] = 20

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 10
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
    state['valid_freq'] = 50
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1

    return state
