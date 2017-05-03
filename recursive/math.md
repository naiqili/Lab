# LSTM

## Forward

$$i=\sigma(W^i_{mid} x_{mid} + W^i_{left} h_{left} + W^i_{right} h_{right} + b^i)$$

$$f_{left}=\sigma(W^f_{mid} x_{mid} + W^{f}_{left,left} h_{left} + W^{f}_{left,right} h_{right} + b^f)$$

$$f_{right}=\sigma(W^f_{mid} x_{mid} + W^{f}_{right, left} h_{left} + W^{f}_{right, right} h_{right} + b^f)$$

$$o=\sigma(W^o_{mid} x_{mid} + W^o_{left} h_{left} + W^o_{right} h_{right} + b^o)$$

$$u=\tanh(W^u_{mid} x_{mid} + W^u_{left} h_{left} + W^u_{right} h_{right} + b^u)$$

$$c = i \odot u + f_{left} \odot c_{left} + f_{right} \odot c_{right}$$

$$h = o \odot \tanh(c)$$

## Backward

$$i_{lr} = \sigma(W^i_{lr} emb_{lr} + W^i_p h_{parent} + b^i)$$

$$f_{lr} = \sigma(W^f_{lr} emb_{lr} + W^f_p h_{parent} + b^f)$$

$$o_{lr} = \sigma(W^o_{lr} emb_{lr} + W^o_p h_{parent} + b^o)$$

$$u_{lr} = \tanh(W^u_{lr} emb_{lr} + W^u_p h_{parent} + b^u)$$

$$c_{lr} = i_{lr} \odot u_{lr} + f_{lr} \odot c_{parent}$$

$$h_{lr} = o_{lr} \odot \tanh c_{lr}$$

$$lr \in \{left, right\}$$