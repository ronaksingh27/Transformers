import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self,d_model: int,vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.emebedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.emebedding(x)*math.sqrt(self.d_model)
    

class PositionalEmbedding(nn.Module):
    def __init( self,d_model:int, seq_len: int, dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Create a 2D tensor matrix
        pe = torch.zeros(seq_len,d_model)

        #Crearte a vector of shape seq_len
        position = torch.arrange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
  