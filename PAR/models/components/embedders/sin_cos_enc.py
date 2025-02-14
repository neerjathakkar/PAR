import torch
import math

class SinCosPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_position=100):
        super(SinCosPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_position = max_position

        # Create a matrix of size (max_position, d_model) with positional encodings
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Calculate sin on even indices and cos on odd indices
        pe = torch.zeros(max_position, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register the positional encodings as a buffer so that it is not updated during training
        self.register_buffer('pe', pe)

    def forward(self, indices):
        """
        Inputs:
        - indices: A list or tensor of arbitrary indices (e.g., [3, 9, 0, 80])
        Outputs:
        - Positional encodings for those specific indices
        """
        if torch.is_tensor(indices):
            indices.to(self.pe.device)
        else:
            indices = torch.tensor(indices, device=self.pe.device)
        return self.pe[indices, :]

if __name__ == "__main__":
    # Create a positional encoding module
    pe = SinCosPositionalEncoding(d_model=8, max_position=100)

    indices = [3, 9, 0, 80]
    encodings = pe(indices)
    print(encodings)
    
    indices = torch.tensor([0, 1, 2, 3])
    encodings = pe(indices)
    print(encodings)

    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    encodings = pe(indices)
    print(encodings)

    indices = torch.tensor([[0,1], [0,1]])
    print(indices)
    encodings = pe(indices)
    print(encodings)