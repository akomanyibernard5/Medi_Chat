import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, max_len):
        super(NeuralNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Attention
        self.attn = nn.Linear(embed_size, 1)

        # Final classification layers
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)  # (batch_size, seq_len, embed_size)

        # Attention scores
        attn_weights = torch.softmax(self.attn(embeds), dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(attn_weights * embeds, dim=1)       # (batch_size, embed_size)

        x = F.relu(self.fc1(context))                           # (batch_size, hidden_size)
        x = self.fc2(x)                                         # (batch_size, output_size)
        return x  # I applied sigmoid during inference for multi-label
