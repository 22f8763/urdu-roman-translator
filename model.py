# model.py - FIXED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        
        # Use bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Linear layers to project bidirectional states to decoder dimensions
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: [batch_size, src_len]
        batch_size = src.shape[0]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        
        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack outputs
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # Handle bidirectional hidden states
        # hidden/cell: [num_layers * 2, batch_size, hidden_size]
        # Reshape to separate forward and backward
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        # Concatenate forward and backward for each layer
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)  # [num_layers, batch_size, hidden_size*2]
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)  # [num_layers, batch_size, hidden_size*2]
        
        # Project back to original hidden size
        hidden = self.hidden_projection(hidden)  # [num_layers, batch_size, hidden_size]
        cell = self.cell_projection(cell)  # [num_layers, batch_size, hidden_size]
        
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_size, num_layers=4, dropout=0.3):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_input, hidden, cell):
        # tgt_input: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(tgt_input))  # [batch_size, seq_len, emb_dim]
        
        # LSTM forward
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [batch_size, seq_len, hidden_size]
        
        # Project to vocabulary
        prediction = self.fc_out(output)  # [batch_size, seq_len, output_size]
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Check that hidden sizes match
        assert encoder.hidden_size == decoder.hidden_size, \
            f"Hidden size mismatch: encoder={encoder.hidden_size}, decoder={decoder.hidden_size}"

    def _prepare_decoder_hidden(self, enc_hidden, enc_cell):
        """Handle layer count mismatch between encoder and decoder"""
        enc_layers = self.encoder.num_layers
        dec_layers = self.decoder.num_layers
        
        if enc_layers == dec_layers:
            return enc_hidden, enc_cell
        elif enc_layers < dec_layers:
            # Need to expand encoder states to match decoder layers
            # Strategy: repeat the last encoder layer for additional decoder layers
            batch_size = enc_hidden.shape[1]
            hidden_size = enc_hidden.shape[2]
            
            # Calculate how many times to repeat and remainder
            repeat_times = dec_layers // enc_layers
            remainder = dec_layers % enc_layers
            
            # Repeat encoder layers
            if repeat_times > 1:
                hidden_expanded = enc_hidden.repeat(repeat_times, 1, 1)
                cell_expanded = enc_cell.repeat(repeat_times, 1, 1)
            else:
                hidden_expanded = enc_hidden
                cell_expanded = enc_cell
            
            # Handle remainder
            if remainder > 0:
                # Take the last `remainder` layers from encoder and append
                hidden_expanded = torch.cat([hidden_expanded, enc_hidden[-remainder:]], dim=0)
                cell_expanded = torch.cat([cell_expanded, enc_cell[-remainder:]], dim=0)
            
            return hidden_expanded, cell_expanded
        else:
            # More encoder layers than decoder layers - take the last dec_layers
            return enc_hidden[-dec_layers:], enc_cell[-dec_layers:]

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len] — includes SOS and EOS
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Encode source
        _, enc_hidden, enc_cell = self.encoder(src, src_lengths)
        
        # Prepare decoder hidden states (handle layer mismatch)
        hidden, cell = self._prepare_decoder_hidden(enc_hidden, enc_cell)

        # First input to decoder is SOS token
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]

        for t in range(1, tgt_len):  # Start from index 1 (after SOS)
            # Decoder forward pass
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            # output: [batch_size, 1, vocab_size]
            
            # Store the output
            outputs[:, t, :] = output.squeeze(1)  # Remove seq dimension

            # Decide next input: teacher forcing or prediction
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = tgt[:, t].unsqueeze(1)  # [batch_size, 1]
            else:
                top1 = output.argmax(2)  # [batch_size, 1]
                decoder_input = top1

        return outputs  # [batch_size, tgt_len, vocab_size]


# ===========================
# HELPER: Create Model Instance
# ===========================
def create_model(vocab, device, embedding_dim=256, hidden_size=512, enc_layers=2, dec_layers=4, dropout=0.3):
    src_vocab_size = len(vocab['src_char2idx'])
    tgt_vocab_size = len(vocab['tgt_char2idx'])

    encoder = Encoder(
        input_size=src_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=enc_layers,
        dropout=dropout
    )

    decoder = Decoder(
        output_size=tgt_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=dec_layers,
        dropout=dropout
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    return model


# Test function to verify shapes
def test_model_shapes():
    """Test function to verify all tensor shapes work correctly"""
    import pickle
    
    try:
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(vocab, device)
        
        # Test with dummy data
        batch_size = 4
        src_len = 10
        tgt_len = 8
        
        # Create dummy tensors
        src = torch.randint(1, len(vocab['src_char2idx']), (batch_size, src_len)).to(device)
        tgt = torch.randint(1, len(vocab['tgt_char2idx']), (batch_size, tgt_len)).to(device)
        src_lengths = torch.randint(5, src_len+1, (batch_size,)).to(device)
        
        print("Testing model with:")
        print(f"  src shape: {src.shape}")
        print(f"  tgt shape: {tgt.shape}")
        print(f"  src_lengths: {src_lengths}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(src, tgt, src_lengths, teacher_forcing_ratio=0.5)
            print(f"  output shape: {outputs.shape}")
            print("✅ Model shapes test passed!")
            
        return True
        
    except Exception as e:
        print(f"❌ Shape test failed: {e}")
        return False


# Optional: Print model structure
if __name__ == "__main__":
    test_model_shapes()