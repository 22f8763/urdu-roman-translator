# train.py - SIMPLIFIED VERSION
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
import math
from model import create_model  # This now includes the fixed Seq2Seq
from dataset import UrduRomanDataset, collate_fn
from torch.utils.data import DataLoader, random_split

# ===========================
# CONFIGURATION
# ===========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
ENC_LAYERS = 2
DEC_LAYERS = 4
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
TEACHER_FORCING_RATIO = 0.5

# Paths
DATA_PATH = "tensor_pairs.pkl"
VOCAB_PATH = "vocab.pkl"
MODEL_SAVE_PATH = "best_model.pth"

# ===========================
# LOAD DATA & VOCAB
# ===========================
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

with open(DATA_PATH, "rb") as f:
    tensor_pairs = pickle.load(f)

# Split data
total = len(tensor_pairs)
train_size = int(0.5 * total)
val_size = int(0.25 * total)
test_size = total - train_size - val_size

train_pairs, val_pairs, _ = random_split(
    tensor_pairs,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = UrduRomanDataset(train_pairs)
val_dataset = UrduRomanDataset(val_pairs)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ===========================
# BUILD MODEL
# ===========================
model = create_model(
    vocab=vocab,
    device=DEVICE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    enc_layers=ENC_LAYERS,
    dec_layers=DEC_LAYERS,
    dropout=DROPOUT
)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
PAD_IDX = vocab['special_tokens']['PAD']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test model shapes before training
print("\n🧪 Testing model shapes...")
try:
    dummy_src = torch.randint(1, len(vocab['src_char2idx']), (2, 5)).to(DEVICE)
    dummy_tgt = torch.randint(1, len(vocab['tgt_char2idx']), (2, 6)).to(DEVICE)
    dummy_src_len = torch.tensor([5, 4]).to(DEVICE)
    
    with torch.no_grad():
        test_output = model(dummy_src, dummy_tgt, dummy_src_len, 0.5)
        print(f"✅ Shape test passed! Output shape: {test_output.shape}")
except Exception as e:
    print(f"❌ Shape test failed: {e}")
    exit(1)

# ===========================
# TRAINING FUNCTIONS
# ===========================
def train_epoch(model, data_loader, optimizer, criterion, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(data_loader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        src_len = src_len.to(DEVICE)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            output = model(src, tgt, src_len, teacher_forcing_ratio)
            # output: [batch_size, tgt_len, vocab_size]
            
            # Reshape for loss calculation
            # Skip SOS token (position 0) in both output and target
            output_for_loss = output[:, 1:].contiguous().view(-1, output.shape[-1])
            tgt_for_loss = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output_for_loss, tgt_for_loss)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}:")
            print(f"  src shape: {src.shape}, tgt shape: {tgt.shape}")
            print(f"  src_len: {src_len}")
            print(f"  Error: {e}")
            raise e
    
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt, src_len, tgt_len in data_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_len = src_len.to(DEVICE)
            
            # No teacher forcing during evaluation
            output = model(src, tgt, src_len, teacher_forcing_ratio=0.0)
            
            # Reshape for loss
            output_for_loss = output[:, 1:].contiguous().view(-1, output.shape[-1])
            tgt_for_loss = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output_for_loss, tgt_for_loss)
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

# ===========================
# TRAINING LOOP
# ===========================
best_valid_loss = float('inf')
print("\n🚀 Starting Training...\n")

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    try:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, TEACHER_FORCING_RATIO)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) % 60)
        
        # Prevent perplexity overflow
        train_ppl = math.exp(min(train_loss, 10))
        valid_ppl = math.exp(min(valid_loss, 10))
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'vocab': vocab,
                'config': {
                    'embedding_dim': EMBEDDING_DIM,
                    'hidden_size': HIDDEN_SIZE,
                    'enc_layers': ENC_LAYERS,
                    'dec_layers': DEC_LAYERS,
                    'dropout': DROPOUT
                }
            }, MODEL_SAVE_PATH)
            print("✅ Saved best model")
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        print(f'\t Val Loss: {valid_loss:.3f} |  Val PPL: {valid_ppl:7.3f}')
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error in epoch {epoch}: {e}")
        break

print(f"\n🎉 Training complete! Best validation loss: {best_valid_loss:.3f}")