# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pickle

class UrduRomanDataset(Dataset):
    def __init__(self, tensor_pairs):
        self.pairs = tensor_pairs  # List of (src_tensor, tgt_tensor)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # Returns (src_tensor, tgt_tensor)

def collate_fn(batch):
    """
    Custom collate function to pad sequences in batch.
    Returns:
        src_padded: [batch_size, src_len]
        tgt_padded: [batch_size, tgt_len]
        src_lengths: [batch_size] — for pack_padded_sequence
        tgt_lengths: [batch_size]
    """
    src_list = [item[0] for item in batch]  # List of source tensors
    tgt_list = [item[1] for item in batch]  # List of target tensors (includes SOS/EOS)

    # Get lengths before padding
    src_lengths = torch.tensor([len(src) for src in src_list], dtype=torch.long)
    tgt_lengths = torch.tensor([len(tgt) for tgt in tgt_list], dtype=torch.long)

    # Pad sequences
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)  # PAD=0
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0)

    return src_padded, tgt_padded, src_lengths, tgt_lengths

# ===========================
# LOAD TENSORS & VOCAB
# ===========================
with open("tensor_pairs.pkl", "rb") as f:
    tensor_pairs = pickle.load(f)

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

print(f"Total tensor pairs: {len(tensor_pairs)}")

# ===========================
# SPLIT DATA
# ===========================
total = len(tensor_pairs)
train_size = int(0.5 * total)
val_size = int(0.25 * total)
test_size = total - train_size - val_size

train_pairs, val_pairs, test_pairs = random_split(
    tensor_pairs,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

print(f"SplitOptions: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

# ===========================
# CREATE DATASETS & DATALOADERS
# ===========================
train_dataset = UrduRomanDataset(train_pairs)
val_dataset = UrduRomanDataset(val_pairs)
test_dataset = UrduRomanDataset(test_pairs)

BATCH_SIZE = 64  # You can experiment with 32, 128 later

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"✅ Created DataLoaders with batch_size={BATCH_SIZE}")

# ===========================
# SAVE SPLIT LOADERS & VOCAB (Optional)
# ===========================
# We'll keep vocab and loaders in memory for training script, but you can save if needed:
torch.save({
    'train_loader': train_loader,
    'val_loader': val_loader,
    'test_loader': test_loader,
    'vocab': vocab
}, 'data_loaders_vocab.pth')

print("✅ Saved data_loaders_vocab.pth (optional)")

# ===========================
# SANITY CHECK: Inspect one batch
# ===========================
print("\n🔍 Sanity check: One training batch")
for batch in train_loader:
    src, tgt, src_len, tgt_len = batch
    print(f"Source batch shape: {src.shape}")
    print(f"Target batch shape: {tgt.shape}")
    print(f"Source lengths: {src_len[:5]}")  # first 5
    print(f"Target lengths: {tgt_len[:5]}")
    break

print("\n🎉 Phase 3 Complete! Ready for Model Architecture.")        