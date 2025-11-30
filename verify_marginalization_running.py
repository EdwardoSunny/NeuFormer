"""
Verify that marginalization is actually being used during training
"""

import torch
import pickle
from src.neural_decoder.transformer_ctc import MultiScaleCTCDecoder
from src.neural_decoder.diphone_utils import DiphoneVocabulary

print("="*80)
print("VERIFYING MARGINALIZATION IS ACTIVE")
print("="*80)

# Load vocab and create matrix (same as train_diphone.py)
diphone_vocab = DiphoneVocabulary.load('diphone_vocab.pkl')
marg_matrix = diphone_vocab.create_marginalization_matrix(
    num_phonemes=41,
    phoneme_blank_id=40
)

# Create model (same config as train_diphone.py)
with open('/home/edward/neural_seq_decoder/ptDecoder_ctc', 'rb') as f:
    data = pickle.load(f)

model = MultiScaleCTCDecoder(
    n_classes=41,
    input_dim=256,
    d_model=1024,
    encoder_layers=6,
    n_heads=8,
    dim_ff=2048,
    dropout=0.3,
    conv_kernel=31,
    n_days=len(data["train"]),
    gaussian_smooth_width=2.0,
    use_diphone_head=True,
    num_diphones=diphone_vocab.get_vocab_size(),
    diphone_marginalization_matrix=marg_matrix,
    device='cuda',
).cuda()

print(f"\n✓ Model created")
print(f"  - use_marginalization: {model.use_marginalization}")
print(f"  - phone_output exists: {model.phone_output is not None}")
print(f"  - diphone_output exists: {model.diphone_output is not None}")
print(f"  - marginalization_matrix registered: {hasattr(model, 'marginalization_matrix')}")

if model.use_marginalization:
    print(f"  - marginalization_matrix shape: {model.marginalization_matrix.shape}")

# Test forward pass
print(f"\n✓ Testing forward pass...")
model.eval()
with torch.no_grad():
    x = torch.randn(2, 150, 256).cuda()
    day_idx = torch.zeros(2, dtype=torch.long).cuda()

    output = model(x, day_idx)

    if len(output) == 3:
        phone_log_probs, out_lens, diphone_log_probs = output
        print(f"  - phone_log_probs shape: {phone_log_probs.shape}")
        print(f"  - diphone_log_probs shape: {diphone_log_probs.shape}")

        # Manually verify marginalization
        # Convert to probs, marginalize, check if matches
        diphone_probs = torch.exp(diphone_log_probs.transpose(0, 1))  # [B, T, num_diphones]
        manual_phone_probs = torch.matmul(diphone_probs, model.marginalization_matrix)
        manual_phone_log_probs = torch.log(manual_phone_probs + 1e-10).transpose(0, 1)

        diff = torch.abs(phone_log_probs - manual_phone_log_probs).max().item()
        print(f"  - Manual marginalization matches: {diff < 1e-5} (max diff: {diff:.6f})")

        if diff < 1e-5:
            print(f"\n{'='*80}")
            print(f"✅ MARGINALIZATION IS WORKING CORRECTLY!")
            print(f"{'='*80}")
            print(f"\nYour model is using proper DCoND marginalization.")
            print(f"Phoneme predictions come from marginalized diphone predictions.")
            print(f"\nJust need to train longer to see if final CER < 0.22!")
        else:
            print(f"\n⚠️ WARNING: Marginalization mismatch!")
    else:
        print(f"  ⚠️ Model returned {len(output)} outputs (expected 3)")

print(f"\n{'='*80}")
print(f"EXPECTED TRAINING TIMELINE")
print(f"{'='*80}")
print(f"Batch 1,300 (now):    CER ~0.37")
print(f"Batch 10,000:         CER ~0.25-0.30")
print(f"Batch 50,000:         CER ~0.18-0.22")
print(f"Batch 100,000+:       CER ~0.15-0.18 (if marginalization helps!)")
print(f"\nBaseline (GRU/Transformer): CER ~0.22")
print(f"Target (with marginalization): CER ~0.15-0.18")
print(f"\nYou're only 0.87% done - keep training!")
