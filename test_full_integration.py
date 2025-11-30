"""
Quick integration test to verify full DCoND Step 3 implementation
Tests both training and evaluation modes
"""

import torch
import pickle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.neural_decoder.transformer_ctc import MultiScaleCTCDecoder
from src.neural_decoder.diphone_utils import DiphoneVocabulary
from src.neural_decoder.dataset import SpeechDataset

print("="*80)
print("FULL INTEGRATION TEST: DCoND Steps 1+2+3")
print("="*80)

# Configuration matching train_diphone.py exactly
config = {
    'n_classes': 41,
    'input_dim': 256,
    'd_model': 1024,
    'encoder_layers': 6,
    'n_heads': 8,
    'dim_ff': 2048,
    'dropout': 0.3,
    'conv_kernel': 31,
    'gaussian_smooth_width': 2.0,
    'batch_size': 4,
    'learning_rate': 0.0005,
    'label_smoothing': 0.1,
    # Step 1+2
    'use_diphone_head': True,
    'use_diphone_marginalization': True,
    'diphone_alpha_schedule': 'constant',
    # Step 3
    'use_multiscale_ctc': True,
    'multiscale_lambda_fast': 0.3,
    'multiscale_lambda_slow': 0.3,
}

# Load data and vocab
print("\n1. Loading data and vocabulary...")
with open('/home/edward/neural_seq_decoder/ptDecoder_ctc', 'rb') as f:
    data = pickle.load(f)

diphone_vocab = DiphoneVocabulary.load('diphone_vocab.pkl')
print(f"   ✓ Diphone vocab: {diphone_vocab.num_diphones} diphones")

# Create marginalization matrix
print("\n2. Creating marginalization matrix...")
marg_matrix = diphone_vocab.create_marginalization_matrix(
    num_phonemes=41,
    phoneme_blank_id=40
)
print(f"   ✓ Matrix shape: {marg_matrix.shape}")

# Create dataset
print("\n3. Creating dataset...")
dataset = SpeechDataset(data['train'][:1], diphone_vocab=diphone_vocab)

def _padding(batch):
    X, y, X_lens, y_lens, days, y_diphone, y_diphone_lens = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    y_diphone_padded = pad_sequence(y_diphone, batch_first=True, padding_value=0)
    return (
        X_padded, y_padded, torch.stack(X_lens), torch.stack(y_lens),
        torch.stack(days), y_diphone_padded, torch.stack(y_diphone_lens),
    )

loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=_padding)
print(f"   ✓ Dataset: {len(dataset)} samples")

# Create model
print("\n4. Creating model with ALL features enabled...")
model = MultiScaleCTCDecoder(
    n_classes=config['n_classes'],
    input_dim=config['input_dim'],
    d_model=config['d_model'],
    encoder_layers=config['encoder_layers'],
    n_heads=config['n_heads'],
    dim_ff=config['dim_ff'],
    dropout=config['dropout'],
    conv_kernel=config['conv_kernel'],
    n_days=1,
    gaussian_smooth_width=config['gaussian_smooth_width'],
    use_diphone_head=config['use_diphone_head'],
    num_diphones=diphone_vocab.get_vocab_size(),
    diphone_marginalization_matrix=marg_matrix,
    use_multiscale_ctc=config['use_multiscale_ctc'],
    device='cuda',
).cuda()

print(f"   ✓ Model created")
print(f"     - use_marginalization: {model.use_marginalization}")
print(f"     - use_multiscale_ctc: {model.use_multiscale_ctc}")

# Get batch
X, y, X_len, y_len, day_idx, y_diphone, y_diphone_len = next(iter(loader))
X = X.cuda()
y = y.cuda()
X_len = X_len.cuda()
y_len = y_len.cuda()
day_idx = day_idx.cuda()
y_diphone = y_diphone.cuda()
y_diphone_len = y_diphone_len.cuda()

print(f"\n5. Testing TRAINING mode (model.train())...")
model.train()

# Forward pass
model_output = model(X, day_idx, X_len)
print(f"   ✓ Forward pass returns {len(model_output)} outputs")

# Unpack outputs
if len(model_output) == 7:
    (phone_log_probs, out_lens, diphone_log_probs,
     fast_phone_log_probs, fast_out_lens,
     slow_phone_log_probs, slow_out_lens) = model_output
    print(f"   ✓ Correctly unpacked 7 outputs")
else:
    print(f"   ✗ ERROR: Expected 7 outputs, got {len(model_output)}")
    exit(1)

# Setup losses
loss_ctc = torch.nn.CTCLoss(blank=40, reduction="mean", zero_infinity=True)
loss_ctc_diphone = torch.nn.CTCLoss(blank=diphone_vocab.blank_id, reduction="mean", zero_infinity=True)

# Compute main losses
phone_loss = loss_ctc(phone_log_probs, y, out_lens, y_len)
diphone_loss = loss_ctc_diphone(diphone_log_probs, y_diphone, out_lens, y_diphone_len)
alpha = 0.5
main_loss = alpha * phone_loss + (1.0 - alpha) * diphone_loss

# Compute auxiliary losses
fast_loss = loss_ctc(fast_phone_log_probs, y, fast_out_lens, y_len)
slow_loss = loss_ctc(slow_phone_log_probs, y, slow_out_lens, y_len)

# Total loss
lambda_fast = config['multiscale_lambda_fast']
lambda_slow = config['multiscale_lambda_slow']
total_loss = main_loss + lambda_fast * fast_loss + lambda_slow * slow_loss

print(f"\n   Loss breakdown:")
print(f"     phone_loss:   {phone_loss.item():.4f}")
print(f"     diphone_loss: {diphone_loss.item():.4f}")
print(f"     main_loss:    {main_loss.item():.4f}")
print(f"     fast_loss:    {fast_loss.item():.4f}")
print(f"     slow_loss:    {slow_loss.item():.4f}")
print(f"     total_loss:   {total_loss.item():.4f}")
print(f"   ✓ All losses computed successfully")

# Backward pass
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
optimizer.zero_grad()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
optimizer.step()
print(f"   ✓ Backward pass and optimizer step successful")

print(f"\n6. Testing EVALUATION mode (model.eval())...")
model.eval()

with torch.no_grad():
    model_output = model(X, day_idx, X_len)
    print(f"   ✓ Forward pass returns {len(model_output)} outputs")

    # Simulate trainer evaluation code
    if len(model_output) == 7:
        pred, adjustedLens = model_output[0], model_output[1]
        print(f"   ✓ Correctly unpacked outputs for evaluation")
    elif len(model_output) == 3:
        pred, adjustedLens, _ = model_output
        print(f"   ✓ Correctly unpacked 3 outputs (no multi-scale)")
    else:
        pred, adjustedLens = model_output
        print(f"   ✓ Correctly unpacked 2 outputs (baseline)")

    # Compute eval loss
    eval_loss = loss_ctc(pred, y, adjustedLens, y_len)
    print(f"   ✓ Eval loss: {eval_loss.item():.4f}")

    # Test CTC decoding
    from edit_distance import SequenceMatcher
    total_edit_distance = 0
    total_seq_length = 0

    for i in range(pred.shape[1]):
        decoded_seq = torch.argmax(pred[:adjustedLens[i], i, :], dim=-1)
        decoded_seq = torch.unique_consecutive(decoded_seq)
        decoded_seq = decoded_seq.cpu().numpy()
        decoded_seq = [x for x in decoded_seq if x != 40]  # Remove blank

        true_seq = y[i, :y_len[i]].cpu().numpy()

        matcher = SequenceMatcher(a=true_seq.tolist(), b=decoded_seq)
        total_edit_distance += matcher.distance()
        total_seq_length += len(true_seq)

    cer = total_edit_distance / total_seq_length if total_seq_length > 0 else 1.0
    print(f"   ✓ CER: {cer:.4f}")

print(f"\n{'='*80}")
print(f"✅ FULL INTEGRATION TEST PASSED!")
print(f"{'='*80}")
print(f"\nVerified:")
print(f"  ✓ Model initialization with all DCoND steps")
print(f"  ✓ Training mode: 7 outputs returned")
print(f"  ✓ All 4 losses computed (phone, diphone, fast, slow)")
print(f"  ✓ Total loss = main + λ_fast*fast + λ_slow*slow")
print(f"  ✓ Backward pass and optimizer step")
print(f"  ✓ Evaluation mode: correct output unpacking")
print(f"  ✓ Eval loss computation")
print(f"  ✓ CTC decoding")
print(f"\n🚀 Ready to launch full training!")
print(f"{'='*80}")
