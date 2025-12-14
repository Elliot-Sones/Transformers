import torch
import os
from pathlib import Path

def optimize_checkpoint(path, key_mapping):
    print(f"Processing {path}...")
    try:
        # Load
        file_path = Path(path)
        if not file_path.exists():
            print(f"❌ File not found: {path}")
            return
            
        original_size = file_path.stat().st_size / (1024*1024)
        print(f"  Original size: {original_size:.2f} MB")
        
        # Load on CPU
        ckpt = torch.load(file_path, map_location='cpu')
        
        # Create new dict
        new_ckpt = {}
        
        # Copy only essential keys
        for keep_key in key_mapping['keep']:
            if keep_key in ckpt:
                new_ckpt[keep_key] = ckpt[keep_key]
        
        # Handle special cases / remapping if needed
        if 'rename' in key_mapping:
            for old_k, new_k in key_mapping['rename'].items():
                if old_k in ckpt:
                    new_ckpt[new_k] = ckpt[old_k]
                    
        # Save to new path
        new_path = file_path.parent / "model_optimized.pt"
        torch.save(new_ckpt, new_path)
        
        new_size = new_path.stat().st_size / (1024*1024)
        print(f"  ✅ Saved to {new_path}")
        print(f"  New size: {new_size:.2f} MB")
        print(f"  Reduction: {100 * (1 - new_size/original_size):.1f}%")
        
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")

# Defnitions of what to keep for each model type
configs = [
    {
        'path': 'encoder_transformer/sentiment/checkpoints/best.pt',
        'key': {
            'keep': ['model_state_dict', 'config', 'best_acc', 'tokenizer_path']
        }
    },
    {
        'path': 'decoder_transformer/checkpoints/best.pt',
        'key': {
            'keep': ['model_state_dict', 'meta', 'ema_state_dict'] 
        }
    },
    {
        'path': 'machine_translation/checkpoints/best.pt',
        'key': {
            'keep': ['config', 'model'] # 'optimizer' is dropped
        }
    }
]

print("Starting optimization...")
for config in configs:
    optimize_checkpoint(config['path'], config['key'])
print("Done!")
