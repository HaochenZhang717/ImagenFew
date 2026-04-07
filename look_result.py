import torch
from pprint import pprint

ckpt = torch.load('ImagenTime', map_location='cpu')

print(f"Type: {type(ckpt)}")

if isinstance(ckpt, dict):
    print(f"\nTop-level keys: {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"\n  '{k}' (dict, {len(v)} entries):")
            for kk, vv in list(v.items())[:5]:
                print(f"    {kk}: {type(vv).__name__} {getattr(vv, 'shape', '')}")
            if len(v) > 5:
                print(f"    ... ({len(v) - 5} more)")
        elif hasattr(v, 'shape'):
            print(f"  '{k}': {type(v).__name__} shape={v.shape}")
        else:
            print(f"  '{k}': {type(v).__name__} = {v}")
else:
    pprint(ckpt)