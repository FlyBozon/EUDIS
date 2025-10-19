import pickle
import torch

print("="*60)
print("Inspecting model configuration and weights")
print("="*60)

with open('visual_model_config.pkl', 'rb') as f:
    config = pickle.load(f)

print("\n[CONFIG]")
for key, value in config.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {key}: {value}")

print("\n[MODEL WEIGHTS]")
model_state = torch.load('best_drone_visual_model.pth', map_location='cpu')

if isinstance(model_state, dict):
    print(f"  Type: Dictionary with {len(model_state)} keys")
    for key in list(model_state.keys())[:10]:
        if isinstance(model_state[key], torch.Tensor):
            print(f"  {key}: shape={model_state[key].shape}, dtype={model_state[key].dtype}")
        else:
            print(f"  {key}: {type(model_state[key])}")
else:
    print(f"  Type: {type(model_state)}")
    if hasattr(model_state, 'state_dict'):
        print("  Has state_dict method")
    print(f"  Attributes: {dir(model_state)[:5]}")

print("\n" + "="*60)