#!/usr/bin/env python3
"""
Update drone.wbt with new trees and rocks for 1km path.
"""

# Read the world file
with open('../worlds/drone.wbt', 'r') as f:
    lines = f.readlines()

# Find where trees start (first Pine node) and where Mavic2Pro starts
tree_start_idx = None
mavic_start_idx = None

for i, line in enumerate(lines):
    if 'Pine {' in line and tree_start_idx is None:
        tree_start_idx = i
    if 'Mavic2Pro {' in line:
        mavic_start_idx = i
        break

print(f"Found trees starting at line {tree_start_idx + 1}")
print(f"Found Mavic2Pro at line {mavic_start_idx + 1}")

# Read the new environment nodes
with open('environment_clean.txt', 'r') as f:
    new_environment = f.read().strip()

# Build the new world file
new_lines = []

# Keep everything before the trees
new_lines.extend(lines[:tree_start_idx])

# Add the new environment
new_lines.append(new_environment + '\n')

# Add the Mavic2Pro and everything after
new_lines.extend(lines[mavic_start_idx:])

# Write the updated world file
with open('../worlds/drone.wbt', 'w') as f:
    f.writelines(new_lines)

print("[OK] Updated drone.wbt with 200 trees and 150 rocks for 1km path!")
