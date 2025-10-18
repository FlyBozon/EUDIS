#!/usr/bin/env python3
"""
Generate trees and rocks for Webots drone world.
Creates VRML nodes for Pine trees and Rock objects distributed along the flight path.
"""

import random
import math

# Configuration
START_X = 0
END_X = 1000
LATERAL_RANGE = 40  # Y coordinate range: -20 to +20

# Tree settings
NUM_TREES = 200
TREE_MIN_DISTANCE = 8  # Minimum distance between trees

# Rock settings
NUM_ROCKS = 150
ROCK_MIN_SCALE = 8
ROCK_MAX_SCALE = 30
ROCK_MIN_DISTANCE = 5  # Minimum distance between rocks


def distance_2d(p1, p2):
    """Calculate 2D distance between two points (x, y)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_positions(num_objects, min_distance, existing_positions=None):
    """
    Generate positions for objects ensuring minimum distance.
    Returns list of (x, y) tuples.
    """
    if existing_positions is None:
        existing_positions = []

    positions = existing_positions.copy()
    attempts = 0
    max_attempts = num_objects * 100

    while len(positions) < len(existing_positions) + num_objects and attempts < max_attempts:
        attempts += 1

        # Generate random position
        x = random.uniform(START_X, END_X)
        y = random.uniform(-LATERAL_RANGE, LATERAL_RANGE)
        new_pos = (x, y)

        # Check minimum distance to all existing positions
        too_close = False
        for existing_pos in positions:
            if distance_2d(new_pos, existing_pos) < min_distance:
                too_close = True
                break

        if not too_close:
            positions.append(new_pos)

    if len(positions) < len(existing_positions) + num_objects:
        print(f"Warning: Could only place {len(positions) - len(existing_positions)} objects (requested {num_objects})")

    return positions[len(existing_positions):]


def generate_trees(tree_positions):
    """Generate VRML code for Pine trees."""
    tree_nodes = []

    for i, (x, y) in enumerate(tree_positions, 1):
        rotation = random.uniform(0, 2 * math.pi)

        tree_node = f"""Pine {{
  translation {x:.1f} {y:.1f} 0
  rotation 0 0 1 {rotation:.1f}
  name "pine_{i}"
}}"""
        tree_nodes.append(tree_node)

    return tree_nodes


def generate_rocks(rock_positions):
    """Generate VRML code for Rock objects."""
    rock_nodes = []

    for i, (x, y) in enumerate(rock_positions, 1):
        rotation = random.uniform(0, 2 * math.pi)
        scale = random.randint(ROCK_MIN_SCALE, ROCK_MAX_SCALE)

        rock_node = f"""Rock {{
  translation {x:.1f} {y:.1f} 0
  rotation 0 0 1 {rotation:.1f}
  scale {scale}
  name "rock_{i}"
}}"""
        rock_nodes.append(rock_node)

    return rock_nodes


def main():
    """Generate and output all environment objects."""
    random.seed(42)  # For reproducibility

    print("Generating environment objects...")
    print(f"Flight path: X={START_X} to X={END_X} (±{LATERAL_RANGE}m lateral)")
    print()

    # Generate tree positions
    print(f"Generating {NUM_TREES} trees...")
    tree_positions = generate_positions(NUM_TREES, TREE_MIN_DISTANCE)
    print(f"Placed {len(tree_positions)} trees")

    # Generate rock positions (avoiding trees)
    print(f"Generating {NUM_ROCKS} rocks...")
    rock_positions = generate_positions(NUM_ROCKS, ROCK_MIN_DISTANCE, tree_positions)
    print(f"Placed {len(rock_positions)} rocks")
    print()

    # Generate VRML nodes
    tree_nodes = generate_trees(tree_positions)
    rock_nodes = generate_rocks(rock_positions)

    # Output to file
    output_file = "environment_nodes.txt"
    with open(output_file, 'w') as f:
        f.write("# Generated Pine Trees\n")
        f.write("# Copy these nodes into your Webots world file\n\n")
        for node in tree_nodes:
            f.write(node + "\n")

        f.write("\n# Generated Rocks\n\n")
        for node in rock_nodes:
            f.write(node + "\n")

    print(f"[OK] Generated {len(tree_nodes)} trees and {len(rock_nodes)} rocks")
    print(f"[OK] Output written to: {output_file}")
    print()
    print("To use:")
    print("1. Open the generated file")
    print("2. Copy the tree and rock nodes")
    print("3. Paste them into your drone.wbt file (replace existing trees/rocks)")


if __name__ == "__main__":
    main()
