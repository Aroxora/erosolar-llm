#!/usr/bin/env python3
"""
Create an animated GIF showing infinite self-improvement gains.
Visualizes the decreasing master scalar and continuous improvement loop.
Author: Bo Shang
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import io
from pathlib import Path

def create_frame(frame, total_frames):
    """Create a single frame of the animation."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    cyan = '#00d4ff'
    green = '#10a37f'
    magenta = '#ff00ff'
    gold = '#ffd700'

    # Title with pulsing effect
    pulse = 0.9 + 0.1 * np.sin(frame * 0.3)
    ax.text(0, 1.35, 'INFINITE GAINS', fontsize=24 * pulse, fontweight='bold',
            ha='center', va='center', color=cyan, fontfamily='monospace')
    ax.text(0, 1.15, 'Self-Improving Chain-of-Thought LLM', fontsize=11,
            ha='center', va='center', color='#888888', fontfamily='monospace')

    # Rotating spiral of improvement
    t = np.linspace(0, 4*np.pi, 200)
    phase = frame * 0.15
    r = 0.3 + 0.4 * t / (4*np.pi)
    x = r * np.cos(t + phase)
    y = r * np.sin(t + phase)

    # Draw spiral with gradient color
    for i in range(len(t)-1):
        progress = i / len(t)
        color = plt.cm.cool(progress)
        ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=2.5, alpha=0.8)

    # Center infinity symbol - rotating
    inf_phase = frame * 0.1
    infinity_t = np.linspace(0, 2*np.pi, 100)
    scale = 0.18
    inf_x = scale * np.sin(infinity_t + inf_phase)
    inf_y = scale * np.sin(infinity_t + inf_phase) * np.cos(infinity_t + inf_phase)
    ax.plot(inf_x, inf_y, color=gold, linewidth=4)
    ax.fill(inf_x, inf_y, color=gold, alpha=0.4)

    # Orbiting version nodes
    num_nodes = 5
    for i in range(num_nodes):
        angle = 2 * np.pi * i / num_nodes + phase * 0.5
        node_r = 0.95
        nx = node_r * np.cos(angle)
        ny = node_r * np.sin(angle)

        # Glow effect
        glow = plt.Circle((nx, ny), 0.12, color=green, alpha=0.3)
        ax.add_patch(glow)

        # Node circle
        circle = plt.Circle((nx, ny), 0.08, color=green, alpha=0.95)
        ax.add_patch(circle)

        # Version label
        version = f'v0.{i+1:02d}'
        ax.text(nx, ny, version, fontsize=7, ha='center', va='center',
                color='white', fontweight='bold')

    # Animated arrows
    for i in range(num_nodes):
        angle1 = 2 * np.pi * i / num_nodes + phase * 0.5
        angle2 = 2 * np.pi * ((i+1) % num_nodes) / num_nodes + phase * 0.5
        arrow_progress = (frame % 20) / 20.0

        # Arrow position along arc
        arrow_angle = angle1 + (angle2 - angle1 + 2*np.pi/num_nodes) * arrow_progress
        arrow_r = 0.95
        ax.annotate('',
                   xy=(arrow_r * np.cos(arrow_angle + 0.1), arrow_r * np.sin(arrow_angle + 0.1)),
                   xytext=(arrow_r * np.cos(arrow_angle - 0.1), arrow_r * np.sin(arrow_angle - 0.1)),
                   arrowprops=dict(arrowstyle='->', color=cyan, lw=2, alpha=0.9))

    # Master scalar - animating decrease
    cycle_progress = (frame % total_frames) / total_frames
    base_scalar = 0.15
    target_scalar = 0.058
    current_scalar = base_scalar - (base_scalar - target_scalar) * cycle_progress

    # Scalar box with glow
    box = patches.FancyBboxPatch((-0.55, -1.35), 1.1, 0.4,
                                  boxstyle="round,pad=0.03",
                                  facecolor='#1a1a2e', edgecolor=magenta, linewidth=2.5)
    ax.add_patch(box)

    ax.text(0, -1.0, 'MASTER SCALAR', fontsize=11, ha='center', va='center',
            color=magenta, fontweight='bold', fontfamily='monospace')

    # Scalar value with color based on progress
    scalar_color = plt.cm.RdYlGn(cycle_progress)
    ax.text(0, -1.2, f'{current_scalar:.4f}', fontsize=20, ha='center', va='center',
            color=scalar_color, fontweight='bold', fontfamily='monospace')

    # Progress bar
    progress_width = 0.9
    progress_fill = progress_width * cycle_progress
    ax.add_patch(patches.Rectangle((-0.45, -0.8), progress_width, 0.1,
                                    facecolor='#333', edgecolor='#555', linewidth=1))
    ax.add_patch(patches.Rectangle((-0.45, -0.8), progress_fill, 0.1,
                                    facecolor=green, edgecolor='none'))

    # Stats with animation
    round_num = int(cycle_progress * 100) + 1
    samples = 1000 + int(cycle_progress * 50000)
    ax.text(-1.35, -0.45, '↑ Diversity', fontsize=10, color=green, fontfamily='monospace', fontweight='bold')
    ax.text(-1.35, -0.62, '↓ Similarity', fontsize=10, color=cyan, fontfamily='monospace', fontweight='bold')
    ax.text(0.75, -0.45, f'Round {round_num}', fontsize=10, color='#aaa', fontfamily='monospace')
    ax.text(0.75, -0.62, f'{samples:,} samples', fontsize=10, color='#aaa', fontfamily='monospace')

    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)
    return img

def create_icon():
    """Create static app icon."""
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    cyan = '#00d4ff'
    green = '#10a37f'
    gold = '#ffd700'

    # Circular glow background
    for r, alpha in [(0.95, 0.15), (0.85, 0.2), (0.75, 0.25)]:
        circle = plt.Circle((0, 0), r, color=green, alpha=alpha, fill=True)
        ax.add_patch(circle)

    # Outer ring
    circle = plt.Circle((0, 0), 0.95, color=green, alpha=0.8, fill=False, linewidth=4)
    ax.add_patch(circle)

    # Infinity symbol
    t = np.linspace(0, 2*np.pi, 100)
    scale = 0.45
    inf_x = scale * np.sin(t)
    inf_y = scale * np.sin(t) * np.cos(t)
    ax.plot(inf_x, inf_y, color=gold, linewidth=8, solid_capstyle='round')

    # Upward arrow
    ax.annotate('', xy=(0, 0.75), xytext=(0, -0.75),
                arrowprops=dict(arrowstyle='->', color=cyan, lw=5,
                               mutation_scale=20))

    icon_path = Path('/Users/bo/Downloads/infinite_gains_icon.png')
    fig.savefig(str(icon_path), dpi=256, bbox_inches='tight', pad_inches=0,
                facecolor='#0d1117', transparent=False)
    plt.close(fig)
    print(f"Icon saved to {icon_path}")

def main():
    total_frames = 60
    frames = []

    print("Generating frames...")
    for i in range(total_frames):
        if i % 10 == 0:
            print(f"  Frame {i}/{total_frames}")
        img = create_frame(i, total_frames)
        frames.append(img)

    # Save as GIF with infinite loop
    output_path = Path('/Users/bo/Downloads/infinite_gains.gif')
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame = 10fps
        loop=0,  # 0 = infinite loop
        optimize=True
    )
    print(f"GIF saved to {output_path} ({len(frames)} frames, infinite loop)")

    # Create icon
    create_icon()

if __name__ == '__main__':
    main()
