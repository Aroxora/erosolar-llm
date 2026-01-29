#!/usr/bin/env python3
"""
Generate graphics using ONLY verified claims from the PCGAT paper.

Verified metrics (from paper/main.tex):
- 35% of GPT-5.2 outputs contain verifiable errors
- 12-18% benchmark improvement with verified data
- Author: Bo Shang, Erosolar AI Research
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import imageio.v2 as imageio
from pathlib import Path


# ============================================================================
# VERIFIED DATA ONLY - from paper/main.tex
# ============================================================================
VERIFIED_METRICS = {
    "Error Detection Rate": "35%",
    "Benchmark Improvement": "12-18%",
    "Verification Methods": "5",
    "Provable Correctness": "≥ 1-ε",
}

AUTHOR = "Bo Shang"
AFFILIATION = "Erosolar AI Research"
PROJECT = "PCGAT"
PAPER_TITLE = "Provably Correct Generative Adversarial Training"


def create_font(size: int, bold: bool = False):
    """Create font with fallback."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_verification_pipeline(draw, frame: int, width: int, height: int):
    """Draw the PCGAT verification pipeline with animation."""

    # Colors
    bg_dark = (15, 20, 35)
    green = (0, 255, 136)
    cyan = (0, 200, 255)
    magenta = (200, 80, 200)
    white = (255, 255, 255)
    dim_white = (150, 150, 150)

    # Title
    title_font = create_font(24, bold=True)
    draw.text((width//2, 40), "PCGAT Verification Pipeline",
              fill=green, font=title_font, anchor="mm")

    subtitle_font = create_font(14)
    draw.text((width//2, 70), "Provably Correct Generative Adversarial Training",
              fill=dim_white, font=subtitle_font, anchor="mm")

    # Pipeline boxes - moved up to make room for metrics
    box_w, box_h = 120, 60
    boxes = [
        ("Teacher\nModel", 100),
        ("Candidate\nGenerator", 270),
        ("Grounded\nVerifiers", 440),
        ("Verified\nDataset", 610),
    ]

    y_center = 200  # Fixed position, higher up

    # Draw connections with animation
    for i, (label, x) in enumerate(boxes[:-1]):
        next_x = boxes[i+1][1]
        # Animated line
        progress = (frame % 30) / 30
        line_length = next_x - x - box_w
        animated_end = x + box_w + int(line_length * progress)

        if i == 1:  # Main flow gets animated
            draw.line([(x + box_w, y_center), (animated_end, y_center)],
                     fill=cyan, width=3)
        else:
            draw.line([(x + box_w, y_center), (next_x, y_center)],
                     fill=dim_white, width=2)

    # Draw boxes
    colors = [dim_white, cyan, green, magenta]
    for i, (label, x) in enumerate(boxes):
        color = colors[i]
        draw.rectangle([x, y_center - box_h//2, x + box_w, y_center + box_h//2],
                      outline=color, width=2)

        # Label
        label_font = create_font(11)
        lines = label.split('\n')
        for j, line in enumerate(lines):
            y_offset = (j - len(lines)/2 + 0.5) * 16
            draw.text((x + box_w//2, y_center + y_offset), line,
                     fill=color, font=label_font, anchor="mm")

    # Verifier sub-components (5 verifiers from paper)
    verifiers = ["Code Exec", "Symbolic", "Factual DB", "Logic", "Multi-Path"]
    v_y = y_center + box_h//2 + 35
    v_x = 440

    small_font = create_font(9)
    for i, v in enumerate(verifiers):
        vx = v_x + (i - 2) * 50
        # Highlight one verifier per cycle
        highlight = (frame // 10) % 5 == i
        color = green if highlight else dim_white
        draw.rectangle([vx - 20, v_y - 12, vx + 20, v_y + 12],
                      outline=color, width=1 if not highlight else 2)
        draw.text((vx, v_y), v[:6], fill=color, font=small_font, anchor="mm")

        # Connection to main verifier box
        draw.line([(vx, v_y - 12), (440 + 60, y_center + box_h//2)],
                 fill=dim_white if not highlight else green, width=1)


def draw_metrics(draw, frame: int, width: int, height: int):
    """Draw verified metrics at the bottom."""

    green = (0, 255, 136)
    cyan = (0, 200, 255)
    dim = (140, 140, 140)
    white = (255, 255, 255)

    # Draw background bar for metrics - always visible
    draw.rectangle([0, 340, width, height], fill=(10, 15, 28))
    draw.line([(0, 340), (width, 340)], fill=green, width=2)

    # Section title
    section_font = create_font(16, bold=True)
    draw.text((width//2, 365), "VERIFIED RESULTS", fill=green, font=section_font, anchor="mm")

    label_font = create_font(14)
    value_font = create_font(22, bold=True)

    # Key verified metrics in two rows
    row1_y = 420
    row2_y = 480

    metrics_row1 = [
        ("Error Detection:", "35%", 200),
        ("Benchmark Improvement:", "12-18%", 580),
    ]

    metrics_row2 = [
        ("Verification Methods:", "5", 200),
        ("Provable Correctness:", "≥ 1-ε", 580),
    ]

    # Always show metrics (no animation delay for clarity)
    for label, value, x in metrics_row1:
        draw.text((x - 90, row1_y), label, fill=dim, font=label_font, anchor="lm")
        draw.text((x + 130, row1_y), value, fill=green, font=value_font, anchor="rm")

    for label, value, x in metrics_row2:
        draw.text((x - 90, row2_y), label, fill=dim, font=label_font, anchor="lm")
        draw.text((x + 130, row2_y), value, fill=cyan, font=value_font, anchor="rm")

    # Source citation
    cite_font = create_font(11)
    draw.text((width//2, height - 25),
             "Source: Verified empirical results from PCGAT paper (Shang, 2026)",
             fill=(90, 90, 90), font=cite_font, anchor="mm")


def generate_pcgat_gif(output_path: str = "deepseeker_qkv_animation.gif"):
    """Generate the PCGAT animation GIF with verified data only."""

    width, height = 800, 600
    frames = []
    num_frames = 60

    bg_color = (15, 20, 35)

    print(f"Generating {num_frames} frames...")

    for frame in range(num_frames):
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Add subtle grid background
        for x in range(0, width, 40):
            draw.line([(x, 0), (x, height)], fill=(25, 30, 45), width=1)
        for y in range(0, height, 40):
            draw.line([(0, y), (width, y)], fill=(25, 30, 45), width=1)

        # Draw main content
        draw_verification_pipeline(draw, frame, width, height)
        draw_metrics(draw, frame, width, height)

        frames.append(np.array(img))

        if (frame + 1) % 10 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")

    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, duration=0.1, loop=0)
    print("Done!")

    return output_path


def enhance_portrait(input_path: str, output_path: str = "Bo_Shang_deepseeker.png"):
    """Enhance Bo Shang's portrait with verified project information."""

    print(f"Loading portrait from {input_path}...")

    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"Portrait not found at {input_path}, creating placeholder...")
        img = Image.new('RGB', (1360, 1760), (20, 40, 20))

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size

    # Apply green tint effect (deepseeker style)
    r, g, b = img.split()

    # Enhance green channel, reduce red and blue
    r = r.point(lambda x: int(x * 0.3))
    g = g.point(lambda x: min(255, int(x * 1.4)))
    b = b.point(lambda x: int(x * 0.4))

    img = Image.merge('RGB', (r, g, b))

    # Add subtle glow
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)

    # Create overlay with verified info
    draw = ImageDraw.Draw(img)

    # Colors
    green = (0, 255, 136)
    dim_green = (0, 180, 100)
    dark_green = (0, 80, 50)

    # Top bar with project name
    draw.rectangle([0, 0, width, 60], fill=(0, 20, 10, 200))

    header_font = create_font(24, bold=True)
    small_font = create_font(14)

    # Header text
    draw.text((20, 30), f"PCGAT Research", fill=green, font=header_font, anchor="lm")
    draw.text((width - 20, 30), f"{AUTHOR}", fill=dim_green, font=small_font, anchor="rm")

    # Bottom info bar
    bar_height = 120
    draw.rectangle([0, height - bar_height, width, height], fill=(0, 20, 10))

    # Project branding
    brand_font = create_font(36, bold=True)
    draw.text((width//2, height - 80), "deepseeker", fill=green, font=brand_font, anchor="mm")

    subtitle_font = create_font(16)
    draw.text((width//2, height - 40),
             "Provably Correct Generative Adversarial Training",
             fill=dim_green, font=subtitle_font, anchor="mm")

    # Verified metrics badge
    badge_font = create_font(12)
    draw.text((width//2, height - 15),
             "35% Error Detection | 12-18% Benchmark Improvement",
             fill=dark_green, font=badge_font, anchor="mm")

    # Scanline effect
    for y in range(0, height, 4):
        draw.line([(0, y), (width, y)], fill=(0, 0, 0), width=1)

    # Frame border
    draw.rectangle([0, 0, width-1, height-1], outline=green, width=3)

    print(f"Saving enhanced portrait to {output_path}...")
    img.save(output_path, 'PNG')
    print("Done!")

    return output_path


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("PCGAT Graphics Generator - Verified Data Only")
    print("=" * 60)
    print()
    print("Verified metrics from paper/main.tex:")
    for k, v in VERIFIED_METRICS.items():
        print(f"  - {k}: {v}")
    print()

    # Generate GIF
    gif_path = Path(__file__).parent / "deepseeker_qkv_animation.gif"
    generate_pcgat_gif(str(gif_path))

    print()

    # Enhance portrait (use existing if available)
    portrait_input = Path(__file__).parent / "Bo_Shang_deepseeker.png"
    portrait_output = Path(__file__).parent / "Bo_Shang_deepseeker_enhanced.png"

    if portrait_input.exists():
        enhance_portrait(str(portrait_input), str(portrait_output))
    else:
        print(f"No existing portrait found at {portrait_input}")
        print("Creating new portrait with verified info only...")
        enhance_portrait(str(portrait_input), str(portrait_output))

    print()
    print("=" * 60)
    print("Graphics generated with VERIFIED data only!")
    print("=" * 60)
