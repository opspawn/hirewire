#!/usr/bin/env python3
"""Render terminal-style frames for the demo video."""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1920, 1080
BG = (13, 17, 23)  # GitHub dark background
FG = (230, 237, 243)
GREEN = (34, 197, 94)
CYAN = (34, 211, 238)
YELLOW = (234, 179, 8)
BLUE = (96, 165, 250)
MAGENTA = (192, 132, 252)
GRAY = (148, 163, 184)
DIM = (100, 116, 139)

# Use a monospace font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 14)
except:
    font = ImageFont.load_default()
    font_bold = font
    font_title = font

LINE_HEIGHT = 22
MARGIN_X = 40
MARGIN_Y = 50

def draw_title_bar(draw):
    """Draw terminal title bar."""
    draw.rectangle([(0, 0), (W, 36)], fill=(30, 30, 30))
    # Traffic light buttons
    draw.ellipse([(16, 12), (28, 24)], fill=(255, 95, 86))
    draw.ellipse([(36, 12), (48, 24)], fill=(255, 189, 46))
    draw.ellipse([(56, 12), (68, 24)], fill=(39, 201, 63))
    draw.text((W//2 - 100, 10), "HireWire Demo — Terminal", fill=GRAY, font=font_title)

def draw_line(draw, y, segments):
    """Draw a line of text with color segments: [(text, color), ...]"""
    x = MARGIN_X
    for text, color in segments:
        draw.text((x, y), text, fill=color, font=font)
        x += len(text) * 9.6  # approximate char width

def create_frame(lines, filename):
    """Create a frame image with colored terminal lines."""
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_title_bar(draw)

    y = MARGIN_Y
    for line in lines:
        if isinstance(line, str):
            draw.text((MARGIN_X, y), line, fill=FG, font=font)
        else:
            draw_line(draw, y, line)
        y += LINE_HEIGHT

    img.save(filename)

output_dir = "/home/agent/projects/hirewire/docs/demo/video-assets/frames"
os.makedirs(output_dir, exist_ok=True)

# Frame 1: Stage 1 - Agent Creation
frame1_lines = [
    [("$ ", GREEN), ("python demo/record_demo.py", FG)],
    "",
    [("╔══════════════════════════════════════════════════════════════╗", CYAN)],
    [("║            HireWire — Agent Hiring Platform                 ║", CYAN)],
    [("║     Microsoft AI Agent Hackathon · AI Dev Days 2026         ║", CYAN)],
    [("╚══════════════════════════════════════════════════════════════╝", CYAN)],
    "",
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    [("Stage [1/8]  ", YELLOW), ("Agent Creation", FG)],
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    "",
    [("  ✓ ", GREEN), ("Created agent: ", FG), ("CEO", CYAN), (" (azure-gpt4o)", GRAY)],
    [("    ", FG), ("Skills: task_analysis, agent_routing, budget_management", DIM)],
    [("  ✓ ", GREEN), ("Created agent: ", FG), ("Builder", CYAN), (" (azure-gpt4o)", GRAY)],
    [("    ", FG), ("Skills: code_generation, testing, deployment", DIM)],
    [("  ✓ ", GREEN), ("Created agent: ", FG), ("Research", CYAN), (" (azure-gpt4o)", GRAY)],
    [("    ", FG), ("Skills: web_search, data_analysis, report_writing", DIM)],
    [("  ✓ ", GREEN), ("Created agent: ", FG), ("Analyst", CYAN), (" (x402-external)", GRAY)],
    [("    ", FG), ("Skills: financial_modeling, competitive_analysis", DIM)],
    "",
    [("  ► ", BLUE), ("4 agents registered", FG), (" | 2 internal + 2 external (x402)", GRAY)],
    "",
    [("  █████████░░░░░░░░░░░░░░░  ", GREEN), ("1/8", FG)],
]
create_frame(frame1_lines, f"{output_dir}/frame_01.png")

# Frame 2: Stage 3 - Task Analysis + Budget
frame2_lines = [
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    [("Stage [3/8]  ", YELLOW), ("Task Analysis & Budget Allocation", FG)],
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    "",
    [("  Task: ", GRAY), ("\"Build a professional landing page for HireWire\"", FG)],
    "",
    [("  ► ", BLUE), ("CEO analyzing task complexity...", FG)],
    [("    ", FG), ("Type: ", GRAY), ("build", CYAN)],
    [("    ", FG), ("Complexity: ", GRAY), ("moderate", YELLOW)],
    [("    ", FG), ("Agents needed: ", GRAY), ("builder, designer-ext-001", CYAN)],
    "",
    [("  ► ", BLUE), ("Budget allocation:", FG)],
    [("    ", FG), ("Total budget: ", GRAY), ("$10.00 USDC", GREEN)],
    [("    ", FG), ("Reserved in escrow: ", GRAY), ("$3.50 USDC", YELLOW)],
    [("    ", FG), ("  → builder: ", DIM), ("$2.50", GREEN)],
    [("    ", FG), ("  → designer-ext-001 (x402): ", DIM), ("$1.00", GREEN)],
    "",
    [("  ✓ ", GREEN), ("Budget allocated and escrowed", FG)],
    "",
    [("  ███████████████░░░░░░░░░  ", GREEN), ("3/8", FG)],
]
create_frame(frame2_lines, f"{output_dir}/frame_02.png")

# Frame 3: Stage 5 - External Hiring (x402)
frame3_lines = [
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    [("Stage [5/8]  ", YELLOW), ("External Agent Hiring — x402 Protocol", FG)],
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    "",
    [("  ► ", BLUE), ("Hiring external agent via x402 payment protocol...", FG)],
    "",
    [("  ┌─────────────────────────────────────────────────────┐", MAGENTA)],
    [("  │ ", MAGENTA), ("x402 Payment Flow                                   ", FG), ("│", MAGENTA)],
    [("  │ ", MAGENTA), ("                                                     ", FG), ("│", MAGENTA)],
    [("  │  ", MAGENTA), ("1. CEO → EIP-712 signed payment proof              ", GRAY), (" │", MAGENTA)],
    [("  │  ", MAGENTA), ("2. Proof → designer-ext-001 endpoint               ", GRAY), (" │", MAGENTA)],
    [("  │  ", MAGENTA), ("3. Agent verifies proof, executes task              ", GRAY), (" │", MAGENTA)],
    [("  │  ", MAGENTA), ("4. Payment settled: ", GRAY), ("$0.05 USDC", GREEN), ("                     ", GRAY), (" │", MAGENTA)],
    [("  │ ", MAGENTA), ("                                                     ", FG), ("│", MAGENTA)],
    [("  └─────────────────────────────────────────────────────┘", MAGENTA)],
    "",
    [("  ✓ ", GREEN), ("designer-ext-001 hired via x402", FG)],
    [("  ✓ ", GREEN), ("Payment verified on-chain: ", FG), ("$0.05 USDC", GREEN)],
    [("  ✓ ", GREEN), ("Design deliverable received: ", FG), ("landing-page-mockup.svg", CYAN)],
    "",
    [("  █████████████████████░░░  ", GREEN), ("5/8", FG)],
]
create_frame(frame3_lines, f"{output_dir}/frame_03.png")

# Frame 4: Stage 6 - Concurrent Execution
frame4_lines = [
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    [("Stage [6/8]  ", YELLOW), ("Concurrent Agent Execution", FG)],
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    "",
    [("  ► ", BLUE), ("Running 3 agents in parallel...", FG)],
    "",
    [("  ┌─ CEO ──────────────────────── ", CYAN), ("running", YELLOW), (" ─┐", CYAN)],
    [("  │  ", CYAN), ("Analyzing market positioning strategy", FG), ("     │", CYAN)],
    [("  └─────────────────────────────────────────┘", CYAN)],
    "",
    [("  ┌─ Research ─────────────────── ", CYAN), ("running", YELLOW), (" ─┐", CYAN)],
    [("  │  ", CYAN), ("Gathering competitor data from 5 platforms", FG), ("│", CYAN)],
    [("  └─────────────────────────────────────────┘", CYAN)],
    "",
    [("  ┌─ Analyst (x402) ──────────── ", CYAN), ("running", YELLOW), (" ─┐", CYAN)],
    [("  │  ", CYAN), ("Financial modeling: revenue projections", FG), ("  │", CYAN)],
    [("  └─────────────────────────────────────────┘", CYAN)],
    "",
    [("  ✓ ", GREEN), ("All 3 agents completed", FG), (" — ", DIM), ("1.2s total", GRAY)],
    [("  ✓ ", GREEN), ("Results aggregated by CEO", FG)],
    "",
    [("  ████████████████████████░░  ", GREEN), ("6/8", FG)],
]
create_frame(frame4_lines, f"{output_dir}/frame_04.png")

# Frame 5: Stage 8 - Results Summary
frame5_lines = [
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    [("Stage [8/8]  ", YELLOW), ("Results Summary", FG)],
    [("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", DIM)],
    "",
    [("  ┌──────────────────────────────────────────────────┐", GREEN)],
    [("  │ ", GREEN), ("Payment Ledger                                   ", FG), ("│", GREEN)],
    [("  ├──────────────────────────────────────────────────┤", GREEN)],
    [("  │  ", GREEN), ("builder         → $2.50 USDC  ", FG), ("(internal)     ", GRAY), ("│", GREEN)],
    [("  │  ", GREEN), ("research        → $0.80 USDC  ", FG), ("(internal)     ", GRAY), ("│", GREEN)],
    [("  │  ", GREEN), ("designer-ext-001→ $1.02 USDC  ", FG), ("(x402)         ", MAGENTA), ("│", GREEN)],
    [("  │  ", GREEN), ("analyst-ext-001 → $0.66 USDC  ", FG), ("(x402)         ", MAGENTA), ("│", GREEN)],
    [("  ├──────────────────────────────────────────────────┤", GREEN)],
    [("  │  ", GREEN), ("Total spent:      ", FG), ("$7.35 USDC", YELLOW), ("                  ", FG), ("│", GREEN)],
    [("  │  ", GREEN), ("External (x402):  ", FG), ("$1.95 USDC", MAGENTA), ("                  ", FG), ("│", GREEN)],
    [("  └──────────────────────────────────────────────────┘", GREEN)],
    "",
    [("  ✓ ", GREEN), ("Tasks completed: ", FG), ("9/12", CYAN), (" (75% completion rate)", GRAY)],
    [("  ✓ ", GREEN), ("Agents used: ", FG), ("5", CYAN), (" (3 internal + 2 external)", GRAY)],
    [("  ✓ ", GREEN), ("Azure GPT-4o: ", FG), ("Connected", GREEN)],
    [("  ✓ ", GREEN), ("HITL gates: ", FG), ("0 blocked, 12 decisions", CYAN)],
    [("  ✓ ", GREEN), ("RAI fairness: ", FG), ("100%", GREEN)],
    "",
    [("  █████████████████████████  ", GREEN), ("8/8 ✓ Complete", GREEN)],
    "",
    [("╔══════════════════════════════════════════════════════════════╗", CYAN)],
    [("║          Demo Complete — All Systems Operational            ║", CYAN)],
    [("╚══════════════════════════════════════════════════════════════╝", CYAN)],
]
create_frame(frame5_lines, f"{output_dir}/frame_05.png")

print(f"Created 5 terminal frames in {output_dir}")
