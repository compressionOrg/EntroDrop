#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Pruning Speedup Visualization

This module creates a dual-axis line plot to visualize the relationship between
model pruning ratio, throughput, and speedup across different layers.

Author: Assistant
Date: 2024
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.font_manager as fm

# Set global font configuration for academic papers
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12


def create_dual_axis_plot(
    data: Dict[str, List[float]], 
    save_path: str = "speedup_analysis.pdf",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300
) -> None:
    """
    Create a dual-axis line plot showing pruning ratio vs throughput and speedup.
    
    Args:
        data: Dictionary containing 'Layer', 'Ratio', 'Throughput', 'Speedup' lists
        save_path: Path to save the generated plot
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
    """
    # Extract data
    layers = data['Layer']
    ratios = data['Ratio']
    throughputs = data['Throughput']
    speedups = data['Speedup']
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Define colors for better academic presentation
    color_throughput = '#2E86AB'  # Blue
    color_speedup = '#A23B72'     # Purple
    color_ratio = '#F18F01'       # Orange
    
    # Plot pruning ratio on primary axis
    line1 = ax1.plot(layers, ratios, 'o-', color=color_ratio, 
                     linewidth=2, markersize=6, label='Pruning Ratio')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Pruning Ratio', color=color_ratio)
    ax1.tick_params(axis='y', labelcolor=color_ratio)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary axis for throughput and speedup
    ax2 = ax1.twinx()
    
    # Plot throughput and speedup on secondary axis
    line2 = ax2.plot(layers, throughputs, 's-', color=color_throughput, 
                     linewidth=2, markersize=6, label='Throughput (tokens/s)')
    line3 = ax2.plot(layers, speedups, '^-', color=color_speedup, 
                     linewidth=2, markersize=6, label='Speedup')
    
    ax2.set_ylabel('Throughput (tokens/s) / Speedup', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Combine legends from both axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, 
              fancybox=True, shadow=True)
    
    # Set title and layout
    plt.title('Model Pruning Performance Analysis', fontweight='bold', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Dual-axis plot saved as: {save_path}")


def main() -> None:
    """
    Main function to demonstrate the visualization functions.
    """
    # Sample data based on user's previous input
    sample_data = {
        'Layer': [12, 16, 20, 24, 28],
        'Ratio': [0.25, 0.375, 0.5, 0.625, 0.75],
        'Throughput': [45.2, 52.8, 61.3, 68.9, 76.4],
        'Speedup': [1.2, 1.4, 1.6, 1.8, 2.0]
    }
    
    print("Creating model pruning dual-axis visualization...")
    print("="*50)
    
    # Create dual-axis plot (most recommended)
    print("Creating dual-axis line plot...")
    create_dual_axis_plot(sample_data, "speedup_dual_axis.pdf")
    
    print("\n" + "="*50)
    print("Dual-axis plot has been generated successfully!")
    print("Output file: speedup_dual_axis.pdf")


if __name__ == "__main__":
    main()