#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motivation Plot: WikiText2 PPL vs Pruning Layers with Log Scale

This module creates a line plot showing how WikiText2 perplexity (PPL) changes
with the number of pruning layers for different models, using a logarithmic Y-axis
to better visualize the exponential growth pattern.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# Set matplotlib parameters for academic paper style
plt.rcParams.update({
    'font.family': 'serif',          # 设置字体系列
    'font.serif': ['Times New Roman'], # 设置具体字体
    'axes.labelsize': 10,            # 设置坐标轴标签字体大小
    'xtick.labelsize': 10,            # 设置x轴刻度字体大小
    'ytick.labelsize': 10,            # 设置y轴刻度字体大小
    'legend.fontsize': 10,            # 设置图例字体大小
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300
})


def create_motivation_plot(save_path: str = "motivation.pdf") -> None:
    """
    Create a line plot showing WikiText2 PPL vs Pruning Layers with logarithmic Y-axis.
    
    Args:
        save_path: Path to save the generated plot
    """
    # Data from the provided table
    pruning_layers = [2, 4, 6, 8, 10, 12]
    
    # PPL data for each model
    models_data = {
        'LLaMA2-7B': [6.30, 8.45, 15.46, 25.41, 49.55, 79.49],
        'Mistral-7B-v0.3': [6.19, 8.26, 12.17, 42.18, 811.3, 1833.00],
        'LLaMA3.1-8B': [7.57, 14.20, 45.95, 2811.85, 14202.05, 80382.13]
    }
    
    # Color scheme for academic papers
    colors = {
        'LLaMA2-7B': '#1f77b4',      # Blue
        'Mistral-7B-v0.3': '#ff7f0e', # Orange
        'LLaMA3.1-8B': '#2ca02c'      # Green
    }
    
    # Line styles for better distinction
    line_styles = {
        'LLaMA2-7B': '-',
        'Mistral-7B-v0.3': '--',
        'LLaMA3.1-8B': '-.'
    }
    
    # Markers for data points
    markers = {
        'LLaMA2-7B': 'o',
        'Mistral-7B-v0.3': 's',
        'LLaMA3.1-8B': '^'
    }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Plot each model's data
    for model_name, ppl_values in models_data.items():
        ax.plot(
            pruning_layers, 
            ppl_values,
            color=colors[model_name],
            linestyle=line_styles[model_name],
            marker=markers[model_name],
            markersize=8,
            linewidth=2.5,
            label=model_name,
            markerfacecolor='white',
            markeredgewidth=2,
            markeredgecolor=colors[model_name]
        )
    
    # Set logarithmic scale for Y-axis
    ax.set_yscale('log')
    
    # Customize the plot
    ax.set_xlabel('Number of Pruner Layers')
    ax.set_ylabel('WikiText2 PPL')
    ax.set_title('')
    
    # Set x-axis ticks
    ax.set_xticks(pruning_layers)
    ax.set_xlim(1.5, 12.5)
    
    # Customize grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    legend = ax.legend(
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        edgecolor='black',
        facecolor='white'
    )
    legend.get_frame().set_linewidth(1.0)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=600)
    print(f"Motivation plot saved to: {save_path}")
    
    # Show the plot
    plt.show()


def analyze_degradation_patterns() -> Dict[str, float]:
    """
    Analyze the degradation patterns of different models.
    
    Returns:
        Dictionary containing degradation analysis results
    """
    models_data = {
        'LLaMA2-7B': [6.30, 8.45, 15.46, 25.41, 49.55, 79.49],
        'Mistral-7B-v0.3': [6.19, 8.26, 12.17, 42.18, 811.3, 1833.00],
        'LLaMA3.1-8B': [7.57, 14.20, 45.95, 2811.85, 14202.05, 80382.13]
    }
    
    analysis = {}
    
    for model_name, ppl_values in models_data.items():
        # Calculate degradation ratio (final PPL / initial PPL)
        degradation_ratio = ppl_values[-1] / ppl_values[0]
        analysis[f"{model_name}_degradation_ratio"] = degradation_ratio
        
        # Calculate average growth rate
        log_values = np.log(ppl_values)
        growth_rate = (log_values[-1] - log_values[0]) / (len(log_values) - 1)
        analysis[f"{model_name}_avg_growth_rate"] = growth_rate
    
    return analysis


if __name__ == "__main__":
    # Create the motivation plot
    create_motivation_plot()
    
    # Analyze degradation patterns
    analysis = analyze_degradation_patterns()
    print("\nDegradation Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value:.4f}")