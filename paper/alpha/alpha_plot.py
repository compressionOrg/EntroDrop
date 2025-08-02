#!/usr/bin/env python3
"""
Alpha参数敏感性分析可视化脚本

根据不同alpha值和层数的实验结果，生成折线图和热力图来展示参数对模型性能的影响。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置全局字体
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

def load_alpha_data() -> pd.DataFrame:
    """
    加载alpha参数实验数据
    
    Returns:
        包含实验结果的DataFrame
    """
    # 根据提供的表格数据创建DataFrame
    data = {
        'L': [8, 8, 10, 10, 12, 12] * 9,  # 层数
        'Method': ['Ours (MSE)', 'Ours (MAE)'] * 27,  # 方法
        'Alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 6,  # alpha值
        'Performance': [
            # Alpha 0.1
            46.30, 46.30, 39.21, 43.21, 38.68, 38.68,
            # Alpha 0.2
            46.30, 46.30, 39.21, 43.21, 38.68, 38.68,
            # Alpha 0.3
            46.30, 46.30, 43.21, 43.21, 38.68, 38.68,
            # Alpha 0.4
            46.30, 48.44, 43.21, 43.21, 38.68, 38.68,
            # Alpha 0.5
            48.27, 48.27, 43.21, 43.21, 38.68, 38.68,
            # Alpha 0.6
            46.30, 49.01, 43.21, 43.36, 38.68, 38.77,
            # Alpha 0.7
            48.44, 49.01, 43.21, 43.36, 38.68, 41.21,
            # Alpha 0.8
            49.01, 47.45, 43.36, 44.23, 38.77, 39.70,
            # Alpha 0.9
            47.01, 30.63, 43.84, 30.92, 40.08, 30.35
        ]
    }
    
    # 重新组织数据以匹配实际结构
    rows = []
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # MSE数据
    mse_8_layers = [46.30, 46.30, 46.30, 46.30, 48.27, 46.30, 48.44, 49.01, 47.01, 31.65]
    mse_10_layers = [39.21, 39.21, 43.21, 43.21, 43.21, 43.21, 43.21, 43.36, 43.84, 30.81]
    mse_12_layers = [38.68, 38.68, 38.68, 38.68, 38.68, 38.68, 38.68, 38.77, 40.08, 31.51]
    
    # MAE数据
    mae_8_layers = [46.30, 46.30, 46.30, 48.44, 48.27, 49.01, 49.01, 47.45, 30.63, 31.27]
    mae_10_layers = [43.21, 43.21, 43.21, 43.21, 43.21, 43.36, 43.36, 44.23, 30.92, 30.89]
    mae_12_layers = [38.68, 38.68, 38.68, 38.68, 38.68, 38.77, 41.21, 39.70, 30.35, 30.72]
    
    for i, alpha in enumerate(alpha_values):
        rows.extend([
            {'L': 8, 'Method': 'Ours(MSE)', 'Alpha': alpha, 'Performance': mse_8_layers[i]},
            {'L': 8, 'Method': 'Ours(MAE)', 'Alpha': alpha, 'Performance': mae_8_layers[i]},
            {'L': 10, 'Method': 'Ours(MSE)', 'Alpha': alpha, 'Performance': mse_10_layers[i]},
            {'L': 10, 'Method': 'Ours(MAE)', 'Alpha': alpha, 'Performance': mae_10_layers[i]},
            {'L': 12, 'Method': 'Ours(MSE)', 'Alpha': alpha, 'Performance': mse_12_layers[i]},
            {'L': 12, 'Method': 'Ours(MAE)', 'Alpha': alpha, 'Performance': mae_12_layers[i]}
        ])
    
    return pd.DataFrame(rows)

def create_line_plot(df: pd.DataFrame, save_path: str = "finetuning.pdf") -> None:
    """
    创建折线图展示alpha参数对性能的影响（两个子图上下排列）
    
    Args:
        df: 包含实验数据的DataFrame
        save_path: 保存路径
    """
    # 创建两个子图，上下排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 学术风格配色
    colors = {
        8: '#1f77b4',   # 蓝色
        10: '#ff7f0e',  # 橙色
        12: '#2ca02c'   # 绿色
    }
    
    markers = {8: 'o', 10: 's', 12: '^'}  # 圆形、方形、三角形
    
    # 绘制MSE方法的折线图（上子图）
    mse_data = df[df['Method'] == 'Ours(MSE)']
    for layers in mse_data['L'].unique():
        subset = mse_data[mse_data['L'] == layers]
        if not subset.empty:
            ax1.plot(subset['Alpha'], subset['Performance'], 
                    color=colors[layers],
                    linestyle='-',
                    marker=markers[layers],
                    markersize=8,
                    linewidth=2.5,
                    label=f'{layers} layers',
                    alpha=0.8)
    
    # 设置上子图样式
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Performance (%)')
    ax1.set_title('Ours(MSE) Performance vs Alpha Parameter')
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_xlim(0.05, 1.05)
    ax1.set_ylim(25, 55)
    ax1.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.legend(frameon=True, fancybox=False, shadow=False, 
              edgecolor='#CCCCCC', facecolor='white', framealpha=0.9)
    
    # 绘制MAE方法的折线图（下子图）
    mae_data = df[df['Method'] == 'Ours(MAE)']
    for layers in mae_data['L'].unique():
        subset = mae_data[mae_data['L'] == layers]
        if not subset.empty:
            ax2.plot(subset['Alpha'], subset['Performance'], 
                    color=colors[layers],
                    linestyle='--',
                    marker=markers[layers],
                    markersize=8,
                    linewidth=2.5,
                    label=f'{layers} layers',
                    alpha=0.8)
    
    # 设置下子图样式
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel('Performance (%)')
    ax2.set_title('Ours(MAE) Performance vs Alpha Parameter')
    ax2.grid(True, alpha=0.3, linewidth=0.8)
    ax2.set_xlim(0.05, 1.05)
    ax2.set_ylim(25, 55)
    ax2.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.legend(frameon=True, fancybox=False, shadow=False, 
              edgecolor='#CCCCCC', facecolor='white', framealpha=0.9)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存为PDF
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.show()
    print(f"折线图已保存到: {save_path}")

def create_heatmap(df: pd.DataFrame, save_path: str = "alpha_heatmap.pdf") -> None:
    """
    创建热力图展示参数敏感性（两个子图上下排列）
    
    Args:
        df: 包含实验数据的DataFrame
        save_path: 保存路径
    """
    # 为MSE和MAE分别创建热力图，上下排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
    
    for i, method in enumerate(['Ours(MSE)', 'Ours(MAE)']):
        method_data = df[df['Method'] == method]
        
        # 创建透视表
        pivot_table = method_data.pivot(index='L', columns='Alpha', values='Performance')
        
        # 确保行和列的顺序
        pivot_table = pivot_table.reindex([8, 10, 12])  # 层数从小到大
        pivot_table = pivot_table.reindex(columns=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # alpha从小到大
        
        # 创建热力图
        ax = ax1 if i == 0 else ax2
        
        # 使用学术风格的颜色映射
        cmap = 'RdYlBu_r' if method == 'Ours(MSE)' else 'RdYlGn'
        
        sns.heatmap(pivot_table, 
                   annot=True, 
                   fmt='.2f', 
                   cmap=cmap,
                   center=pivot_table.values.mean(),
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'shrink': 0.8},
                   ax=ax,
                   annot_kws={'fontsize': 9})
        
        # 添加子图标识
        subplot_label = 'a' if i == 0 else 'b'
        # 将标题放在图的下方，缩短间距
        ax.text(0.5, -0.1, f'({subplot_label}) {method} Zero-Shot Performance', 
                fontsize=12, fontweight='medium', ha='center', va='top', 
                transform=ax.transAxes)
        ax.set_xlabel(r'$\alpha$', fontsize=11, fontweight='normal')
        ax.set_ylabel('Number of Pruned Layers', fontsize=11, fontweight='medium')
        
        # 将x轴标签和刻度移动到上方
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        
        # 设置刻度标签
        ax.set_xticklabels([f'{x:.1f}' for x in pivot_table.columns], rotation=0)
        ax.set_yticklabels([f'{int(y)}' for y in pivot_table.index], rotation=0)
    
    # 调整布局
    plt.tight_layout()
    # 保存为PDF
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.show()
    print(f"热力图已保存到: {save_path}")

def create_combined_analysis(df: pd.DataFrame) -> None:
    """
    创建综合分析图表
    
    Args:
        df: 包含实验数据的DataFrame
    """
    # 找出最优参数组合
    best_configs = df.loc[df.groupby(['Method', 'L'])['Performance'].idxmax()]
    
    print("\n=== 最优参数配置分析 ===")
    for _, row in best_configs.iterrows():
        print(f"{row['Method']} {row['L']}层: Alpha={row['Alpha']:.1f}, Performance={row['Performance']:.2f}%")
    
    # 计算性能统计
    print("\n=== 性能统计分析 ===")
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        print(f"\n{method}:")
        print(f"  平均性能: {method_data['Performance'].mean():.2f}%")
        print(f"  最高性能: {method_data['Performance'].max():.2f}%")
        print(f"  最低性能: {method_data['Performance'].min():.2f}%")
        print(f"  标准差: {method_data['Performance'].std():.2f}%")

def main():
    """
    主函数
    """
    print("开始生成Alpha参数敏感性分析图表...")
    
    # 加载数据
    df = load_alpha_data()
    print(f"数据加载完成，共{len(df)}条记录")
    
    # 创建折线图
    # create_line_plot(df)
    
    # 创建热力图
    create_heatmap(df, "alpha.pdf")
    
    # 综合分析
    create_combined_analysis(df)
    
    print("\n所有图表生成完成！")

if __name__ == "__main__":
    main()