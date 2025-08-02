#!/usr/bin/env python3
"""
雷达图生成脚本

根据表格数据创建雷达图，展示不同方法在各个评估指标上的表现。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from typing import List, Dict, Any
import seaborn as sns

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


def create_radar_chart(
    data: Dict[str, List[float]], 
    categories: List[str],
    title: str = "",
    save_path: str = "radar_chart.pdf"
) -> None:
    """
    创建雷达图
    
    Args:
        data: 包含不同方法数据的字典，键为方法名，值为各指标的数值列表
        categories: 评估指标名称列表
        title: 图表标题
        save_path: 保存路径
    """
    # 计算角度
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 创建图形 - 优化尺寸比例避免遮挡
    fig, ax = plt.subplots(figsize=(6, 4.5), subplot_kw=dict(projection='polar'))
    
    # 设置学术论文风格的颜色 - 高对比度配色方案
    # 这些颜色具有鲜明对比，在黑白打印时也能很好地区分，且符合学术出版标准
    colors = [
        '#1f77b4',  # 鲜明蓝色 - Dense
        '#ff7f0e',  # 鲜明橙色 - Ours (MSE)
        '#2ca02c',  # 鲜明绿色 - Ours (MAE)
        '#d62728',  # 鲜明红色 - Ours (MSE)+LoRA
        '#9467bd',  # 鲜明紫色 - Ours (MAE)+LoRA
        '#8c564b',  # 棕色 - 额外对比
        '#e377c2',  # 粉红色 - 基准方法
        '#17becf'   # 青色 - 补充数据
    ]
    
    # 为每个方法绘制雷达图
    for i, (method, values) in enumerate(data.items()):
        values += values[:1]  # 闭合图形
        color = colors[i % len(colors)]
        
        # 绘制线条和填充 - 学术风格优化
        ax.plot(angles, values, 'o-', linewidth=2.5, label=method, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.15, color=color)  # 降低透明度以提高可读性
    
    # 设置标签 - 学术风格优化
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='medium')
    
    # 设置y轴范围 - 更精细的网格
    ax.set_ylim(0, 90)
    ax.set_yticks([20, 40, 60, 80, 90])
    ax.set_yticklabels(['20', '40', '60', '80', '90'], fontsize=9, color='#333333')
    ax.grid(True, alpha=0.3, linewidth=0.8)  # 更细的网格线
    
    # 添加标题和图例 - 学术风格
    plt.title(title, size=14, fontweight='medium', pad=20, color='#333333')
    plt.legend(loc='center', bbox_to_anchor=(0.5, 0.16), fontsize=8, 
              frameon=True, fancybox=False, shadow=False, 
              edgecolor='#CCCCCC', facecolor='white', framealpha=0.95)
    
    # 保存图片 - 优化布局参数避免遮挡
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    print(f"雷达图已保存到: {save_path}")

def load_and_process_data() -> Dict[str, Any]:
    """
    加载和处理数据
    
    Returns:
        处理后的数据字典
    """
    # 定义评估指标
    metrics = ['BoolQ', 'PIQA', 'HeSW', 'WinoG', 'ARC-e', 'ARC-c', 'OBQA', 'MTQA']
    
    # 定义数据
    data = {
        'Dense': [82.14, 80.20, 60.89, 73.88, 79.63, 48.89, 33.20, 35.38],
        'Ours (MSE) w/o LoRA': [45.81, 62.89, 34.12, 61.01, 40.45, 26.96, 17.6, 21.31],
        'Ours (MAE) w/o LoRA': [63.09, 62.35, 33.71, 56.04, 42.85, 31.06, 21.40, 19.20],
        'Ours (MSE) w/ LoRA': [71.47, 67.95, 44.55, 68.03, 62.92, 36.69, 23.20, 26.00],
        'Ours (MAE) w/ LoRA': [67.83, 70.13, 44.53, 66.38, 63.64, 33.87, 26.20, 25.46]
    }
    
    return {
        'data': data,
        'metrics': metrics
    }

def create_comparison_charts():
    """
    创建对比雷达图
    """
    data_dict = load_and_process_data()
    if not data_dict:
        return
    
    data = data_dict['data']
    metrics = data_dict['metrics']
    
    # 创建模型性能对比图
    create_radar_chart(
        data,
        metrics,
        "",  # 空标题
        "finetuning.pdf"  # 保存文件名
    )
    


def main():
    """
    主函数
    """
    print("开始生成雷达图...")
    create_comparison_charts()
    print("所有雷达图生成完成！")

if __name__ == "__main__":
    main()