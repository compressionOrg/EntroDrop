#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从日志文件中提取num prune值和各个评估指标的结果
"""

import re
import json
import csv
from typing import Dict, List, Any

def extract_results_from_log(log_file_path: str) -> List[Dict[str, Any]]:
    """
    从日志文件中提取实验结果
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        包含所有实验结果的列表
    """
    results = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_num_prune = None
    
    for i, line in enumerate(lines):
        # 查找num prune
        if 'Num prune:' in line:
            match = re.search(r'Num prune: (\d+)', line)
            if match:
                current_num_prune = int(match.group(1))
        
        # 查找包含完整评估结果的行
        if "evaluate_grasp - INFO - {'wikitext2':" in line and current_num_prune is not None:
            # 这一行包含完整的评估结果字典
            result_line = line.strip()
            
            # 提取字典部分
            start_idx = result_line.find("{'wikitext2':")
            if start_idx == -1:
                continue
                
            dict_str = result_line[start_idx:]
            
            # 处理numpy类型
            dict_str = re.sub(r'np\.float64\((.*?)\)', r'\1', dict_str)
            
            try:
                # 使用eval解析字典
                result_dict = eval(dict_str)
                
                # 提取各个指标并放大100倍，四舍五入保留两位小数
                def scale_and_round(value):
                    if value == 'N/A' or value is None:
                        return 'N/A'
                    return round(float(value) * 100, 2)
                
                experiment_result = {
                    'num_prune': current_num_prune,
                    'BoolQ': scale_and_round(result_dict.get('boolq', {}).get('acc', 'N/A')),
                    'PIQA': scale_and_round(result_dict.get('piqa', {}).get('acc', 'N/A')),
                    'HellaSwag': scale_and_round(result_dict.get('hellaswag', {}).get('acc', 'N/A')),
                    'WinoGrande': scale_and_round(result_dict.get('winogrande', {}).get('acc', 'N/A')),
                    'ARC-easy': scale_and_round(result_dict.get('arc_easy', {}).get('acc', 'N/A')),
                    'ARC-challenge': scale_and_round(result_dict.get('arc_challenge', {}).get('acc', 'N/A')),
                    'OpenBookQA': scale_and_round(result_dict.get('openbookqa', {}).get('acc', 'N/A')),
                    'MathQA': scale_and_round(result_dict.get('mathqa', {}).get('acc', 'N/A')),
                    'mean_acc': scale_and_round(result_dict.get('mean', 'N/A'))
                }
                
                results.append(experiment_result)
                current_num_prune = None  # 重置，避免重复
                
            except Exception as e:
                print(f"解析实验结果时出错 (num_prune={current_num_prune}): {e}")
                print(f"问题行: {dict_str[:100]}...")
                continue
    
    return results

def format_results_table(results: List[Dict[str, Any]]) -> str:
    """
    将结果格式化为表格
    
    Args:
        results: 实验结果列表
        
    Returns:
        格式化的表格字符串
    """
    if not results:
        return "没有找到有效的实验结果"
    
    # 表头
    header = "| Num Prune | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-easy | ARC-challenge | OpenBookQA | MathQA | Mean Acc |"
    separator = "|-----------|-------|------|-----------|------------|----------|---------------|------------|--------|----------|"
    
    table_lines = [header, separator]
    
    # 数据行
    for result in results:
        def format_value(val):
            return str(val) if val == 'N/A' else f"{val:.2f}"
        
        row = f"| {result['num_prune']} | {format_value(result['BoolQ'])} | {format_value(result['PIQA'])} | {format_value(result['HellaSwag'])} | {format_value(result['WinoGrande'])} | {format_value(result['ARC-easy'])} | {format_value(result['ARC-challenge'])} | {format_value(result['OpenBookQA'])} | {format_value(result['MathQA'])} | {format_value(result['mean_acc'])} |"
        table_lines.append(row)
    
    return "\n".join(table_lines)

def save_to_csv(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    将结果保存为CSV文件
    
    Args:
        results: 实验结果列表
        output_file: 输出CSV文件路径
    """
    if not results:
        return
    
    fieldnames = ['num_prune', 'BoolQ', 'PIQA', 'HellaSwag', 'WinoGrande', 
                  'ARC-easy', 'ARC-challenge', 'OpenBookQA', 'MathQA', 'mean_acc']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    log_file = "/home/zhangyingying/cyl/prune/EntroDrop/logs/llama3.1-8b/llama3.1_8b_shortgpt_l1_alpha_0.7.log"
    
    print("正在提取实验结果...")
    results = extract_results_from_log(log_file)
    
    if not results:
        print("未找到有效的实验结果")
        return
    
    print(f"\n找到 {len(results)} 个实验结果:\n")
    
    # 按num_prune排序
    results.sort(key=lambda x: x['num_prune'])
    
    # 打印表格
    print(format_results_table(results))
    
    # 保存为JSON文件
    json_output_file = "/home/zhangyingying/cyl/prune/EntroDrop/extracted_shortgpt_l1_alpha_0.7_results.json"
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存为CSV文件
    csv_output_file = "/home/zhangyingying/cyl/prune/EntroDrop/extracted_shortgpt_l1_alpha_0.7_results.csv"
    save_to_csv(results, csv_output_file)
    
    print(f"\n结果已保存到:")
    print(f"JSON格式: {json_output_file}")
    print(f"CSV格式: {csv_output_file}")
    
    # 打印简化版本
    print("\n简化版本:")
    print("Num Prune -> Mean Accuracy (%)")
    for result in results:
        mean_val = result['mean_acc'] if result['mean_acc'] != 'N/A' else 'N/A'
        print(f"{result['num_prune']:2d} -> {mean_val}")

if __name__ == "__main__":
    main()