#!/usr/bin/env python3
"""
一键设置 data_mine 数据训练环境
"""

import os
import subprocess
import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path


def run_command(cmd, check=True, show_output=True):
    """执行shell命令"""
    print(f"执行命令: {cmd}")
    if show_output:
        # 实时显示输出
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        result = subprocess.CompletedProcess(cmd, process.returncode)
        if check and result.returncode != 0:
            print(f"命令执行失败: {cmd}")
            sys.exit(1)
    else:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"命令执行失败: {cmd}")
            print(f"错误信息: {result.stderr}")
            sys.exit(1)
    return result





def process_data_mine(skip_data_prep=False):
    """处理data_mine数据的完整流程"""
    
    print("="*60)
    print("开始处理 data_mine 数据")
    print("="*60)
    
    # 步骤1: 数据预处理
    if not skip_data_prep:
        print("\n" + "="*60)
        print("步骤1: 数据预处理 - 分离蛋白质和配体")
        print("="*60)
        cmd = "python process/process_data_mine.py"
        run_command(cmd)
    else:
        print("\n" + "="*60)
        print("步骤1: 跳过数据预处理")
        print("="*60)
    
    # 检查数据预处理结果
    meta_file = "data/data_train/data_mine/dfs/meta_filter_w_pocket.csv"
    if not os.path.exists(meta_file):
        print(f"错误: 找不到元数据文件 {meta_file}")
        print("请先运行数据预处理步骤")
        sys.exit(1)
    
    df = pd.read_csv(meta_file)
    print(f"数据预处理完成，共有 {len(df)} 个样本")
    
    # 步骤2: 提取蛋白质口袋
    print("\n" + "="*60)
    print("步骤2: 提取蛋白质口袋")
    print("="*60)
    cmd = ("python process/extract_pockets.py --db_name data_mine "
           "--path_df data/data_train/data_mine/dfs/meta_filter_w_pocket.csv "
           "--root data/data_train/data_mine/files")
    run_command(cmd)
    
    # 步骤3: 处理蛋白质-分子复合物
    print("\n" + "="*60)
    print("步骤3: 处理蛋白质-分子复合物")
    print("="*60)
    cmd = "python process/process_pocmol.py --db_name data_mine"
    run_command(cmd)
    
    # 步骤4: 处理扭转角信息
    print("\n" + "="*60)
    print("步骤4: 处理扭转角信息") 
    print("="*60)
    cmd = "python process/process_torsional_info.py --db_name data_mine"
    run_command(cmd)
    
    # 步骤5: 处理分解信息
    print("\n" + "="*60)
    print("步骤5: 处理分解信息")
    print("="*60)
    cmd = "python process/process_decompose_info.py --db_name data_mine"
    run_command(cmd)
    
    print("\n="*60)
    print("data_mine 数据处理完成！")
    print("="*60)


def create_training_config():
    """使用现有的训练配置文件"""
    
    print("\n使用现有的训练配置文件...")
    
    # 使用现有的配置文件
    output_config_path = "configs/train/train_data_mine.yml"
    if os.path.exists(output_config_path):
        print(f"使用训练配置文件: {output_config_path}")
        return output_config_path
    
    print("配置文件不存在，请检查 configs/train/train_data_mine.yml 文件")
    sys.exit(1)


def update_assembly_config():
    """更新训练数据分割配置"""
    
    print("\n更新训练数据分割...")
    
    # 检查是否存在assemblies目录
    assemblies_dir = "data/data_train/assemblies"
    os.makedirs(assemblies_dir, exist_ok=True)
    
    # 创建简单的训练/验证分割
    meta_file = "data/data_train/data_mine/dfs/meta_filter_w_pocket.csv"
    df = pd.read_csv(meta_file)
    
    # 80/20 分割
    n_total = len(df)
    n_train = int(n_total * 0.8)
    
    data_ids = df['data_id'].tolist()
    
    # 创建分割数据
    split_data = []
    for i, data_id in enumerate(data_ids):
        split = 'train' if i < n_train else 'val'
        split_data.append({
            'data_id': data_id,
            'split': split,
            'db': 'data_mine',
            'task': 'sbdd'
        })
    
    split_df = pd.DataFrame(split_data)
    split_file = os.path.join(assemblies_dir, 'split_train_val.csv')
    split_df.to_csv(split_file, index=False)
    
    print(f"数据分割文件已创建: {split_file}")
    print(f"训练集: {n_train} 样本, 验证集: {n_total - n_train} 样本")
    
    # 生成assembly lmdb
    print("生成训练数据集合...")
    cmd = "python process/make_assembly_lmdb.py"
    run_command(cmd)


def print_next_steps(config_path):
    """打印后续步骤"""
    
    print("\n" + "="*60)
    print("数据准备完成！后续步骤:")
    print("="*60)
    print("\n1. 开始训练:")
    print(f"   python scripts/train_pl.py --config {config_path}")
    
    print("\n2. 监控训练进度:")
    print("   训练日志将保存在 lightning_logs/ 目录中")
    print("   可以使用 tensorboard 查看训练曲线:")
    print("   tensorboard --logdir lightning_logs")
    
    print("\n3. 测试模型:")
    print("   训练完成后，模型权重将保存在 lightning_logs/*/checkpoints/ 中")
    print("   可以使用 scripts/sample_use.py 进行推理测试")
    
    print("\n4. 数据统计:")
    meta_file = "data/data_train/data_mine/dfs/meta_filter_w_pocket.csv"
    if os.path.exists(meta_file):
        df = pd.read_csv(meta_file)
        print(f"   总样本数: {len(df)}")
        print(f"   独特配体数: {df['smiles'].nunique()}")
        print(f"   蛋白质来源: {df['pdbid'].unique()}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='一键设置data_mine数据训练环境')
    parser.add_argument('--skip_data_prep', action='store_true',
                       help='跳过数据预处理步骤（如果已经运行过）')

    args = parser.parse_args()
    
    try:
        # 处理数据
        process_data_mine(args.skip_data_prep)
        
        # 创建训练配置
        config_path = create_training_config()
        
        # 更新数据分割
        update_assembly_config()
        
        # 打印后续步骤
        print_next_steps(config_path)
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 