#!/usr/bin/env python3
"""
处理 data/data_mine 中的蛋白-分子复合物数据
将其转换为 PocketXMol 训练所需的格式
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.PDB import PDBParser, PDBIO, Select
import argparse
import shutil
import glob

class LigandSelector(Select):
    """选择配体原子的类"""
    def accept_residue(self, residue):
        # 排除标准氨基酸和水分子
        standard_residues = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HOH', 'WAT'
        ]
        return residue.get_resname() not in standard_residues

class ProteinSelector(Select):
    """选择蛋白质原子的类"""
    def accept_residue(self, residue):
        # 只选择标准氨基酸
        standard_residues = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]
        return residue.get_resname() in standard_residues

def extract_ligand_and_protein(pdb_file, output_protein_file, output_ligand_file):
    """从PDB文件中分离蛋白质和配体"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    io = PDBIO()
    
    # 保存蛋白质部分
    io.set_structure(structure)
    io.save(output_protein_file, ProteinSelector())
    
    # 保存配体部分
    io.set_structure(structure)
    io.save(output_ligand_file, LigandSelector())
    
    return True

def pdb_to_smiles(ligand_pdb_file):
    """将配体PDB文件转换为SMILES"""
    try:
        mol = Chem.MolFromPDBFile(ligand_pdb_file, removeHs=True)
        if mol is None:
            return None
        
        # 尝试清理分子结构
        mol = Chem.RemoveHs(mol)
        if mol is None:
            return None
            
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        return smiles if smiles else None
    except Exception as e:
        print(f"Error converting {ligand_pdb_file} to SMILES: {e}")
        return None

def ligand_pdb_to_sdf(ligand_pdb_file, output_sdf_file):
    """将配体PDB文件转换为SDF格式"""
    try:
        mol = Chem.MolFromPDBFile(ligand_pdb_file, removeHs=True)
        if mol is None:
            return False
        
        mol = Chem.RemoveHs(mol)
        if mol is None:
            return False
        
        # 写入SDF文件
        writer = Chem.SDWriter(output_sdf_file)
        writer.write(mol)
        writer.close()
        return True
    except Exception as e:
        print(f"Error converting {ligand_pdb_file} to SDF: {e}")
        return False

def process_data_mine(input_dir, output_dir):
    """处理data_mine中的所有PDB文件"""
    
    # 创建输出目录结构
    os.makedirs(f"{output_dir}/dfs", exist_ok=True)
    os.makedirs(f"{output_dir}/files/proteins", exist_ok=True)
    os.makedirs(f"{output_dir}/files/mols", exist_ok=True)
    
    # 创建临时目录存放中间文件
    temp_dir = f"{output_dir}/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 获取所有PDB文件
    pdb_files = glob.glob(f"{input_dir}/*.pdb")
    
    print(f"Found {len(pdb_files)} PDB files to process")
    
    meta_data = []
    successful_count = 0
    failed_count = 0
    
    # 创建详细的进度条
    pbar = tqdm(pdb_files, desc="Processing PDB files", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, pdb_file in enumerate(pbar):
        try:
            # 提取data_id
            filename = os.path.basename(pdb_file)
            data_id = filename.replace('.pdb', '')
            
            # 更新进度条描述
            pbar.set_description(f"Processing {filename[:30]:<30} [✓{successful_count} ✗{failed_count}]")
            
            # 临时文件路径
            temp_protein_file = f"{temp_dir}/{data_id}_protein.pdb"
            temp_ligand_file = f"{temp_dir}/{data_id}_ligand.pdb"
            
            # 分离蛋白质和配体
            success = extract_ligand_and_protein(pdb_file, temp_protein_file, temp_ligand_file)
            if not success:
                failed_count += 1
                pbar.set_description(f"Failed {filename[:30]:<30} [✓{successful_count} ✗{failed_count}]")
                continue
            
            # 检查配体文件是否有内容
            if not os.path.exists(temp_ligand_file) or os.path.getsize(temp_ligand_file) < 100:
                failed_count += 1
                pbar.set_description(f"No ligand {filename[:30]:<30} [✓{successful_count} ✗{failed_count}]")
                continue
            
            # 获取配体的SMILES
            smiles = pdb_to_smiles(temp_ligand_file)
            if not smiles:
                failed_count += 1
                pbar.set_description(f"No SMILES {filename[:30]:<30} [✓{successful_count} ✗{failed_count}]")
                continue
            
            # 转换配体为SDF格式
            output_sdf_file = f"{output_dir}/files/mols/{data_id}_mol.sdf"
            sdf_success = ligand_pdb_to_sdf(temp_ligand_file, output_sdf_file)
            if not sdf_success:
                failed_count += 1
                pbar.set_description(f"SDF failed {filename[:30]:<30} [✓{successful_count} ✗{failed_count}]")
                continue
            
            # 复制蛋白质文件
            output_protein_file = f"{output_dir}/files/proteins/{data_id}_pro.pdb"
            shutil.copy2(temp_protein_file, output_protein_file)
            
            # 添加到元数据
            meta_data.append({
                'data_id': data_id,
                'pdbid': '4JPS',  # 所有都来自4JPS
                'smiles': smiles
            })
            
            successful_count += 1
            pbar.set_description(f"Success {filename[:30]:<30} [✓{successful_count} ✗{failed_count}]")
            
            # 清理临时文件
            if os.path.exists(temp_protein_file):
                os.remove(temp_protein_file)
            if os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
                
        except Exception as e:
            failed_count += 1
            filename = os.path.basename(pdb_file)
            pbar.set_description(f"Error {filename[:30]:<30} [✓{successful_count} ✗{failed_count}]")
            print(f"\nError processing {pdb_file}: {e}")
            continue
    
    # 完成时更新最终状态
    pbar.set_description(f"Completed! [✓{successful_count} ✗{failed_count}]")
    pbar.close()
    
    # 清理临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print(f"Processing completed:")
    print(f"  Successful: {successful_count}")
    print(f"  Failed: {failed_count}")
    
    if meta_data:
        # 保存元数据文件
        df = pd.DataFrame(meta_data)
        df.to_csv(f"{output_dir}/dfs/meta_filter_w_pocket.csv", index=False)
        print(f"Saved metadata for {len(meta_data)} samples to meta_filter_w_pocket.csv")
        
        # 显示一些统计信息
        print(f"Unique SMILES: {df['smiles'].nunique()}")
        print(f"Sample data:")
        print(df.head())
        
        return df
    else:
        print("No valid data processed!")
        return None

def main():
    parser = argparse.ArgumentParser(description='Process data_mine PDB files for PocketXMol training')
    parser.add_argument('--input_dir', type=str, default='data/data_mine',
                        help='Input directory containing PDB files')
    parser.add_argument('--output_dir', type=str, default='data/data_train/data_mine',
                        help='Output directory for processed data')
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # 处理数据
    result_df = process_data_mine(args.input_dir, args.output_dir)
    
    if result_df is not None:
        print("\n" + "="*50)
        print("数据处理完成！接下来的步骤:")
        print("1. 激活conda环境: conda activate pyg")
        print("2. 提取蛋白质口袋:")
        print(f"   python process/extract_pockets.py --db_name data_mine \\")
        print(f"   --path_df {args.output_dir}/dfs/meta_filter_w_pocket.csv \\")
        print(f"   --root {args.output_dir}/files")
        print("3. 处理蛋白质-分子复合物:")
        print("   python process/process_pocmol.py --db_name data_mine")
        print("4. 处理扭转角信息:")
        print("   python process/process_torsional_info.py --db_name data_mine")
        print("5. 处理分解信息:")
        print("   python process/process_decompose_info.py --db_name data_mine")
        print("="*50)

if __name__ == "__main__":
    main() 