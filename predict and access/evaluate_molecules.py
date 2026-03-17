"""
分子ADMET评估主脚本

使用机器学习/深度学习方法对生成的分子进行评估，包括：
1. 代谢预测
2. 血浆暴露
3. 血脑屏障渗透性
4. 器官毒性预测
5. 致畸致癌致突变预测
6. 半衰期预测

生成分段评估报告。
"""

import os
import sys
import argparse
from typing import List, Union, Optional
from rdkit import Chem

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ensemble_predictor import EnsembleADMETPredictor
from segmented_report_generator import SegmentedReportGenerator


def load_molecules_from_file(file_path: str) -> List[Chem.Mol]:
    """
    Load molecules from file.
    
    Supports:
    - SMILES file (.smi, .txt)
    - SDF file (.sdf)
    - MOL file (.mol)
    """
    molecules = []
    
    if file_path.endswith('.smi') or file_path.endswith('.txt'):
        # SMILES file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Assume first column is SMILES
                    smiles = line.split()[0]
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        molecules.append(mol)
    
    elif file_path.endswith('.sdf'):
        # SDF file
        supplier = Chem.SDMolSupplier(file_path)
        for mol in supplier:
            if mol is not None:
                molecules.append(mol)
    
    elif file_path.endswith('.mol'):
        # MOL file
        mol = Chem.MolFromMolFile(file_path)
        if mol is not None:
            molecules.append(mol)
    
    return molecules


def evaluate_single_molecule(
    mol: Union[Chem.Mol, str],
    output_dir: str,
    model_dir: Optional[str] = None,
    use_ml: bool = True,
    mol_name: Optional[str] = None
) -> dict:
    """
    Evaluate a single molecule and generate reports.
    
    Args:
        mol: RDKit molecule object or SMILES string
        output_dir: Output directory for reports
        model_dir: Directory containing pre-trained models
        use_ml: Whether to use ML models
        mol_name: Optional molecule name/identifier
        
    Returns:
        Dictionary with report paths
    """
    print(f"\n{'='*60}")
    print(f"开始评估分子: {mol_name or 'Unknown'}")
    print(f"{'='*60}\n")
    
    # Initialize ensemble predictor
    print("初始化ADMET预测器...")
    ensemble_predictor = EnsembleADMETPredictor(
        model_dir=model_dir,
        use_ml=use_ml
    )
    
    # Initialize report generator
    print("初始化报告生成器...")
    report_generator = SegmentedReportGenerator(ensemble_predictor)
    
    # Generate reports
    print("生成评估报告...")
    report_paths = report_generator.generate_segmented_report(
        mol=mol,
        output_dir=output_dir,
        mol_name=mol_name
    )
    
    print("\n评估完成。")
    print(f"\n生成的报告文件：")
    print(f"  - 完整Markdown报告: {report_paths.get('markdown_full', 'N/A')}")
    print(f"  - 分段Markdown报告: {len(report_paths.get('markdown_sections', []))} 个部分")
    print(f"  - JSON报告: {report_paths.get('json', 'N/A')}")
    print(f"  - CSV报告: {report_paths.get('csv', 'N/A')}")
    print(f"  - HTML报告: {report_paths.get('html', 'N/A')}")
    
    return report_paths


def evaluate_batch_molecules(
    mols: List[Union[Chem.Mol, str]],
    output_dir: str,
    model_dir: Optional[str] = None,
    use_ml: bool = True,
    mol_names: Optional[List[str]] = None
) -> List[dict]:
    """
    Evaluate a batch of molecules and generate reports.
    
    Args:
        mols: List of RDKit molecule objects or SMILES strings
        output_dir: Output directory for reports
        model_dir: Directory containing pre-trained models
        use_ml: Whether to use ML models
        mol_names: Optional list of molecule names
        
    Returns:
        List of dictionaries with report paths
    """
    print(f"\n{'='*60}")
    print(f"开始批量评估 {len(mols)} 个分子")
    print(f"{'='*60}\n")
    
    # Initialize ensemble predictor
    print("初始化ADMET预测器...")
    ensemble_predictor = EnsembleADMETPredictor(
        model_dir=model_dir,
        use_ml=use_ml
    )
    
    all_report_paths = []
    
    for i, mol in enumerate(mols):
        mol_name = mol_names[i] if mol_names and i < len(mol_names) else f"mol_{i+1:04d}"
        
        print(f"\n处理分子 {i+1}/{len(mols)}: {mol_name}")
        
        # Create subdirectory for this molecule
        mol_output_dir = os.path.join(output_dir, mol_name)
        os.makedirs(mol_output_dir, exist_ok=True)
        
        # Initialize report generator
        report_generator = SegmentedReportGenerator(ensemble_predictor)
        
        # Generate reports
        report_paths = report_generator.generate_segmented_report(
            mol=mol,
            output_dir=mol_output_dir,
            mol_name=mol_name
        )
        
        all_report_paths.append(report_paths)
        
        print("  完成")
    
    print(f"\n{'='*60}")
    print(f"批量评估完成！共处理 {len(mols)} 个分子")
    print(f"{'='*60}\n")
    
    return all_report_paths


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='分子ADMET评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估单个SMILES字符串
  python evaluate_molecules.py -s "CCO" -o ./reports
  
  # 评估SMILES文件中的多个分子
  python evaluate_molecules.py -f molecules.smi -o ./reports
  
  # 评估SDF文件
  python evaluate_molecules.py -f molecules.sdf -o ./reports --model-dir ./models
  
  # 使用规则基础方法（不使用ML模型）
  python evaluate_molecules.py -s "CCO" -o ./reports --no-ml
        """
    )
    
    parser.add_argument(
        '-s', '--smiles',
        type=str,
        help='SMILES字符串（单个分子）'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='包含分子的文件路径（.smi, .sdf, .mol）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='预训练模型目录（可选）'
    )
    
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='不使用机器学习模型，使用规则基础方法'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='分子名称（仅用于单个分子评估）'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.smiles and not args.file:
        parser.error("必须提供 -s/--smiles 或 -f/--file 参数")
    
    if args.smiles and args.file:
        parser.error("不能同时提供 -s/--smiles 和 -f/--file 参数")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine use_ml
    use_ml = not args.no_ml
    
    # Evaluate molecules
    if args.smiles:
        # Single molecule
        evaluate_single_molecule(
            mol=args.smiles,
            output_dir=args.output,
            model_dir=args.model_dir,
            use_ml=use_ml,
            mol_name=args.name
        )
    
    elif args.file:
        # Batch molecules
        if not os.path.exists(args.file):
            print(f"错误: 文件不存在: {args.file}")
            sys.exit(1)
        
        molecules = load_molecules_from_file(args.file)
        
        if not molecules:
            print(f"错误: 未能从文件中加载任何分子: {args.file}")
            sys.exit(1)
        
        print(f"成功加载 {len(molecules)} 个分子")
        
        evaluate_batch_molecules(
            mols=molecules,
            output_dir=args.output,
            model_dir=args.model_dir,
            use_ml=use_ml
        )


if __name__ == '__main__':
    main()

