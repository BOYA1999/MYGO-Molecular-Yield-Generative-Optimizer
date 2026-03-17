"""
示例：如何使用ADMET评估系统

本示例演示如何使用评估系统对分子进行评估并生成分段报告。
"""

import os
from rdkit import Chem
from ensemble_predictor import EnsembleADMETPredictor
from segmented_report_generator import SegmentedReportGenerator


def example_single_molecule():
    """示例1: 评估单个分子"""
    print("=" * 60)
    print("示例1: 评估单个分子")
    print("=" * 60)
    
    # 初始化预测器（使用规则基础方法，不需要模型文件）
    ensemble_predictor = EnsembleADMETPredictor(
        model_dir=None,
        use_ml=False  # 使用规则基础方法
    )
    
    # 初始化报告生成器
    report_generator = SegmentedReportGenerator(ensemble_predictor)
    
    # 准备分子（示例：乙醇）
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print(f"错误: 无法解析SMILES: {smiles}")
        return
    
    # 创建输出目录
    output_dir = "./example_reports/single_molecule"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告
    print(f"\n正在评估分子: {smiles}")
    report_paths = report_generator.generate_segmented_report(
        mol=mol,
        output_dir=output_dir,
        mol_name="ethanol"
    )
    
    print("\n评估完成。")
    print(f"\n生成的报告文件：")
    print(f"  - 完整Markdown报告: {report_paths.get('markdown_full', 'N/A')}")
    print(f"  - 分段Markdown报告: {len(report_paths.get('markdown_sections', []))} 个部分")
    for i, section_path in enumerate(report_paths.get('markdown_sections', []), 1):
        print(f"    Section {i}: {os.path.basename(section_path)}")
    print(f"  - JSON报告: {report_paths.get('json', 'N/A')}")
    print(f"  - CSV报告: {report_paths.get('csv', 'N/A')}")
    print(f"  - HTML报告: {report_paths.get('html', 'N/A')}")


def example_batch_molecules():
    """示例2: 批量评估多个分子"""
    print("\n" + "=" * 60)
    print("示例2: 批量评估多个分子")
    print("=" * 60)
    
    # 初始化预测器
    ensemble_predictor = EnsembleADMETPredictor(
        model_dir=None,
        use_ml=False
    )
    
    # 准备多个分子
    smiles_list = [
        "CCO",           # 乙醇
        "CCN(CC)CC",     # 三乙胺
        "c1ccccc1",      # 苯
        "CC(=O)O",       # 乙酸
    ]
    
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
        else:
            print(f"警告: 无法解析SMILES: {smiles}")
    
    print(f"\n成功加载 {len(mols)} 个分子")
    
    # 创建输出目录
    output_dir = "./example_reports/batch_molecules"
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量评估
    report_generator = SegmentedReportGenerator(ensemble_predictor)
    
    for i, mol in enumerate(mols):
        mol_name = f"mol_{i+1:04d}"
        mol_output_dir = os.path.join(output_dir, mol_name)
        os.makedirs(mol_output_dir, exist_ok=True)
        
        smiles = Chem.MolToSmiles(mol)
        print(f"\n处理分子 {i+1}/{len(mols)}: {smiles}")
        
        report_paths = report_generator.generate_segmented_report(
            mol=mol,
            output_dir=mol_output_dir,
            mol_name=mol_name
        )
        
        print(f"  完成 - 报告保存在: {mol_output_dir}")
    
    print(f"\n批量评估完成。共处理 {len(mols)} 个分子")


def example_direct_prediction():
    """示例3: 直接使用预测器获取结果（不生成报告）"""
    print("\n" + "=" * 60)
    print("示例3: 直接使用预测器获取结果")
    print("=" * 60)
    
    # 初始化预测器
    ensemble_predictor = EnsembleADMETPredictor(
        model_dir=None,
        use_ml=False
    )
    
    # 准备分子
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    
    # 运行所有预测
    print(f"\n正在预测分子: {smiles}")
    results = ensemble_predictor.predict_all(mol)
    
    # 显示结果
    print(f"\n预测结果：")
    print(f"  - SMILES: {results.get('smiles', 'N/A')}")
    print(f"  - 综合ADMET评分: {results.get('overall_admet_score', 0.0):.3f}")
    print(f"  - 预测状态: {'成功' if results.get('success', False) else '失败'}")
    
    print(f"\n详细预测结果：")
    predictions = results.get('predictions', {})
    for pred_name, pred_result in predictions.items():
        if pred_result.get('success', False):
            pred_data = pred_result.get('prediction', {})
            if isinstance(pred_data, dict):
                print(f"\n  {pred_name}:")
                for key, value in list(pred_data.items())[:3]:  # 只显示前3个
                    print(f"    - {key}: {value}")


if __name__ == '__main__':
    print("ADMET评估系统使用示例")
    print("=" * 60)
    
    # 运行示例
    try:
        example_single_molecule()
        example_batch_molecules()
        example_direct_prediction()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        print("\n提示: 查看生成的报告文件了解详细评估结果")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

