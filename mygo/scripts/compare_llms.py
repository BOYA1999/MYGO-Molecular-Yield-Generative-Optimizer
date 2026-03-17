"""
Compare different LLMs for molecular generation guidance

This script runs the same generation task with three different LLMs
and compares their performance and guidance quality.
"""

import os
import sys
sys.path.append('.')
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from easydict import EasyDict

from scripts.sample_llm import main as sample_main, create_llm_agent
from llm_agents import PocketAnalyzer, GenerationAdvisor, MoleculeEvaluator


def run_generation_with_llm(config_path: str, llm_type: str, output_dir: str, **kwargs):
    """
    Run molecular generation with a specific LLM.
    
    Args:
        config_path: Path to config file
        llm_type: Type of LLM ('gpt4', 'claude', 'deepseek')
        output_dir: Output directory
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Running generation with {llm_type.upper()}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Create output directory for this LLM
    llm_output_dir = os.path.join(output_dir, llm_type)
    os.makedirs(llm_output_dir, exist_ok=True)
    
    # Import and run sample_llm.main with modified args
    import sys
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = [
            'compare_llms.py',
            '--config', config_path,
            '--outdir', llm_output_dir,
            '--llm_type', llm_type,
            '--use_llm',
            '--guidance_frequency', str(kwargs.get('guidance_frequency', 20)),
        ]
        
        # Run generation (this is a simplified version - actual implementation
        # would need to properly call the main function or refactor it)
        from scripts.sample_llm import main
        main()
        
        elapsed_time = time.time() - start_time
        
        # Load results
        results_file = os.path.join(llm_output_dir, 'llm_guidance_summary.json')
        guidance_summary = {}
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                guidance_summary = json.load(f)
        
        # Count generated molecules
        sdf_dir = os.path.join(llm_output_dir, 'SDF')
        num_molecules = 0
        if os.path.exists(sdf_dir):
            num_molecules = len([f for f in os.listdir(sdf_dir) if f.endswith('.sdf')])
        
        return {
            'llm_type': llm_type,
            'success': True,
            'elapsed_time': elapsed_time,
            'num_molecules': num_molecules,
            'guidance_summary': guidance_summary,
            'output_dir': llm_output_dir,
        }
        
    except Exception as e:
        print(f"Error running generation with {llm_type}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'llm_type': llm_type,
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time,
        }
    finally:
        sys.argv = original_argv


def compare_llm_agents_directly(pocket_pdb_path: str, test_smiles: str):
    """
    Directly compare LLM agents on the same tasks.
    
    Args:
        pocket_pdb_path: Path to pocket PDB file
        test_smiles: Test SMILES string for evaluation
        
    Returns:
        Dictionary with comparison results
    """
    llm_types = ['gpt4', 'claude', 'deepseek']
    results = {}
    
    for llm_type in llm_types:
        print(f"\n{'='*80}")
        print(f"Testing {llm_type.upper()}")
        print(f"{'='*80}\n")
        
        try:
            # Create LLM agent
            llm_agent = create_llm_agent(llm_type=llm_type)
            
            # Create analyzers
            pocket_analyzer = PocketAnalyzer(llm_agent)
            generation_advisor = GenerationAdvisor(llm_agent)
            molecule_evaluator = MoleculeEvaluator(llm_agent)
            
            # Test pocket analysis
            print(f"Testing pocket analysis with {llm_type}...")
            with open(pocket_pdb_path, 'r') as f:
                pocket_content = f.read()
            pocket_result = pocket_analyzer.analyze_pocket(pocket_content)
            
            # Test molecule evaluation
            print(f"Testing molecule evaluation with {llm_type}...")
            eval_result = molecule_evaluator.evaluate_molecule(test_smiles)
            
            # Get LLM stats
            llm_stats = llm_agent.get_stats()
            
            results[llm_type] = {
                'success': True,
                'pocket_analysis': pocket_result.get('success', False),
                'molecule_evaluation': eval_result.get('success', False),
                'llm_stats': llm_stats,
                'pocket_guidance_keys': list(pocket_result.get('guidance', {}).keys()) if pocket_result.get('success') else [],
                'eval_drug_score': eval_result.get('drug_likeness_score', 0) if eval_result.get('success') else 0,
            }
            
        except Exception as e:
            print(f"Error testing {llm_type}: {e}")
            results[llm_type] = {
                'success': False,
                'error': str(e),
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare different LLMs for molecular generation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--outdir', type=str, default='./outputs_llm_comparison', help='Output directory')
    parser.add_argument('--compare_mode', type=str, default='direct', choices=['direct', 'generation'],
                       help='Comparison mode: direct (test LLM agents) or generation (run full generation)')
    parser.add_argument('--pocket_pdb', type=str, help='Path to pocket PDB file (for direct comparison)')
    parser.add_argument('--test_smiles', type=str, default='CCO', help='Test SMILES string (for direct comparison)')
    args = parser.parse_args()
    
    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("LLM Comparison for Molecular Yield Generative Optimizer (MYGO) Molecular Generation")
    print("="*80)
    
    if args.compare_mode == 'direct':
        # Direct comparison of LLM agents
        if not args.pocket_pdb:
            print("Error: --pocket_pdb required for direct comparison mode")
            return
        
        results = compare_llm_agents_directly(args.pocket_pdb, args.test_smiles)
        
        # Save results
        results_file = os.path.join(output_dir, 'llm_comparison_direct.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("Comparison Summary")
        print("="*80)
        
        summary_data = []
        for llm_type, result in results.items():
            if result.get('success'):
                summary_data.append({
                    'LLM': llm_type.upper(),
                    'Pocket Analysis': 'OK' if result.get('pocket_analysis') else 'FAIL',
                    'Molecule Evaluation': 'OK' if result.get('molecule_evaluation') else 'FAIL',
                    'Drug-likeness Score': f"{result.get('eval_drug_score', 0):.2f}",
                    'API Calls': result.get('llm_stats', {}).get('call_count', 0),
                })
            else:
                summary_data.append({
                    'LLM': llm_type.upper(),
                    'Pocket Analysis': 'FAIL',
                    'Molecule Evaluation': 'FAIL',
                    'Drug-likeness Score': 'N/A',
                    'API Calls': 0,
                    'Error': result.get('error', 'Unknown error'),
                })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Save as CSV
        csv_file = os.path.join(output_dir, 'llm_comparison_direct.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {results_file}")
        print(f"Summary saved to: {csv_file}")
        
    else:
        # Full generation comparison
        llm_types = ['gpt4', 'claude', 'deepseek']
        all_results = {}
        
        for llm_type in llm_types:
            result = run_generation_with_llm(
                config_path=args.config,
                llm_type=llm_type,
                output_dir=output_dir,
                guidance_frequency=20
            )
            all_results[llm_type] = result
        
        # Save comparison results
        results_file = os.path.join(output_dir, 'llm_comparison_generation.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("Generation Comparison Summary")
        print("="*80)
        
        summary_data = []
        for llm_type, result in all_results.items():
            if result.get('success'):
                summary_data.append({
                    'LLM': llm_type.upper(),
                    'Success': 'OK',
                    'Time (s)': f"{result.get('elapsed_time', 0):.1f}",
                    'Molecules': result.get('num_molecules', 0),
                    'Output Dir': result.get('output_dir', 'N/A'),
                })
            else:
                summary_data.append({
                    'LLM': llm_type.upper(),
                    'Success': 'FAIL',
                    'Time (s)': f"{result.get('elapsed_time', 0):.1f}",
                    'Error': result.get('error', 'Unknown error'),
                })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Save as CSV
        csv_file = os.path.join(output_dir, 'llm_comparison_generation.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {results_file}")
        print(f"Summary saved to: {csv_file}")


if __name__ == '__main__':
    main()

