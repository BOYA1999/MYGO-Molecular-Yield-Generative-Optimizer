"""
LLM-guided Molecular Generation Script

Extends the base sample.py script with LLM guidance capabilities.
"""

import os
import sys
sys.path.append('.')
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
from easydict import EasyDict

from scripts.train import DataModule
from models.maskfill import PMAsymDenoiser
from models.llm_guided_maskfill import LLMGuidedPMAsymDenoiser
from models.sample import sample_loop3, seperate_outputs2
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.dataset import UseDataset
from utils.sample_noise import get_sample_noiser
from process.utils_process import extract_pocket, get_input_from_file, make_dummy_mol_with_coordinate

# Import LLM agents
from llm_agents import GPT4Agent, ClaudeAgent, DeepSeekAgent
from llm_agents import PocketAnalyzer, GenerationAdvisor, MoleculeEvaluator


def create_llm_agent(llm_type: str = "gpt4", **kwargs):
    """
    Create an LLM agent based on type.
    
    Args:
        llm_type: Type of LLM ('gpt4', 'claude', 'deepseek')
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        LLM agent instance
    """
    if llm_type.lower() == "gpt4":
        return GPT4Agent(**kwargs)
    elif llm_type.lower() == "claude":
        return ClaudeAgent(**kwargs)
    elif llm_type.lower() == "deepseek":
        return DeepSeekAgent(**kwargs)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def main():
    parser = argparse.ArgumentParser(description='LLM-guided Molecular Yield Generative Optimizer (MYGO) Molecular Generation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--outdir', type=str, default='./outputs_llm', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto/cpu/cuda:0')
    parser.add_argument('--llm_type', type=str, default='gpt4', choices=['gpt4', 'claude', 'deepseek'],
                       help='Type of LLM to use')
    parser.add_argument('--use_llm', action='store_true', help='Enable LLM guidance')
    parser.add_argument('--guidance_frequency', type=int, default=20,
                       help='Frequency of intermediate LLM evaluation (every N steps)')
    args = parser.parse_args()
    
    print("Note: `scripts/sample_llm.py` is experimental.")
    print("It demonstrates an LLM guidance interface and may require adaptation to your training config.")
    
    # Device detection
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda:0'
            print(f"Using GPU: {args.device}")
        else:
            args.device = 'cpu'
            print("No CUDA GPU detected, using CPU")
    
    # Load config
    config = make_config(args.config)
    config_name = os.path.basename(args.config).replace('.yml', '').replace('.yaml', '')
    seed = config.sample.seed + np.sum([ord(s) for s in args.outdir] + [ord(s) for s in args.config])
    seed_all(seed)
    config.sample.complete_seed = seed.item()
    
    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    cfg_dir = os.path.dirname(config.model.checkpoint).replace('checkpoints', 'train_config')
    
    if os.path.exists(cfg_dir):
        train_config_files = os.listdir(cfg_dir)
        if train_config_files:
            train_config = make_config(os.path.join(cfg_dir, train_config_files[0]))
        else:
            raise FileNotFoundError(f'train_config directory exists but is empty: {cfg_dir}')
    elif 'hyper_parameters' in ckpt and 'config' in ckpt['hyper_parameters']:
        train_config = ckpt['hyper_parameters']['config']
    else:
        raise FileNotFoundError('Cannot find training config')
    
    # Create log directory
    log_root = args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = get_logger('sample_llm', log_dir)
    logger.info(f'Loading from checkpoint: {config.model.checkpoint}')
    logger.info(args)
    logger.info(config)
    
    save_config(config, os.path.join(log_dir, os.path.basename(args.config)))
    
    # Create output directories
    sdf_dir = os.path.join(log_dir, 'SDF')
    os.makedirs(sdf_dir, exist_ok=True)
    
    # Load transforms
    logger.info('Loading transforms...')
    for samp_trans in config.get('transforms', {}).keys():
        if samp_trans in train_config.transforms.keys():
            train_config.transforms.get(samp_trans).update(config.transforms.get(samp_trans))
    
    # Align with scripts/sample.py style (featurizers + task transform)
    dm = DataModule(train_config)
    featurizer_list = dm.get_featurizers()
    featurizer = featurizer_list[-1]
    in_dims = dm.get_in_dims()
    task_trans = get_transforms(config.task.transform, mode='use')
    transforms = Compose(featurizer_list + [task_trans])
    
    # Load data
    logger.info('Loading data...')
    protein_path = config.data.protein_path
    reference_ligand = config.data.get('reference_ligand', None)
    
    pocmol_data, pocket_pdb, _ = get_input_data(
        protein_path=protein_path,
        input_ligand=reference_ligand,
        is_pep=config.data.get('is_pep', False),
        pocket_args=config.data.get('pocket_args', {}),
        pocmol_args=config.data.get('pocmol_args', {})
    )
    
    dataset = UseDataset(pocmol_data, n=config.sample.num_mols, task=config.task.name, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=config.sample.batch_size, shuffle=False)
    
    # Initialize model
    logger.info('Initializing model...')
    data_sample = next(iter(dataloader))
    num_node_types = in_dims['num_node_types']
    num_edge_types = in_dims['num_edge_types']
    pocket_in_dim = in_dims.get('pocket_in_dim', data_sample['pocket_atom_feature'].shape[-1])
    
    base_model = PMAsymDenoiser(
        config=train_config.model,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        pocket_in_dim=pocket_in_dim
    )
    # Compatible with Lightning and native PyTorch checkpoints
    if 'state_dict' in ckpt:
        state_dict = {k[6:]: value for k, value in ckpt['state_dict'].items() if k.startswith('model.')}
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        raise KeyError('Checkpoint missing model weights. Expected "state_dict" or "model_state_dict".')
    base_model.load_state_dict(state_dict)
    base_model.to(args.device)
    base_model.eval()
    
    # Create LLM-guided model
    if args.use_llm:
        logger.info(f'Initializing LLM agent: {args.llm_type}...')
        try:
            llm_agent = create_llm_agent(llm_type=args.llm_type)
            pocket_analyzer = PocketAnalyzer(llm_agent)
            generation_advisor = GenerationAdvisor(llm_agent)
            molecule_evaluator = MoleculeEvaluator(llm_agent)
            
            model = LLMGuidedPMAsymDenoiser(
                base_model=base_model,
                pocket_analyzer=pocket_analyzer,
                generation_advisor=generation_advisor,
                molecule_evaluator=molecule_evaluator,
                use_llm_guidance=True,
                guidance_frequency=args.guidance_frequency
            )
            
            # Analyze pocket for initialization guidance
            logger.info('Analyzing protein pocket with LLM...')
            pocket_analysis = model.analyze_pocket(pocket_pdb)
            if pocket_analysis and pocket_analysis.get("success"):
                logger.info("Pocket analysis completed")
                logger.info(f"Guidance: {pocket_analysis.get('guidance', {})}")
            else:
                logger.warning("Pocket analysis failed, continuing without LLM guidance")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.info("Continuing without LLM guidance...")
            model = LLMGuidedPMAsymDenoiser(
                base_model=base_model,
                use_llm_guidance=False
            )
    else:
        logger.info('LLM guidance disabled')
        model = LLMGuidedPMAsymDenoiser(
            base_model=base_model,
            use_llm_guidance=False
        )
    
    model.eval()
    
    # Get noise scheduler
    logger.info('Initializing noise scheduler...')
    noiser = get_sample_noiser(
        config.noise,
        num_node_types,
        num_edge_types,
        mode='sample',
        device=args.device,
        ref_config=train_config.noise
    )
    
    # Generate molecules
    logger.info(f'Generating {config.sample.num_mols} molecules...')
    all_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(args.device)
            batch, outputs, trajs = sample_loop3(
                batch, model.base_model, noiser, args.device, is_ar='', off_tqdm=False
            )
            
            generated_list, _, _ = seperate_outputs2(batch, outputs, trajs, off_tqdm=True)
            for i_mol in range(len(generated_list)):
                try:
                    mol_info = featurizer.decode_output(**generated_list[i_mol])
                    mol_info.update({
                        'task': config.task.name,
                        'db': 'use',
                        'data_id': f'use_{batch_idx}_{i_mol}',
                        'key': ''
                    })
                    mol = reconstruct_from_generated_with_edges(mol_info)
                    if mol is None:
                        continue
                    smiles = Chem.MolToSmiles(mol)
                    if (not smiles) or ('.' in smiles):
                        continue
                    
                    # Evaluate with LLM
                    if args.use_llm and model.molecule_evaluator:
                        evaluation = model.evaluate_molecule(smiles)
                        if evaluation and evaluation.get("success"):
                            logger.info(f"LLM evaluation completed for {smiles}")
                    
                    # Save molecule
                    mol_name = f"mol_{len(all_results)+1:04d}"
                    sdf_path = os.path.join(sdf_dir, f"{mol_name}.sdf")
                    writer = Chem.SDWriter(sdf_path)
                    writer.write(mol)
                    writer.close()
                    
                    all_results.append({
                        "mol_name": mol_name,
                        "smiles": smiles,
                        "sdf_path": sdf_path,
                    })
                except Exception as e:
                    logger.warning(f"Error processing molecule: {e}")
                    continue
    
    logger.info(f"Generated {len(all_results)} molecules")
    logger.info(f"Output directory: {log_dir}")
    
    # Save guidance summary if LLM was used
    if args.use_llm:
        guidance_summary = model.get_guidance_summary()
        import json
        with open(os.path.join(log_dir, 'llm_guidance_summary.json'), 'w') as f:
            json.dump(guidance_summary, f, indent=2, default=str)
        logger.info("Saved LLM guidance summary")


if __name__ == '__main__':
    main()

