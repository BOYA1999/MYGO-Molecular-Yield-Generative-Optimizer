import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import shutil
import sys
import argparse
import gc
import time
from pathlib import Path

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
torch.set_float32_matmul_precision('medium')

sys.path.append('.')
from models.maskfill import PMAsymDenoiser
from models.loss import get_loss_func
from utils.dataset import ForeverTaskDataset
from utils.transforms import FeaturizeMol, Compose, get_transforms, FeaturizePocket
from utils.misc import *
from utils.train import get_optimizer, get_scheduler, GradualWarmupScheduler
from utils.sample_noise import get_sample_noiser


def copy_py_files(src_dir, dst_dir, base=False):
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        # Get the absolute path of the item
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            if (not base) or (item in ['scripts', 'models', 'notebooks', 'utils', 'process', 'evaluate']):
                # If the item is a directory, recursively call the function on it
                copy_py_files(item_path, os.path.join(dst_dir, item))
        elif (item.endswith('.py') or item.endswith('.sh') or item.endswith('.ipynb')):
            # If the item is a file and ends with .py, copy it to the destination directory
            shutil.copy(item_path, dst_dir)


def get_grad_norm(model, norm_type=2):
    """Calculate gradient norms for the model parameters."""
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class DataModule:
    """Data module for loading and preparing datasets."""
    
    def __init__(self, config, global_rank=0, world_size=1):
        self.config = config
        self.global_rank = global_rank
        self.world_size = world_size
        
    def get_featurizers(self):
        featurizer = FeaturizeMol(self.config.transforms.featurizer)
        if 'featurizer_pocket' in self.config.transforms:
            feat_pocket = FeaturizePocket(self.config.transforms.featurizer_pocket)
            return [feat_pocket, featurizer]  # pocket first because mol need to substract pocket center
        else:
            return [featurizer]

    def get_in_dims(self, featurizers=None):
        if featurizers is None:
            featurizers = self.get_featurizers()
        num_node_types = featurizers[-1].num_node_types
        num_edge_types = featurizers[-1].num_edge_types
        in_dims = {
            'num_node_types': num_node_types,
            'num_edge_types': num_edge_types,
        }
        if len(featurizers) == 2:
            in_dims.update({
                'pocket_in_dim': featurizers[0].feature_dim,
            })
        return in_dims
        
    def setup(self):
        # # Transforms
        featurizers = self.get_featurizers()
        in_dims = self.get_in_dims(featurizers)
        task_trans = get_transforms(self.config.transforms.task, mode='train',
                                    num_node_types=in_dims['num_node_types'],)
        noiser = get_sample_noiser(self.config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                                   mode='train')
        transform_list = featurizers + [task_trans, noiser]
        if 'cut_peptide' in self.config.transforms:
            transform_list = [get_transforms(self.config.transforms.cut_peptide)] + transform_list
        self.transforms = Compose(transform_list)
        follow_batch = sum([getattr(t, 'follow_batch', []) for t in self.transforms.transforms], [])
        exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in self.transforms.transforms], [])

        # # Datasets and sampler
        data_cfg = self.config.data
        num_samplers_args = {'num_workers': self.config.train.num_workers,
                             'global_rank': self.global_rank,
                             'world_size': self.world_size}
        train_set = ForeverTaskDataset(data_cfg.dataset, data_cfg.task_db_weights,'train',
                                       transforms=self.transforms, shuffle=True, **num_samplers_args)
        if num_samplers_args['world_size'] > 100:
            divider = 4
        else:
            divider = 1
        num_samplers_args['num_workers'] = self.config.train.num_workers//divider
        val_set = ForeverTaskDataset(data_cfg.dataset, data_cfg.task_db_weights, 'val',
                                     transforms=self.transforms, shuffle=False, **num_samplers_args)

        # # Dataloaders
        train_cfg = self.config.train
        is_vscode = os.environ.get("TERM_PROGRAM") == "vscode"
        self.train_loader = DataLoader(train_set, batch_size=train_cfg.batch_size if not is_vscode else 40,
                                       num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory,
                                       follow_batch=follow_batch, exclude_keys=exclude_keys,
                                       persistent_workers=train_cfg.persistent_workers,
        )
        self.val_loader = DataLoader(val_set, batch_size=train_cfg.batch_size if not is_vscode else 40,
                                     num_workers=train_cfg.num_workers//divider, pin_memory=train_cfg.pin_memory,
                                     follow_batch=follow_batch, exclude_keys=exclude_keys,
                                     persistent_workers=train_cfg.persistent_workers,
        )


class MYGOModel(nn.Module):
    """Main model wrapper for training."""
    
    def __init__(self, config, num_node_types, num_edge_types, **kwargs):
        super(MYGOModel, self).__init__()
        self.config = config

        # Model
        if self.config.model.name == 'pm_asym_denoiser':
            self.model = PMAsymDenoiser(config=self.config.model,
                                  num_node_types=num_node_types,
                                  num_edge_types=num_edge_types, **kwargs)
        
        if getattr(self.config.model, 'pretrained', ''):
            ckpt = torch.load(self.config.model.pretrained, map_location='cpu')
            self.model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items()
                                        if k.startswith('model.')})
            print('Load pretrained model from', self.config.model.pretrained)

        self.loss_func = get_loss_func(self.config.loss)

    def forward(self, batch):
        return self.model(batch)
    
    def reduce_batch(self, batch):
        """Reduce batch size when OOM occurs."""
        # free memory
        for p in self.model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()

        # drop last 50 percent
        new_bs = int(len(batch) * 0.5)
        print(f"\nOut of memory error occurred in step. Reduce bs {len(batch)} to {new_bs}")
        device = batch.batch.device
        follow_batch = [k.replace('_batch','') for k in batch.keys() if k.endswith('_batch')]
        batch_cpu = batch.cpu()
        del batch
        torch.cuda.empty_cache()
        data_list = batch_cpu.to_data_list()
        del data_list[new_bs:]
        batch = Batch.from_data_list(data_list[:new_bs], follow_batch=follow_batch).to(device)
        return batch


class Trainer:
    """Training manager without PyTorch Lightning."""
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, 
                 config, device, log_dir, warmup_step=0, resume_ckpt=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.warmup_step = warmup_step
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.val_count = 0  # Track validation calls for scheduler frequency
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Checkpoint directory
        self.checkpoint_dir = Path(log_dir) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine scheduler type for proper stepping
        self.scheduler_needs_metric = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        # Get scheduler frequency (how often to call scheduler.step during validation)
        scheduler_params = config.train.scheduler.params
        self.scheduler_frequency = scheduler_params.get('frequency', 1)
        
        # Resume from checkpoint if provided
        if resume_ckpt:
            self.load_checkpoint(resume_ckpt)
    
    def save_checkpoint(self, filename='last.ckpt', is_best=False):
        """Save training checkpoint."""
        # Save the inner model (PMAsymDenoiser) state dict without 'model.' prefix
        # to maintain compatibility with the original checkpoint format
        checkpoint = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'val_count': self.val_count,
            'model_state_dict': self.model.model.state_dict(),  # Access inner model
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')
        
        if is_best:
            best_path = self.checkpoint_dir / 'best.ckpt'
            shutil.copy(filepath, best_path)
            print(f'Best checkpoint updated: {best_path}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load the inner model state dict
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.val_count = checkpoint.get('val_count', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f'Resumed from step {self.global_step}, epoch {self.current_epoch}, val_count {self.val_count}')
    
    def warmup_lr(self):
        """Apply learning rate warmup."""
        if self.global_step < self.warmup_step:
            if self.global_step == 0:
                self.base_lr_list = [pg['lr'] for pg in self.optimizer.param_groups]
            ratio = float((self.global_step + 1) / self.warmup_step)
            for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.base_lr_list[i] * ratio
        
    def clip_gradients(self, max_norm):
        """Clip gradients by norm."""
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def training_step(self, batch, batch_idx):
        """Execute one training step."""
        step_start_time = time.time()
        print(f"\nTraining Step {self.global_step} | Batch {batch_idx} | Samples: {batch.num_graphs}")
        
        # Apply warmup
        self.warmup_lr()
        
        # Forward pass with OOM handling
        forward_start = time.time()
        print("  Starting forward pass...")
        while True:
            try:
                outputs = self.model(batch)
                forward_time = time.time() - forward_start
                print(f"  Forward pass completed in {forward_time:.2f}s")
                break
            except Exception as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e):
                    print("\n  OOM in forward pass, reducing batch size...")
                    print('     OOM details:', len(batch.node_type_batch), len(batch.pocket_pos_batch))
                    batch = self.model.reduce_batch(batch)
                else:
                    raise e

        # Loss calculation
        loss_start = time.time()
        print("  Calculating loss...")
        loss_dict = self.model.loss_func(batch, outputs)
        loss_time = time.time() - loss_start
        print(f"  Loss calculation completed in {loss_time:.2f}s")

        if 'loss' in loss_dict:
            loss = loss_dict['loss']
        else:
            loss = loss_dict['mixed/total']
            
        print(f"  Loss: {loss.item():.4f}")

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clip_val = getattr(self.config.train, 'gradient_clip_val', None)
        if gradient_clip_val:
            self.clip_gradients(gradient_clip_val)
        
        # Optimizer step
        self.optimizer.step()
        
        # Learning rate scheduling (step-based)
        # Only step schedulers that don't need metrics (ReduceLROnPlateau needs validation loss)
        if not self.scheduler_needs_metric:
            scheduler_params = self.config.train.scheduler.params
            if scheduler_params.get('interval', 'epoch') == 'step':
                self.scheduler.step()

        # Logging
        log_start = time.time()
        print("  Logging metrics...")
        
        # Log to TensorBoard
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/lr', current_lr, self.global_step)
        
        # Calculate gradient norm
        grad_norm = get_grad_norm(self.model, norm_type=2)
        self.writer.add_scalar('train/grad_norm', grad_norm, self.global_step)
        
        # Memory usage
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            self.writer.add_scalar('train/memory_GB', mem_gb, self.global_step)

        log_time = time.time() - log_start
        total_time = time.time() - step_start_time
        print(f"  Logging completed in {log_time:.2f}s")
        print(f"  Total step time: {total_time:.2f}s")
        print(f"  Speed: {batch.num_graphs/total_time:.2f} samples/sec")

        return loss.item()

    def validation_step(self, batch, batch_idx):
        """Execute one validation step."""
        val_start_time = time.time()
        
        # Forward pass with OOM handling
        while True:
            try:
                outputs = self.model(batch)
                break
            except Exception as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e):
                    print("\n  OOM in validation, reducing batch size...")
                    batch = self.model.reduce_batch(batch)
                else:
                    raise e
        
        # Loss calculation
        loss_dict = self.model.loss_func(batch, outputs)
        
        return loss_dict
    
    def validate(self):
        """Run validation on the entire validation set."""
        print(f"\nStarting Validation at step {self.global_step}")
        self.model.eval()
        
        all_loss_dicts = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                batch = batch.to(self.device)
                loss_dict = self.validation_step(batch, batch_idx)
                all_loss_dicts.append(loss_dict)
        
        # Aggregate validation losses
        aggregated_losses = {}
        for key in all_loss_dicts[0].keys():
            values = [d[key].item() if isinstance(d[key], torch.Tensor) else d[key] 
                     for d in all_loss_dicts]
            aggregated_losses[key] = sum(values) / len(values)
        
        # Log to TensorBoard
        for key, value in aggregated_losses.items():
            self.writer.add_scalar(f'val/{key}', value, self.global_step)
        
        # Get main validation loss
        if 'mixed/total' in aggregated_losses:
            val_loss = aggregated_losses['mixed/total']
        elif 'loss' in aggregated_losses:
            val_loss = aggregated_losses['loss']
        else:
            val_loss = list(aggregated_losses.values())[0]
        
        print(f"  Validation Loss: {val_loss:.4f}")
        
        # Increment validation counter
        self.val_count += 1
        
        # Step scheduler if it needs metrics (like ReduceLROnPlateau)
        # Only step every 'frequency' validations
        if self.scheduler_needs_metric and (self.val_count % self.scheduler_frequency == 0):
            self.scheduler.step(val_loss)
            print(f"  Scheduler stepped (val_count={self.val_count}, frequency={self.scheduler_frequency})")
        
        self.model.train()
        return val_loss
    
    def train(self, max_steps, val_check_interval, ckpt_every_n_steps):
        """Main training loop."""
        print("\nStarting Training")
        print(f"  Max steps: {max_steps}")
        print(f"  Validation interval: {val_check_interval}")
        print(f"  Checkpoint interval: {ckpt_every_n_steps}")
        
        self.model.train()
        
        # Create infinite data iterator
        train_iter = iter(self.train_loader)
        
        progress_bar = tqdm(total=max_steps, initial=self.global_step, desc='Training')
        
        while self.global_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                self.current_epoch += 1
                
                # Step scheduler at epoch level if needed
                if not self.scheduler_needs_metric:
                    scheduler_params = self.config.train.scheduler.params
                    if scheduler_params.get('interval', 'epoch') == 'epoch':
                        self.scheduler.step()
            
            batch = batch.to(self.device)
            
            # Training step
            loss = self.training_step(batch, self.global_step)
            
            self.global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Validation
            if self.global_step % val_check_interval == 0:
                val_loss = self.validate()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(filename=f'step={self.global_step}.ckpt', is_best=True)
            
            # Save checkpoint
            if self.global_step % ckpt_every_n_steps == 0:
                self.save_checkpoint(filename=f'step={self.global_step}.ckpt')
            
            # Save last checkpoint
            if self.global_step % 100 == 0:
                self.save_checkpoint(filename='last.ckpt')
        
        progress_bar.close()
        
        # Final checkpoint
        self.save_checkpoint(filename='final.ckpt')
        self.writer.close()
        
        print("\nTraining finished.")


is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    is_vscode = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
        default='configs/train/train_pxm.yml')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs. -1 for auto-detect')
    parser.add_argument('--device', type=int, default=0, help='GPU device id. Only for single GPU training.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--logdir', type=str, default='lightning_logs_tasked')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    
    # Device detection
    print("Detecting available devices...")
    
    if args.num_gpus == -1:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.device}')
            print(f"CUDA GPU detected, using GPU {args.device}")
        else:
            device = torch.device('cpu')
            print("No CUDA GPU detected, using CPU")
    else:
        if args.num_gpus > 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{args.device}')
            print(f"Using CUDA GPU {args.device}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    
    print(f"Final device: {device}")
    
    # Log directory
    if is_vscode:
        args.logdir = os.path.join('vscode', args.logdir)
    dir_names = os.path.dirname(args.config).split('/')
    is_train = dir_names.index('train')
    names = dir_names[is_train+1:]
    args.logdir = '/'.join([args.logdir] + names)

    # Load configs
    config = make_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Setup log directory
    log_dir = Path(args.logdir) / config_name
    if args.tag:
        log_dir = log_dir / args.tag
    else:
        # Create version directory
        version = 0
        while (log_dir / f'version_{version}').exists():
            version += 1
        log_dir = log_dir / f'version_{version}'
    
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Data module
    dm = DataModule(config, global_rank=0, world_size=1)
    dm.setup()
    in_dims = dm.get_in_dims()

    # Model
    model = MYGOModel(config, **in_dims)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler_config = config.train.scheduler
    scheduler = get_scheduler(scheduler_config.instance, optimizer)
    warmup_step = getattr(scheduler_config, "warmup_step", 0)
    if warmup_step > 0:
        print(f'Warmup steps: {warmup_step}')

    # Resume checkpoint path
    ckpt_path = None
    if args.resume:
        ckpt_path = os.path.join(os.path.dirname(str(log_dir)),
                        args.resume, 'checkpoints/last.ckpt')
        print(f'Resume from {ckpt_path}')

    # Save source code
    curr_dir = '.'
    save_dir = log_dir / "src"
    copy_py_files(curr_dir, str(save_dir), base=True)
    
    # Save config
    config_dir = log_dir / 'train_config'
    config_dir.mkdir(exist_ok=True)
    save_config(config, str(config_dir / os.path.basename(args.config)))

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=dm.train_loader,
        val_loader=dm.val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        log_dir=str(log_dir),
        warmup_step=warmup_step,
        resume_ckpt=ckpt_path
    )

    # Start training
    trainer.train(
        max_steps=config.train.max_steps,
        val_check_interval=config.train.val_check_interval,
        ckpt_every_n_steps=config.train.ckpt_every_n_steps
    )
