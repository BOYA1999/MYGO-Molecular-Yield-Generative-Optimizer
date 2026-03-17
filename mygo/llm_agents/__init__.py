"""
LLM Agents Module for Molecular Yield Generative Optimizer (MYGO)

This module provides Async LLM-based agents for enhancing molecular generation.
Architecture:
- BaseLLM: Abstract base class defining the async interface
- Agents (GPT4, Claude, DeepSeek): Concrete implementations
- Advisors/Analyzers: Business logic wrappers around Agents
"""

# Base Interface
from .base_llm import BaseLLM

# Concrete Agents
from .gpt4_agent import GPT4Agent
from .claude_agent import ClaudeAgent
from .deepseek_agent import DeepSeekAgent

# Local LLM (optional, for groups with sufficient resources)
try:
    from .local_chemistry_llm import LocalChemistryLLM
    _LOCAL_LLM_AVAILABLE = True
except ImportError:
    _LOCAL_LLM_AVAILABLE = False
    LocalChemistryLLM = None

# Business Logic Agents
from .pocket_analyzer import PocketAnalyzer
from .generation_advisor import GenerationAdvisor
from .molecule_evaluator import MoleculeEvaluator

__all__ = [
    'BaseLLM',
    'GPT4Agent',
    'ClaudeAgent',
    'DeepSeekAgent',
    'PocketAnalyzer',
    'GenerationAdvisor',
    'MoleculeEvaluator',
]

# Conditionally add LocalChemistryLLM if available
if _LOCAL_LLM_AVAILABLE:
    __all__.append('LocalChemistryLLM')

__version__ = "0.2.0"  # version marker