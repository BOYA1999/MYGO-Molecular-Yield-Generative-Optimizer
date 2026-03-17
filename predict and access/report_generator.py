"""
Report Generator

Generates comprehensive ADMET evaluation reports in multiple formats.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
from rdkit import Chem

from .ensemble_predictor import EnsembleADMETPredictor


class ReportGenerator:
    """
    Generates ADMET evaluation reports in multiple formats.
    
    Supports:
    - JSON format (detailed)
    - CSV format (tabular)
    - HTML format (visual report)
    """
    
    def __init__(self, ensemble_predictor: Optional[EnsembleADMETPredictor] = None):
        """
        Initialize report generator.
        
        Args:
            ensemble_predictor: EnsembleADMETPredictor instance (optional)
        """
        self.ensemble_predictor = ensemble_predictor
    
    def generate_report(
        self,
        mol: Union[Chem.Mol, str],
        output_dir: str,
        formats: List[str] = ['json', 'csv', 'html'],
        mol_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate evaluation report for a molecule.
        
        Args:
            mol: RDKit molecule object or SMILES string
            output_dir: Output directory for reports
            formats: List of formats to generate ('json', 'csv', 'html')
            mol_name: Optional molecule name/identifier
            
        Returns:
            Dictionary with paths to generated report files
        """
        if self.ensemble_predictor is None:
            raise ValueError("Ensemble predictor not provided")
        
        # Run predictions
        results = self.ensemble_predictor.predict_all(mol)
        
        # Get SMILES
        if isinstance(mol, str):
            smiles = mol
        else:
            smiles = Chem.MolToSmiles(mol)
        
        if mol_name is None:
            mol_name = f"mol_{smiles[:20]}"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        report_paths = {}
        
        # Generate reports in requested formats
        if 'json' in formats:
            json_path = os.path.join(output_dir, f"{mol_name}_report.json")
            self._generate_json_report(results, json_path)
            report_paths['json'] = json_path
        
        if 'csv' in formats:
            csv_path = os.path.join(output_dir, f"{mol_name}_report.csv")
            self._generate_csv_report(results, csv_path)
            report_paths['csv'] = csv_path
        
        if 'html' in formats:
            html_path = os.path.join(output_dir, f"{mol_name}_report.html")
            self._generate_html_report(results, html_path, mol_name)
            report_paths['html'] = html_path
        
        return report_paths
    
    def generate_batch_report(
        self,
        mols: List[Union[Chem.Mol, str]],
        output_dir: str,
        formats: List[str] = ['json', 'csv'],
        mol_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate reports for a batch of molecules.
        
        Args:
            mols: List of RDKit molecule objects or SMILES strings
            output_dir: Output directory for reports
            formats: List of formats to generate
            mol_names: Optional list of molecule names
            
        Returns:
            Dictionary with paths to generated report files
        """
        if self.ensemble_predictor is None:
            raise ValueError("Ensemble predictor not provided")
        
        # Run batch predictions
        all_results = self.ensemble_predictor.predict_batch(mols)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        report_paths = {}
        
        # Generate summary CSV
        if 'csv' in formats:
            csv_path = os.path.join(output_dir, "batch_report.csv")
            self._generate_batch_csv_report(all_results, csv_path)
            report_paths['csv'] = csv_path
        
        # Generate summary JSON
        if 'json' in formats:
            json_path = os.path.join(output_dir, "batch_report.json")
            self._generate_batch_json_report(all_results, json_path)
            report_paths['json'] = json_path
        
        return report_paths
    
    def _generate_json_report(self, results: Dict[str, Any], output_path: str):
        """Generate JSON format report."""
        # Add metadata
        report = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "version": "1.0",
                "smiles": results.get("smiles", ""),
            },
            "overall_admet_score": results.get("overall_admet_score", 0.0),
            "predictions": results.get("predictions", {}),
            "summary": self.ensemble_predictor.get_summary(results.get("predictions", {}))
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    def _generate_csv_report(self, results: Dict[str, Any], output_path: str):
        """Generate CSV format report."""
        rows = []
        
        smiles = results.get("smiles", "")
        overall_score = results.get("overall_admet_score", 0.0)
        
        predictions = results.get("predictions", {})
        
        # Extract key values for CSV
        row = {
            "SMILES": smiles,
            "Overall_ADMET_Score": overall_score,
        }
        
        # Metabolism
        if 'metabolism' in predictions and predictions['metabolism'].get('success'):
            pred = predictions['metabolism'].get('prediction', {})
            if isinstance(pred, dict):
                row["Metabolic_Stability"] = pred.get('metabolic_stability', 'N/A')
                row["CYP450_Substrate"] = pred.get('cyp450_substrate_likelihood', 'N/A')
                row["Clearance"] = pred.get('estimated_clearance', 'N/A')
        
        # Plasma exposure
        if 'plasma_exposure' in predictions and predictions['plasma_exposure'].get('success'):
            pred = predictions['plasma_exposure'].get('prediction', {})
            if isinstance(pred, dict):
                row["Bioavailability"] = pred.get('bioavailability', 'N/A')
                row["Plasma_Protein_Binding"] = pred.get('plasma_protein_binding_percentage', 'N/A')
                row["Estimated_Cmax"] = pred.get('estimated_cmax_ng_ml', 'N/A')
                row["Estimated_AUC"] = pred.get('estimated_auc_ng_h_ml', 'N/A')
        
        # BBB
        if 'bbb' in predictions and predictions['bbb'].get('success'):
            pred = predictions['bbb'].get('prediction', {})
            if isinstance(pred, dict):
                row["BBB_Penetration"] = pred.get('bbb_penetration', 'N/A')
                row["BBB_Score"] = pred.get('permeability_score', 'N/A')
                row["Brain_Plasma_Ratio"] = pred.get('brain_to_plasma_ratio', 'N/A')
        
        # Organ toxicity
        if 'organ_toxicity' in predictions and predictions['organ_toxicity'].get('success'):
            pred = predictions['organ_toxicity'].get('prediction', {})
            if isinstance(pred, dict):
                row["Overall_Toxicity"] = pred.get('overall_toxicity', 'N/A')
                row["Hepatotoxicity"] = pred.get('hepatotoxicity', 'N/A')
                row["Nephrotoxicity"] = pred.get('nephrotoxicity', 'N/A')
                row["Cardiotoxicity"] = pred.get('cardiotoxicity', 'N/A')
                row["Neurotoxicity"] = pred.get('neurotoxicity', 'N/A')
        
        # TCM
        if 'tcm' in predictions and predictions['tcm'].get('success'):
            pred = predictions['tcm'].get('prediction', {})
            if isinstance(pred, dict):
                row["Overall_TCM_Risk"] = pred.get('overall_tcm_risk', 'N/A')
                row["Teratogenicity"] = pred.get('teratogenicity', 'N/A')
                row["Carcinogenicity"] = pred.get('carcinogenicity', 'N/A')
                row["Mutagenicity"] = pred.get('mutagenicity', 'N/A')
        
        # Half-life
        if 'half_life' in predictions and predictions['half_life'].get('success'):
            pred = predictions['half_life'].get('prediction', {})
            if isinstance(pred, dict):
                row["Half_Life_Hours"] = pred.get('half_life_hours', 'N/A')
                row["Half_Life_Classification"] = pred.get('classification', 'N/A')
                row["Clearance_Classification"] = pred.get('clearance_classification', 'N/A')
        
        rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    def _generate_html_report(self, results: Dict[str, Any], output_path: str, mol_name: str):
        """Generate HTML format report."""
        smiles = results.get("smiles", "")
        overall_score = results.get("overall_admet_score", 0.0)
        predictions = results.get("predictions", {})
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ADMET Evaluation Report - {mol_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .score {{
            font-size: 24px;
            font-weight: bold;
            color: {'#27ae60' if overall_score >= 0.7 else '#f39c12' if overall_score >= 0.4 else '#e74c3c'};
        }}
        .prediction-section {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .prediction-item {{
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-left: 3px solid #3498db;
        }}
        .label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .value {{
            color: #34495e;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ADMET Evaluation Report</h1>
        <p><strong>Molecule:</strong> {mol_name}</p>
        <p><strong>SMILES:</strong> <code>{smiles}</code></p>
        <p><strong>Generation Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overall ADMET Score</h2>
        <div class="score">Score: {overall_score:.3f}</div>
        <p>Score range: 0.0 (poor) to 1.0 (excellent)</p>
        
        <h2>Detailed Predictions</h2>
"""
        
        # Add each prediction section
        for pred_name, pred_result in predictions.items():
            if not pred_result.get('success', False):
                continue
            
            pred_data = pred_result.get('prediction', {})
            if not pred_data:
                continue
            
            html_content += f"""
        <div class="prediction-section">
            <h3>{pred_name.replace('_', ' ').title()}</h3>
"""
            
            if isinstance(pred_data, dict):
                for key, value in pred_data.items():
                    html_content += f"""
            <div class="prediction-item">
                <span class="label">{key.replace('_', ' ').title()}:</span>
                <span class="value">{value}</span>
            </div>
"""
            else:
                html_content += f"""
            <div class="prediction-item">
                <span class="value">{pred_data}</span>
            </div>
"""
            
            html_content += """
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_batch_csv_report(self, all_results: List[Dict[str, Any]], output_path: str):
        """Generate CSV report for batch of molecules."""
        rows = []
        
        for result in all_results:
            smiles = result.get("smiles", "")
            overall_score = result.get("overall_admet_score", 0.0)
            predictions = result.get("predictions", {})
            
            row = {
                "SMILES": smiles,
                "Overall_ADMET_Score": overall_score,
            }
            
            # Extract key predictions (same as single report)
            # ... (similar extraction logic as _generate_csv_report)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    def _generate_batch_json_report(self, all_results: List[Dict[str, Any]], output_path: str):
        """Generate JSON report for batch of molecules."""
        report = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "version": "1.0",
                "total_molecules": len(all_results),
            },
            "results": all_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

