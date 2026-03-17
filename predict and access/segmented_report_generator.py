"""
Segmented Report Generator

Generates comprehensive ADMET evaluation reports in segmented format.
Supports outputting reports in sections to handle large content.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
from rdkit import Chem

from .ensemble_predictor import EnsembleADMETPredictor


class SegmentedReportGenerator:
    """
    Generates ADMET evaluation reports in segmented format.
    
    Supports:
    - Segmented Markdown output (for large reports)
    - JSON format (detailed)
    - CSV format (tabular)
    - HTML format (visual report)
    """
    
    def __init__(self, ensemble_predictor: Optional[EnsembleADMETPredictor] = None):
        """
        Initialize segmented report generator.
        
        Args:
            ensemble_predictor: EnsembleADMETPredictor instance (optional)
        """
        self.ensemble_predictor = ensemble_predictor
    
    def generate_segmented_report(
        self,
        mol: Union[Chem.Mol, str],
        output_dir: str,
        mol_name: Optional[str] = None,
        max_section_length: int = 5000
    ) -> Dict[str, str]:
        """
        Generate segmented evaluation report for a molecule.
        
        Args:
            mol: RDKit molecule object or SMILES string
            output_dir: Output directory for reports
            mol_name: Optional molecule name/identifier
            max_section_length: Maximum characters per section
            
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
        
        # Generate segmented Markdown report
        md_paths = self._generate_segmented_markdown(
            results, output_dir, mol_name, max_section_length
        )
        report_paths['markdown_sections'] = md_paths
        
        # Generate full Markdown report
        full_md_path = os.path.join(output_dir, f"{mol_name}_full_report.md")
        self._generate_full_markdown_report(results, full_md_path, mol_name)
        report_paths['markdown_full'] = full_md_path
        
        # Generate JSON report
        json_path = os.path.join(output_dir, f"{mol_name}_report.json")
        self._generate_json_report(results, json_path)
        report_paths['json'] = json_path
        
        # Generate CSV report
        csv_path = os.path.join(output_dir, f"{mol_name}_report.csv")
        self._generate_csv_report(results, csv_path)
        report_paths['csv'] = csv_path
        
        # Generate HTML report
        html_path = os.path.join(output_dir, f"{mol_name}_report.html")
        self._generate_html_report(results, html_path, mol_name)
        report_paths['html'] = html_path
        
        return report_paths
    
    def _generate_segmented_markdown(
        self,
        results: Dict[str, Any],
        output_dir: str,
        mol_name: str,
        max_section_length: int
    ) -> List[str]:
        """
        Generate segmented Markdown report.
        
        Returns:
            List of file paths for each section
        """
        smiles = results.get("smiles", "")
        overall_score = results.get("overall_admet_score", 0.0)
        predictions = results.get("predictions", {})
        
        section_paths = []
        section_num = 1
        
        # Section 1: Header and Overview
        header_content = self._generate_header_section(mol_name, smiles, overall_score)
        header_path = os.path.join(output_dir, f"{mol_name}_section_{section_num:02d}_header.md")
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(header_content)
        section_paths.append(header_path)
        section_num += 1
        
        # Section 2: Metabolism Prediction
        if 'metabolism' in predictions and predictions['metabolism'].get('success'):
            metab_content = self._generate_metabolism_section(predictions['metabolism'])
            metab_path = os.path.join(output_dir, f"{mol_name}_section_{section_num:02d}_metabolism.md")
            with open(metab_path, 'w', encoding='utf-8') as f:
                f.write(metab_content)
            section_paths.append(metab_path)
            section_num += 1
        
        # Section 3: Plasma Exposure Prediction
        if 'plasma_exposure' in predictions and predictions['plasma_exposure'].get('success'):
            plasma_content = self._generate_plasma_exposure_section(predictions['plasma_exposure'])
            plasma_path = os.path.join(output_dir, f"{mol_name}_section_{section_num:02d}_plasma_exposure.md")
            with open(plasma_path, 'w', encoding='utf-8') as f:
                f.write(plasma_content)
            section_paths.append(plasma_path)
            section_num += 1
        
        # Section 4: BBB Permeability Prediction
        if 'bbb' in predictions and predictions['bbb'].get('success'):
            bbb_content = self._generate_bbb_section(predictions['bbb'])
            bbb_path = os.path.join(output_dir, f"{mol_name}_section_{section_num:02d}_bbb.md")
            with open(bbb_path, 'w', encoding='utf-8') as f:
                f.write(bbb_content)
            section_paths.append(bbb_path)
            section_num += 1
        
        # Section 5: Organ Toxicity Prediction
        if 'organ_toxicity' in predictions and predictions['organ_toxicity'].get('success'):
            tox_content = self._generate_organ_toxicity_section(predictions['organ_toxicity'])
            tox_path = os.path.join(output_dir, f"{mol_name}_section_{section_num:02d}_organ_toxicity.md")
            with open(tox_path, 'w', encoding='utf-8') as f:
                f.write(tox_content)
            section_paths.append(tox_path)
            section_num += 1
        
        # Section 6: TCM Prediction
        if 'tcm' in predictions and predictions['tcm'].get('success'):
            tcm_content = self._generate_tcm_section(predictions['tcm'])
            tcm_path = os.path.join(output_dir, f"{mol_name}_section_{section_num:02d}_tcm.md")
            with open(tcm_path, 'w', encoding='utf-8') as f:
                f.write(tcm_content)
            section_paths.append(tcm_path)
            section_num += 1
        
        # Section 7: Half-life Prediction
        if 'half_life' in predictions and predictions['half_life'].get('success'):
            hl_content = self._generate_half_life_section(predictions['half_life'])
            hl_path = os.path.join(output_dir, f"{mol_name}_section_{section_num:02d}_half_life.md")
            with open(hl_path, 'w', encoding='utf-8') as f:
                f.write(hl_content)
            section_paths.append(hl_path)
            section_num += 1
        
        # Section 8: Summary and Recommendations
        summary_content = self._generate_summary_section(results)
        summary_path = os.path.join(output_dir, f"{mol_name}_section_{section_num:02d}_summary.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        section_paths.append(summary_path)
        
        return section_paths
    
    def _generate_header_section(self, mol_name: str, smiles: str, overall_score: float) -> str:
        """Generate header section."""
        return f"""# ADMET评估报告 - {mol_name}

## 报告信息

- **生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **分子名称**: {mol_name}
- **SMILES**: `{smiles}`
- **综合ADMET评分**: {overall_score:.3f} / 1.000

## 评分说明

综合ADMET评分范围：0.0（较差）到 1.0（优秀）

评分基于以下六个方面的预测结果：
1. 代谢预测 (Metabolism Prediction)
2. 血浆暴露 (Plasma Exposure)
3. 血脑屏障渗透性 (BBB Permeability)
4. 器官毒性预测 (Organ Toxicity)
5. 致畸致癌致突变预测 (TCM Prediction)
6. 半衰期预测 (Half-life Prediction)

---

## 报告结构

本报告分为以下部分：
- Section 1: 报告头部和概述（本部分）
- Section 2: 代谢预测详细结果
- Section 3: 血浆暴露预测详细结果
- Section 4: 血脑屏障渗透性预测详细结果
- Section 5: 器官毒性预测详细结果
- Section 6: 致畸致癌致突变预测详细结果
- Section 7: 半衰期预测详细结果
- Section 8: 综合总结和建议

---

"""
    
    def _generate_metabolism_section(self, metab_result: Dict[str, Any]) -> str:
        """Generate metabolism prediction section."""
        if not metab_result.get('success', False):
            return f"""# 代谢预测 (Metabolism Prediction)

## 预测状态

**预测失败**

错误信息: {metab_result.get('error', 'Unknown error')}

---
"""
        
        pred = metab_result.get('prediction', {})
        details = metab_result.get('details', {})
        confidence = metab_result.get('confidence', 0.0)
        method = metab_result.get('method', 'unknown')
        
        content = f"""# 代谢预测 (Metabolism Prediction)

## 预测概述

本部分使用**机器学习/深度学习方法**对分子的代谢性质进行预测和验证。

- **预测方法**: {method.upper()}
- **预测置信度**: {confidence:.2%}

---

## 主要预测结果

"""
        
        if isinstance(pred, dict):
            stability = pred.get('metabolic_stability', 'N/A')
            stability_score = pred.get('stability_score', 'N/A')
            cyp450 = pred.get('cyp450_substrate_likelihood', 'N/A')
            clearance = pred.get('estimated_clearance', 'N/A')
            
            content += f"""
### 1. 代谢稳定性 (Metabolic Stability)

- **稳定性等级**: {stability}
- **稳定性评分**: {stability_score if isinstance(stability_score, (int, float)) else 'N/A'}

代谢稳定性反映了分子在体内被代谢酶降解的难易程度。高稳定性意味着分子在体内停留时间更长，有利于药效的发挥。

### 2. CYP450底物可能性 (CYP450 Substrate Likelihood)

- **CYP450底物可能性**: {cyp450}

CYP450酶是肝脏中最重要的代谢酶系统。分子作为CYP450底物的可能性影响其代谢速率和药物相互作用风险。

### 3. 清除率估计 (Estimated Clearance)

- **清除率**: {clearance}

清除率反映了分子从体内被清除的速率。快速清除可能导致药物浓度不足，而慢速清除可能导致药物积累。

"""
            
            # Add stability factors if available
            if 'stability_factors' in details:
                content += "### 影响代谢稳定性的因素\n\n"
                for factor in details['stability_factors']:
                    content += f"- {factor}\n"
                content += "\n"
            
            # Add CYP450 factors if available
            if 'cyp450_factors' in details:
                content += "### 影响CYP450相互作用的因素\n\n"
                for factor in details['cyp450_factors']:
                    content += f"- {factor}\n"
                content += "\n"
            
            # Add molecular properties if available
            if 'molecular_properties' in details:
                props = details['molecular_properties']
                content += "### 相关分子性质\n\n"
                content += "| 性质 | 数值 |\n"
                content += "|------|------|\n"
                for key, value in props.items():
                    key_display = key.replace('_', ' ').title()
                    content += f"| {key_display} | {value} |\n"
                content += "\n"
        
        content += """
## 预测方法说明

本预测采用以下方法：

1. **机器学习模型**: 使用预训练的机器学习模型（如随机森林、梯度提升、神经网络等）基于分子描述符进行预测
2. **规则基础方法**: 基于已知的代谢规律和结构-活性关系进行预测
3. **集成方法**: 结合多种预测方法的结果，提高预测准确性

## 参考文献

预测方法基于以下研究领域：
- 代谢稳定性预测模型
- CYP450底物识别
- 药物代谢动力学（DMPK）研究

---

"""
        return content
    
    def _generate_plasma_exposure_section(self, plasma_result: Dict[str, Any]) -> str:
        """Generate plasma exposure prediction section."""
        if not plasma_result.get('success', False):
            return f"""# 血浆暴露预测 (Plasma Exposure Prediction)

## 预测状态

**预测失败**

错误信息: {plasma_result.get('error', 'Unknown error')}

---
"""
        
        pred = plasma_result.get('prediction', {})
        details = plasma_result.get('details', {})
        confidence = plasma_result.get('confidence', 0.0)
        method = plasma_result.get('method', 'unknown')
        
        content = f"""# 血浆暴露预测 (Plasma Exposure Prediction)

## 预测概述

本部分使用**机器学习/深度学习方法**对分子的血浆暴露性质进行预测和验证。

- **预测方法**: {method.upper()}
- **预测置信度**: {confidence:.2%}

---

## 主要预测结果

"""
        
        if isinstance(pred, dict):
            bioav = pred.get('bioavailability', 'N/A')
            bioav_score = pred.get('bioavailability_score', 'N/A')
            ppb = pred.get('plasma_protein_binding', 'N/A')
            ppb_pct = pred.get('plasma_protein_binding_percentage', 'N/A')
            cmax = pred.get('estimated_cmax_ng_ml', 'N/A')
            auc = pred.get('estimated_auc_ng_h_ml', 'N/A')
            
            content += f"""
### 1. 口服生物利用度 (Oral Bioavailability)

- **生物利用度**: {bioav}
- **生物利用度评分**: {bioav_score if isinstance(bioav_score, (int, float)) else 'N/A'}

生物利用度反映了口服药物被吸收进入血液循环的比例。高生物利用度意味着药物能够更有效地进入体内。

### 2. 血浆蛋白结合率 (Plasma Protein Binding)

- **结合率等级**: {ppb}
- **结合率百分比**: {ppb_pct}%

血浆蛋白结合率影响药物的游离浓度，进而影响药物的分布和清除。高结合率可能降低药物的有效浓度。

### 3. 最大血浆浓度 (Cmax)

- **估计Cmax**: {cmax} ng/mL

Cmax是药物在血浆中达到的最大浓度，反映了药物的吸收速率和程度。

### 4. 药时曲线下面积 (AUC)

- **估计AUC**: {auc} ng·h/mL

AUC反映了药物在体内的总暴露量，是评估药物疗效和毒性的重要参数。

"""
            
            # Add bioavailability factors if available
            if 'bioavailability_factors' in details:
                content += "### 影响生物利用度的因素\n\n"
                for factor in details['bioavailability_factors']:
                    content += f"- {factor}\n"
                content += "\n"
            
            # Add molecular properties if available
            if 'molecular_properties' in details:
                props = details['molecular_properties']
                content += "### 相关分子性质\n\n"
                content += "| 性质 | 数值 |\n"
                content += "|------|------|\n"
                for key, value in props.items():
                    key_display = key.replace('_', ' ').title()
                    content += f"| {key_display} | {value} |\n"
                content += "\n"
        
        content += """
## 预测方法说明

本预测采用以下方法：

1. **机器学习模型**: 基于大量实验数据训练的预测模型
2. **规则基础方法**: 基于Lipinski五规则等已知规律
3. **物理化学性质**: 结合分子量、LogP、极性表面积等性质

## 参考文献

预测方法基于以下研究领域：
- 药物吸收和分布研究
- 药代动力学（PK）建模
- 生物利用度预测模型

---

"""
        return content
    
    def _generate_bbb_section(self, bbb_result: Dict[str, Any]) -> str:
        """Generate BBB permeability prediction section."""
        if not bbb_result.get('success', False):
            return f"""# 血脑屏障渗透性预测 (BBB Permeability Prediction)

## 预测状态

**预测失败**

错误信息: {bbb_result.get('error', 'Unknown error')}

---
"""
        
        pred = bbb_result.get('prediction', {})
        details = bbb_result.get('details', {})
        confidence = bbb_result.get('confidence', 0.0)
        method = bbb_result.get('method', 'unknown')
        
        content = f"""# 血脑屏障渗透性预测 (BBB Permeability Prediction)

## 预测概述

本部分使用**机器学习/深度学习方法**对分子的血脑屏障（BBB）渗透性进行预测和验证。

- **预测方法**: {method.upper()}
- **预测置信度**: {confidence:.2%}

---

## 主要预测结果

"""
        
        if isinstance(pred, dict):
            penetration = pred.get('bbb_penetration', 'N/A')
            perm_score = pred.get('permeability_score', 'N/A')
            ratio = pred.get('brain_to_plasma_ratio', 'N/A')
            
            content += f"""
### 1. 血脑屏障渗透性 (BBB Penetration)

- **渗透性**: {penetration}

血脑屏障是保护大脑的重要屏障。对于中枢神经系统（CNS）药物，需要良好的BBB渗透性；对于非CNS药物，较低的BBB渗透性可以降低神经毒性风险。

### 2. 渗透性评分 (Permeability Score)

- **评分**: {perm_score}

评分范围：0.0（低渗透性）到 1.0（高渗透性）

### 3. 脑/血浆比率 (Brain-to-Plasma Ratio)

- **比率**: {ratio}

脑/血浆比率反映了药物在脑组织和血浆中的分布情况。

"""
            
            # Add BBB factors if available
            if 'bbb_factors' in details:
                content += "### 影响BBB渗透性的因素\n\n"
                for factor in details['bbb_factors']:
                    content += f"- {factor}\n"
                content += "\n"
            
            # Add molecular properties if available
            if 'molecular_properties' in details:
                props = details['molecular_properties']
                content += "### 相关分子性质\n\n"
                content += "| 性质 | 数值 |\n"
                content += "|------|------|\n"
                for key, value in props.items():
                    key_display = key.replace('_', ' ').title()
                    content += f"| {key_display} | {value} |\n"
                content += "\n"
        
        content += """
## 预测方法说明

本预测采用以下方法：

1. **机器学习模型**: 基于BBB渗透性数据库训练的预测模型
2. **规则基础方法**: 基于已知的BBB渗透性规律（如MW < 450, LogP 1-4, TPSA < 90等）
3. **物理化学性质**: 结合分子量、LogP、极性表面积、氢键供体数等性质

## 参考文献

预测方法基于以下研究领域：
- 血脑屏障渗透性研究
- CNS药物设计
- 药物分布预测模型

---

"""
        return content
    
    def _generate_organ_toxicity_section(self, tox_result: Dict[str, Any]) -> str:
        """Generate organ toxicity prediction section."""
        if not tox_result.get('success', False):
            return f"""# 器官毒性预测 (Organ Toxicity Prediction)

## 预测状态

**预测失败**

错误信息: {tox_result.get('error', 'Unknown error')}

---
"""
        
        pred = tox_result.get('prediction', {})
        details = tox_result.get('details', {})
        confidence = tox_result.get('confidence', 0.0)
        method = tox_result.get('method', 'unknown')
        
        content = f"""# 器官毒性预测 (Organ Toxicity Prediction)

## 预测概述

本部分使用**机器学习/深度学习方法**对分子的器官特异性毒性进行预测和验证。

- **预测方法**: {method.upper()}
- **预测置信度**: {confidence:.2%}

---

## 主要预测结果

"""
        
        if isinstance(pred, dict):
            overall_tox = pred.get('overall_toxicity', 'N/A')
            hepatotox = pred.get('hepatotoxicity', 'N/A')
            nephrotox = pred.get('nephrotoxicity', 'N/A')
            cardiotox = pred.get('cardiotoxicity', 'N/A')
            neurotox = pred.get('neurotoxicity', 'N/A')
            
            content += f"""
### 1. 总体毒性评估 (Overall Toxicity)

- **总体毒性**: {overall_tox}

### 2. 肝毒性 (Hepatotoxicity)

- **肝毒性风险**: {hepatotox}

肝脏是药物代谢的主要器官，肝毒性是药物开发中的重要关注点。

### 3. 肾毒性 (Nephrotoxicity)

- **肾毒性风险**: {nephrotox}

肾脏是药物排泄的主要器官，肾毒性可能导致肾功能损害。

### 4. 心脏毒性 (Cardiotoxicity)

- **心脏毒性风险**: {cardiotox}

心脏毒性可能导致心律失常、心肌损伤等严重副作用。

### 5. 神经毒性 (Neurotoxicity)

- **神经毒性风险**: {neurotox}

神经毒性可能影响中枢或周围神经系统功能。

"""
            
            # Add toxic alerts if available
            if 'toxic_alerts' in details:
                content += "### 检测到的毒性结构警示\n\n"
                for alert_type, alerts in details['toxic_alerts'].items():
                    if alerts:
                        content += f"#### {alert_type.replace('_', ' ').title()}\n\n"
                        for alert in alerts:
                            content += f"- {alert}\n"
                        content += "\n"
            
            # Add molecular properties if available
            if 'molecular_properties' in details:
                props = details['molecular_properties']
                content += "### 相关分子性质\n\n"
                content += "| 性质 | 数值 |\n"
                content += "|------|------|\n"
                for key, value in props.items():
                    key_display = key.replace('_', ' ').title()
                    content += f"| {key_display} | {value} |\n"
                content += "\n"
        
        content += """
## 预测方法说明

本预测采用以下方法：

1. **机器学习模型**: 基于毒性数据库训练的预测模型
2. **结构警示识别**: 识别已知的毒性结构片段（toxicophores）
3. **规则基础方法**: 基于已知的毒性规律和结构-毒性关系

## 参考文献

预测方法基于以下研究领域：
- 药物毒性预测
- 结构-毒性关系研究
- 器官特异性毒性机制

---

"""
        return content
    
    def _generate_tcm_section(self, tcm_result: Dict[str, Any]) -> str:
        """Generate TCM prediction section."""
        if not tcm_result.get('success', False):
            return f"""# 致畸致癌致突变预测 (TCM Prediction)

## 预测状态

**预测失败**

错误信息: {tcm_result.get('error', 'Unknown error')}

---
"""
        
        pred = tcm_result.get('prediction', {})
        details = tcm_result.get('details', {})
        confidence = tcm_result.get('confidence', 0.0)
        method = tcm_result.get('method', 'unknown')
        
        content = f"""# 致畸致癌致突变预测 (TCM Prediction)

## 预测概述

本部分使用**机器学习/深度学习方法**对分子的致畸性（Teratogenicity）、致癌性（Carcinogenicity）和致突变性（Mutagenicity）进行预测和验证。

- **预测方法**: {method.upper()}
- **预测置信度**: {confidence:.2%}

---

## 主要预测结果

"""
        
        if isinstance(pred, dict):
            overall_tcm = pred.get('overall_tcm_risk', 'N/A')
            teratogen = pred.get('teratogenicity', 'N/A')
            carcinogen = pred.get('carcinogenicity', 'N/A')
            mutagen = pred.get('mutagenicity', 'N/A')
            
            content += f"""
### 1. 总体TCM风险 (Overall TCM Risk)

- **总体风险**: {overall_tcm}

### 2. 致畸性 (Teratogenicity)

- **致畸性风险**: {teratogen}

致畸性是指药物可能导致胎儿发育异常的风险。对于育龄期患者使用的药物，致畸性评估至关重要。

### 3. 致癌性 (Carcinogenicity)

- **致癌性风险**: {carcinogen}

致癌性是指药物可能增加癌症发生风险。长期使用的药物需要特别关注致癌性。

### 4. 致突变性 (Mutagenicity)

- **致突变性风险**: {mutagen}

致突变性是指药物可能导致基因突变的风险。致突变性通常与致癌性相关。

"""
            
            # Add TCM alerts if available
            if 'tcm_alerts' in details:
                content += "### 检测到的TCM结构警示\n\n"
                for alert_type, alerts in details['tcm_alerts'].items():
                    if alerts:
                        content += f"#### {alert_type.replace('_', ' ').title()}\n\n"
                        for alert in alerts:
                            content += f"- {alert}\n"
                        content += "\n"
            
            # Add molecular properties if available
            if 'molecular_properties' in details:
                props = details['molecular_properties']
                content += "### 相关分子性质\n\n"
                content += "| 性质 | 数值 |\n"
                content += "|------|------|\n"
                for key, value in props.items():
                    key_display = key.replace('_', ' ').title()
                    content += f"| {key_display} | {value} |\n"
                content += "\n"
        
        content += """
## 预测方法说明

本预测采用以下方法：

1. **机器学习模型**: 基于TCM数据库训练的预测模型
2. **结构警示识别**: 识别已知的TCM相关结构片段
3. **规则基础方法**: 基于已知的TCM规律和结构-活性关系

## 参考文献

预测方法基于以下研究领域：
- 致畸性、致癌性、致突变性预测
- 遗传毒性研究
- 结构-毒性关系研究

---

"""
        return content
    
    def _generate_half_life_section(self, hl_result: Dict[str, Any]) -> str:
        """Generate half-life prediction section."""
        if not hl_result.get('success', False):
            return f"""# 半衰期预测 (Half-life Prediction)

## 预测状态

**预测失败**

错误信息: {hl_result.get('error', 'Unknown error')}

---
"""
        
        pred = hl_result.get('prediction', {})
        details = hl_result.get('details', {})
        confidence = hl_result.get('confidence', 0.0)
        method = hl_result.get('method', 'unknown')
        
        content = f"""# 半衰期预测 (Half-life Prediction)

## 预测概述

本部分使用**机器学习/深度学习方法**对分子的消除半衰期进行预测和验证。

- **预测方法**: {method.upper()}
- **预测置信度**: {confidence:.2%}

---

## 主要预测结果

"""
        
        if isinstance(pred, dict):
            hl_hours = pred.get('half_life_hours', 'N/A')
            classification = pred.get('classification', 'N/A')
            clearance = pred.get('clearance_classification', 'N/A')
            
            content += f"""
### 1. 消除半衰期 (Elimination Half-life)

- **半衰期**: {hl_hours} 小时
- **分类**: {classification}

半衰期是药物浓度降低一半所需的时间，反映了药物从体内清除的速率。

**半衰期分类标准**：
- **短半衰期** (<4小时): 需要频繁给药
- **中等半衰期** (4-24小时): 适合每日1-2次给药
- **长半衰期** (>24小时): 可以每日1次或更少频率给药

### 2. 清除率分类 (Clearance Classification)

- **清除率**: {clearance}

清除率反映了药物从体内被清除的速率，与半衰期密切相关。

"""
            
            # Add half-life factors if available
            if 'half_life_factors' in details:
                content += "### 影响半衰期的因素\n\n"
                for factor in details['half_life_factors']:
                    content += f"- {factor}\n"
                content += "\n"
            
            # Add molecular properties if available
            if 'molecular_properties' in details:
                props = details['molecular_properties']
                content += "### 相关分子性质\n\n"
                content += "| 性质 | 数值 |\n"
                content += "|------|------|\n"
                for key, value in props.items():
                    key_display = key.replace('_', ' ').title()
                    content += f"| {key_display} | {value} |\n"
                content += "\n"
        
        content += """
## 预测方法说明

本预测采用以下方法：

1. **机器学习模型**: 基于药代动力学数据训练的预测模型
2. **规则基础方法**: 基于分子性质（如分子量、LogP、清除率等）的预测
3. **物理化学性质**: 结合影响药物清除的分子性质

## 参考文献

预测方法基于以下研究领域：
- 药代动力学（PK）研究
- 药物清除机制
- 半衰期预测模型

---

"""
        return content
    
    def _generate_summary_section(self, results: Dict[str, Any]) -> str:
        """Generate summary section."""
        overall_score = results.get("overall_admet_score", 0.0)
        predictions = results.get("predictions", {})
        summary = self.ensemble_predictor.get_summary(predictions)
        
        content = f"""# 综合总结和建议 (Summary and Recommendations)

## 综合评估

### 总体ADMET评分

**综合评分**: {overall_score:.3f} / 1.000

**评分等级**:
"""
        
        if overall_score >= 0.7:
            content += "- **优秀** (≥0.7): 分子具有良好的ADMET性质，适合进一步开发\n"
        elif overall_score >= 0.4:
            content += "- **中等** (0.4-0.7): 分子ADMET性质中等，可能需要优化\n"
        else:
            content += "- **较差** (<0.4): 分子ADMET性质较差，需要显著优化\n"
        
        content += f"""
### 预测完成情况

- **总预测数**: {summary.get('total_predictions', 0)}
- **成功预测数**: {summary.get('successful_predictions', 0)}
- **失败预测数**: {summary.get('failed_predictions', 0)}

---

## 关键预测结果摘要

"""
        
        key_preds = summary.get('key_predictions', {})
        
        if 'metabolic_stability' in key_preds:
            content += f"- **代谢稳定性**: {key_preds['metabolic_stability']}\n"
        if 'bioavailability' in key_preds:
            content += f"- **生物利用度**: {key_preds['bioavailability']}\n"
        if 'bbb_penetration' in key_preds:
            content += f"- **BBB渗透性**: {key_preds['bbb_penetration']}\n"
        if 'overall_toxicity' in key_preds:
            content += f"- **总体毒性**: {key_preds['overall_toxicity']}\n"
        if 'overall_tcm_risk' in key_preds:
            content += f"- **TCM风险**: {key_preds['overall_tcm_risk']}\n"
        if 'half_life' in key_preds:
            content += f"- **半衰期**: {key_preds['half_life']}\n"
        
        content += """
---

## 优化建议

基于以上预测结果，建议考虑以下优化方向：

### 1. 代谢优化
- 如果代谢稳定性较低，考虑：
  - 减少易代谢位点
  - 引入代谢阻断基团
  - 优化分子量、LogP等性质

### 2. 吸收和分布优化
- 如果生物利用度较低，考虑：
  - 优化LogP值（通常1-3为佳）
  - 减少氢键供体/受体数量
  - 优化分子量和极性表面积

### 3. 毒性降低
- 如果检测到毒性风险，考虑：
  - 移除或替换毒性结构片段
  - 优化分子结构以减少毒性
  - 进行更详细的毒性评估

### 4. 药代动力学优化
- 如果半衰期不理想，考虑：
  - 调整分子性质以影响清除率
  - 优化给药方案
  - 考虑缓释制剂

---

## 下一步工作建议

1. **实验验证**: 对关键预测结果进行体外/体内实验验证
2. **结构优化**: 基于预测结果进行分子结构优化
3. **进一步评估**: 进行更详细的ADMET评估和药代动力学研究
4. **安全性评估**: 进行全面的安全性评估

---

## 报告说明

本报告基于机器学习/深度学习方法生成，预测结果仅供参考。实际药物开发中需要结合实验数据进行验证。

**报告生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

---

"""
        return content
    
    def _generate_full_markdown_report(
        self,
        results: Dict[str, Any],
        output_path: str,
        mol_name: str
    ):
        """Generate full Markdown report combining all sections."""
        smiles = results.get("smiles", "")
        overall_score = results.get("overall_admet_score", 0.0)
        predictions = results.get("predictions", {})
        
        # Start with header
        content = self._generate_header_section(mol_name, smiles, overall_score)
        
        # Add all prediction sections
        if 'metabolism' in predictions:
            content += self._generate_metabolism_section(predictions['metabolism'])
        
        if 'plasma_exposure' in predictions:
            content += self._generate_plasma_exposure_section(predictions['plasma_exposure'])
        
        if 'bbb' in predictions:
            content += self._generate_bbb_section(predictions['bbb'])
        
        if 'organ_toxicity' in predictions:
            content += self._generate_organ_toxicity_section(predictions['organ_toxicity'])
        
        if 'tcm' in predictions:
            content += self._generate_tcm_section(predictions['tcm'])
        
        if 'half_life' in predictions:
            content += self._generate_half_life_section(predictions['half_life'])
        
        # Add summary
        content += self._generate_summary_section(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_json_report(self, results: Dict[str, Any], output_path: str):
        """Generate JSON format report."""
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
        
        row = {
            "SMILES": smiles,
            "Overall_ADMET_Score": overall_score,
        }
        
        # Extract key values (same as report_generator.py)
        if 'metabolism' in predictions and predictions['metabolism'].get('success'):
            pred = predictions['metabolism'].get('prediction', {})
            if isinstance(pred, dict):
                row["Metabolic_Stability"] = pred.get('metabolic_stability', 'N/A')
                row["CYP450_Substrate"] = pred.get('cyp450_substrate_likelihood', 'N/A')
                row["Clearance"] = pred.get('estimated_clearance', 'N/A')
        
        if 'plasma_exposure' in predictions and predictions['plasma_exposure'].get('success'):
            pred = predictions['plasma_exposure'].get('prediction', {})
            if isinstance(pred, dict):
                row["Bioavailability"] = pred.get('bioavailability', 'N/A')
                row["Plasma_Protein_Binding"] = pred.get('plasma_protein_binding_percentage', 'N/A')
                row["Estimated_Cmax"] = pred.get('estimated_cmax_ng_ml', 'N/A')
                row["Estimated_AUC"] = pred.get('estimated_auc_ng_h_ml', 'N/A')
        
        if 'bbb' in predictions and predictions['bbb'].get('success'):
            pred = predictions['bbb'].get('prediction', {})
            if isinstance(pred, dict):
                row["BBB_Penetration"] = pred.get('bbb_penetration', 'N/A')
                row["BBB_Score"] = pred.get('permeability_score', 'N/A')
                row["Brain_Plasma_Ratio"] = pred.get('brain_to_plasma_ratio', 'N/A')
        
        if 'organ_toxicity' in predictions and predictions['organ_toxicity'].get('success'):
            pred = predictions['organ_toxicity'].get('prediction', {})
            if isinstance(pred, dict):
                row["Overall_Toxicity"] = pred.get('overall_toxicity', 'N/A')
                row["Hepatotoxicity"] = pred.get('hepatotoxicity', 'N/A')
                row["Nephrotoxicity"] = pred.get('nephrotoxicity', 'N/A')
                row["Cardiotoxicity"] = pred.get('cardiotoxicity', 'N/A')
                row["Neurotoxicity"] = pred.get('neurotoxicity', 'N/A')
        
        if 'tcm' in predictions and predictions['tcm'].get('success'):
            pred = predictions['tcm'].get('prediction', {})
            if isinstance(pred, dict):
                row["Overall_TCM_Risk"] = pred.get('overall_tcm_risk', 'N/A')
                row["Teratogenicity"] = pred.get('teratogenicity', 'N/A')
                row["Carcinogenicity"] = pred.get('carcinogenicity', 'N/A')
                row["Mutagenicity"] = pred.get('mutagenicity', 'N/A')
        
        if 'half_life' in predictions and predictions['half_life'].get('success'):
            pred = predictions['half_life'].get('prediction', {})
            if isinstance(pred, dict):
                row["Half_Life_Hours"] = pred.get('half_life_hours', 'N/A')
                row["Half_Life_Classification"] = pred.get('classification', 'N/A')
                row["Clearance_Classification"] = pred.get('clearance_classification', 'N/A')
        
        rows.append(row)
        
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
    <title>ADMET评估报告 - {mol_name}</title>
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
        <h1>ADMET评估报告</h1>
        <p><strong>分子:</strong> {mol_name}</p>
        <p><strong>SMILES:</strong> <code>{smiles}</code></p>
        <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>综合ADMET评分</h2>
        <div class="score">评分: {overall_score:.3f}</div>
        <p>评分范围: 0.0 (较差) 到 1.0 (优秀)</p>
        
        <h2>详细预测结果</h2>
"""
        
        # Add each prediction section
        for pred_name, pred_result in predictions.items():
            if not pred_result.get('success', False):
                continue
            
            pred_data = pred_result.get('prediction', {})
            if not pred_data:
                continue
            
            pred_title = pred_name.replace('_', ' ').title()
            html_content += f"""
        <div class="prediction-section">
            <h3>{pred_title}</h3>
"""
            
            if isinstance(pred_data, dict):
                for key, value in pred_data.items():
                    key_display = key.replace('_', ' ').title()
                    html_content += f"""
            <div class="prediction-item">
                <span class="label">{key_display}:</span>
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

