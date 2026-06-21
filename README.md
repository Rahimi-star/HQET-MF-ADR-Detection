# HQET-MF-ADR-Detection
# HQET-MF: Hybrid Quantum-Enhanced Transformer with Multi-View Filtering for Adverse Drug Reaction Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE_JBHI-orange.svg)](https://ieeexplore.ieee.org/)

## 📋 Overview

**HQET-MF** is a novel hybrid architecture for Adverse Drug Reaction (ADR) detection in medical texts. The framework synergistically integrates:

- **Quantum-enhanced representation learning** using a 6-qubit Variational Quantum Circuit (VQC)
- **Multi-view handcrafted features** (20 linguistic and domain-specific features)
- **Bias-aware active learning** for class imbalance mitigation
- **Semantic post-filtering** for clinically prudent decision-making

This repository contains the complete implementation for the paper:

> *"Hybrid Quantum-Enhanced Transformer with Multi-View Features for Adverse Drug Reaction Detection in Medical Texts"*  
> **Authors:** Maryam Negahi, Azam Bastanfard, Rezvan Rahimi  
> **Journal:** IEEE Journal of Biomedical and Health Informatics (Under Review)

---

## ✨ Key Features

- **Dual Dataset Support**: TwiMed PubMed subset (n=1,000) and ADE Corpus v2 (n=23,516)
- **Quantum Integration**: PennyLane-based VQC with 6 qubits, 2 layers
- **Multi-View Features**: 20 handcrafted features (lexical, syntactic, semantic distance)
- **Active Learning**: Bias-aware query strategy with positive-focus (25%)
- **Comprehensive Evaluation**: 5-fold cross-validation with statistical analysis
- **Reproducibility**: Fixed random seeds, mixed precision training, early stopping

---

## 📊 Performance Summary

| Dataset | F1-Score | Accuracy | Precision | Recall | AUC-ROC | AUPRC |
|---------|----------|----------|-----------|--------|---------|-------|
| **TwiMed PubMed** (n=1,000, 19.1% ADR+) | 0.756 ± 0.017 | 0.913 ± 0.012 | 0.831 ± 0.093 | 0.702 ± 0.054 | 0.972 ± 0.009 | 0.895 ± 0.051 |
| **ADE Corpus v2** (n=23,516, 29.0% ADR+) | 0.902 ± 0.007 | 0.942 ± 0.004 | 0.885 ± 0.011 | 0.920 ± 0.007 | 0.981 ± 0.001 | 0.954 ± 0.006 |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9 or higher
python --version

# Install dependencies
pip install -r requirements.txt
