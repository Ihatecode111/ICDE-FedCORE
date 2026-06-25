# FedCORE: Communication-Efficient Federated Data Preparation via Regime-Aware Meta-Causal Prompting (Submitted to PVLDB Vol. 20)

This repository contains the official core system primitives and execution scripts for the paper **"Communication-Efficient Federated Data Preparation via Regime-Aware Meta-Causal Prompting"**. 

FedCORE is a decentralized data preparation engine designed for resource-constrained edge environments. It decouples causal data purification from network bandwidth bottlenecks by replacing standard $\mathcal{O}(d_{model})$ parameter synchronization with a compact $\mathcal{O}(m)$ Meta-Causal Prompt. Additionally, it incorporates an $\mathcal{O}(1)$ dynamic CWAZ router to maintain stable processing throughput for heterogeneous edge streams.

## 1. Repository Structure

To facilitate clear artifact evaluation, we adopt a flat, minimalist library design. The repository isolates the core causal modules from the end-to-end execution pipelines:

* `models/fedcore_engine.py` - Core implementation of the $\mathcal{O}(m)$ Global Server Aggregation and local Regime-Aware Prompting module.
* `models/cwaz_sniffer.py` - The meta-driven dynamic routing mechanism for throughput maximization (Section 4.3).
* `models/ec_irm_operator.py` - The Mask-Driven Orthogonal Refining Operator enforcing Energy-Conditioned IRM (Section 4.4).
* `main_air_quality.py` - End-to-end execution script for Spatio-temporal Regression workloads.
* `main_credit_card.py` - End-to-end execution script for imbalanced Financial Tabular workloads.
* `main_yelp.py` - End-to-end execution script for Topologically-aware E-commerce Graph workloads.
* `requirements.txt` - Minimal dependency environment file.

## 2. Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/) to manage the environment. The underlying system primitives have been tested on Ubuntu 22.04 with Python 3.10 and PyTorch 2.2+.

```bash
# Clone the repository
git clone [https://github.com/Ihatecode111/FedCORE.git](https://github.com/Ihatecode111/FedCORE.git)
cd FedCORE

# Create and activate the conda environment
conda create -n fedcore python=3.10 -y
conda activate fedcore

# Install dependencies (CPU or CUDA 11.8/12.1 supported)
pip install -r requirements.txt
```

## 3. Core System Primitives & Verification

Rather than providing a monolithic execution script that may encounter issues due to local hardware disparities, we expose the decoupled system primitives. Reviewers and practitioners can independently instantiate and verify the core mathematical mechanisms described in the paper.

### 3.1 Global Orchestration & Regime-Aware Prompting (Section 4.2)
The server transmits a compact prompt $p \in \mathbb{R}^m$, which the client couples with its local environmental signature to generate a dynamic structural mask.

```python
import torch
from models.fedcore_engine import FedCOREServer, RegimeAwarePrompting

# 1. Server broadcasts the O(m) payload (e.g., m=32)
server = FedCOREServer(prompt_dim=32, device='cpu')
global_prompt = server.broadcast()

# 2. Edge Client couples prompt with local context (Signature S)
prompt_module = RegimeAwarePrompting(prompt_dim=32, feature_dim=128)
local_signature = torch.randn(128) # Extracted via backbone GAP

# Generate causal mask (m_c)
causal_mask = prompt_module(global_prompt, local_signature)
```

### 3.2 High-Throughput Dynamic Routing (Section 4.3)
To manage the computational load on edge nodes, the CWAZ Sniffer acts as an $\mathcal{O}(1)$ proxy to isolate long-tail structural bias.

```python
from models.cwaz_sniffer import CWAZSniffer

# Initialize Sniffer with adaptive Exponential Moving Average (EMA)
sniffer = CWAZSniffer(initial_tau=0.5, ema_momentum=0.9)

batch_features = torch.randn(256, 128) # Batch of 256 instances
batch_masks = causal_mask.expand(256, -1)

# Dynamically route the data stream
# Clean instances bypass computation; Dirty instances are isolated
clean_idx, dirty_idx = sniffer(batch_features, batch_masks)
print(f"Bypassed (High-Throughput): {len(clean_idx)} | Routed (CWAZ): {len(dirty_idx)}")
```

### 3.3 Mask-Driven Orthogonal Refining (Section 4.4)
The routed dirty data undergoes mathematical disentanglement via the Energy-Conditioned IRM operator, providing an alternative to standard HSIC kernels.

```python
from models.ec_irm_operator import ECIRMOperator

# Initialize Operator (e.g., for Binary Fraud Classification)
operator = ECIRMOperator(feature_dim=128, num_classes=2, lambda_irm=1.5)

dirty_features = batch_features[dirty_idx]
dirty_masks = batch_masks[dirty_idx]
dirty_targets = torch.randint(0, 2, (len(dirty_idx),))

# Execute causal orthogonal penalty and invariant risk minimization
loss, loss_dict = operator(dirty_features, dirty_masks, dirty_targets)
print(f"Causal Disentanglement Loss Components: {loss_dict}")
```

## 4. Data Acquisition & Compliance

To strictly adhere to GDPR/CCPA privacy compliance and the original distributors' licensing, we do not bundle the raw benchmark datasets within this repository. 

To run the complete training pipelines, please acquire the datasets from their official sources:
* **Beijing Multi-Site Air-Quality**: Available via the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data).
* **Credit Card Fraud Detection**: Available via Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
* **Yelp Open Dataset**: Available via the official Yelp Dataset Portal (https://www.yelp.com/dataset).

**Data Preparation**: Map the downloaded data into heterogeneous Non-IID client silos based on location (Air-Quality), time-windows (Credit Card), or sub-graphs (Yelp) before routing them into the `main_*.py` pipelines.

## 5. Executing End-to-End Pipelines

You can verify the tensor topologies and system logic by running the execution scripts. These scripts validate the parameter dimensions without requiring the physical raw data.

```bash
# Verify the Spatio-temporal pipeline
python main_air_quality.py --prompt_dim 32 --feature_dim 128

# Verify the Imbalanced Tabular pipeline
python main_credit_card.py --prompt_dim 32 --feature_dim 256

# Verify the Graph/Text pipeline
python main_yelp.py --prompt_dim 32 --feature_dim 256
```
