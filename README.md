# FedCORE: Communication-Efficient Federated Data Preparation via Regime-Aware Meta-Causal Prompting (ICDE 2027)

This repository contains the official core system primitives and architectural blueprints for the paper **"Communication-Efficient Federated Data Preparation via Regime-Aware Meta-Causal Prompting"**. 

FedCORE is a decentralized data preparation engine explicitly designed for resource-constrained edge environments. It physically decouples causal data purification from network bandwidth constraints by replacing massive $\mathcal{O}(d_{model})$ parameter synchronization with an ultra-low-dimensional $\mathcal{O}(m)$ Meta-Causal Prompt. Furthermore, it incorporates an $\mathcal{O}(1)$ dynamic CWAZ router to guarantee high-throughput processing for heterogeneous data streams.

## 1. Repository Structure

To facilitate transparent artifact evaluation and pure structural integration, we adopt a flat, minimalist library design. The repository strictly isolates the core system contributions from the macroscopic architectural blueprints:

* `models/fedcore_engine.py` - Core implementation of the $\mathcal{O}(m)$ Global Server Aggregation and local Regime-Aware Prompting module.
* `models/cwaz_sniffer.py` - The meta-driven dynamic routing mechanism for throughput maximization (Section 4.3).
* `models/ec_irm_operator.py` - The Mask-Driven Orthogonal Refining Operator enforcing Energy-Conditioned IRM (Section 4.4).
* `main_air_quality.py` - End-to-end architectural blueprint for Spatio-temporal Regression workloads.
* `main_credit_card.py` - End-to-end architectural blueprint for highly imbalanced Financial Tabular workloads.
* `main_yelp.py` - End-to-end architectural blueprint for Topologically-aware E-commerce Graph workloads.
* `requirements.txt` - Minimal dependency environment file.

## 2. Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/) to manage the environment to avoid package conflicts. The underlying system primitives have been thoroughly tested on Ubuntu 22.04 with Python 3.10 and PyTorch 2.2+.

```bash
# Clone the repository (Anonymous URL for Double-Blind Review)
git clone [https://github.com/anonymous-repo/FedCORE.git](https://github.com/anonymous-repo/FedCORE.git)
cd FedCORE

# Create and activate the conda environment
conda create -n fedcore python=3.10 -y
conda activate fedcore

# Install dependencies (CPU or CUDA 11.8/12.1 supported)
pip install -r requirements.txt
```

*(Note: If you are evaluating the structural codebase on a CPU-only machine, the PyTorch tensors will automatically fallback to the `cpu` device without throwing CUDA execution errors.)*

## 3. Core System Primitives & Verification

Rather than providing a rigid, error-prone monolithic execution script that often fails due to hardware disparities, we provide the raw, decoupled system primitives. Reviewers and practitioners can instantiate and verify the exact mathematical mechanisms described in the paper as follows.

### 3.1 Global Orchestration & Regime-Aware Prompting (Section 4.2)
The server transmits only an ultra-low-dimensional prompt $p \in \mathbb{R}^m$, which the client couples with its local environmental signature to generate a structural mask.

```python
import torch
from models.fedcore_engine import FedCOREServer, RegimeAwarePrompting

# 1. Server broadcasts the O(m) payload (e.g., m=64)
server = FedCOREServer(prompt_dim=64, device='cpu')
global_prompt = server.broadcast()

# 2. Edge Client couples prompt with local context (Signature S)
prompt_module = RegimeAwarePrompting(prompt_dim=64, feature_dim=128)
local_signature = torch.randn(128) # Extracted via backbone GAP

# Generate causal mask (m_c)
causal_mask = prompt_module(global_prompt, local_signature)
```

### 3.2 High-Throughput Dynamic Routing (Section 4.3)
To prevent the edge node from crashing under heavy causal computation, the CWAZ Sniffer acts as an $\mathcal{O}(1)$ proxy to isolate only the long-tail structural bias.

```python
from models.cwaz_sniffer import CWAZSniffer

# Initialize Sniffer with adaptive Exponential Moving Average (EMA)
sniffer = CWAZSniffer(initial_tau=0.5, ema_momentum=0.9)

batch_features = torch.randn(256, 128) # Batch of 256 instances
batch_masks = causal_mask.expand(256, -1)

# Dynamically route the data stream
# Clean instances bypass heavy computation; Dirty instances are strictly isolated
clean_idx, dirty_idx = sniffer(batch_features, batch_masks)
print(f"Bypassed (High-Throughput): {len(clean_idx)} | Routed (CWAZ): {len(dirty_idx)}")
```

### 3.3 Mask-Driven Orthogonal Refining (Section 4.4)
The routed dirty data undergoes rigorous mathematical disentanglement via our Energy-Conditioned IRM operator, completely bypassing the need for computationally heavy HSIC kernels.

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

To strictly adhere to academic integrity, GDPR/CCPA privacy compliance, and the original distributors' licensing restrictions, we do not bundle the raw real-world benchmark datasets within this repository.

To construct the complete end-to-end training pipelines, reviewers and researchers must acquire the raw datasets from their official sources:
* **Beijing Multi-Site Air-Quality**: Available via the UCI Machine Learning Repository.
* **Credit Card Fraud Detection**: Available via Kaggle.
* **Yelp Open Dataset**: Available via the official Yelp Dataset Portal.

**Data Preparation**: Once downloaded, map the dataset distributions into heterogeneous Non-IID client silos based on location (Air-Quality), time-windows (Credit Card), or sub-graphs (Yelp) before feeding them into the provided `main_*.py` architectural blueprints.

## 5. Executing Architectural Blueprints

You can verify the end-to-end tensor topologies and system instantiation by running the blueprint scripts. These scripts validate the parameter dimensions and architectural integrity without requiring the physical raw data.

```bash
# Verify the Spatio-temporal pipeline
python main_air_quality.py --prompt_dim 64 --feature_dim 128

# Verify the Imbalanced Tabular pipeline
python main_credit_card.py --prompt_dim 32 --feature_dim 256

# Verify the Graph/Text pipeline
python main_yelp.py --prompt_dim 64 --feature_dim 256
```

## 6. Acknowledgments

We gratefully acknowledge the developers of the PyTorch Ecosystem and PyTorch Geometric (PyG) for providing the highly optimized tensor operations that enabled the edge-friendly implementation of our linear-time orthogonal constraints.
