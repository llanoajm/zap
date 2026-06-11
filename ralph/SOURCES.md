# Ralph Research Loop — Bibliography

All sources verified via web search / primary source fetch. Date of verification: 2026-06-11.

---

## A. Zap Repository Papers

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| A1 | Degleris, El Gamal, Rajagopal. "GPU Accelerated Security Constrained Optimal Power Flow." arXiv:2410.17203. Oct 2024. | https://arxiv.org/abs/2410.17203 | Primary source: this repo |
| A2 | Degleris, El Gamal, Rajagopal. "Gradient Methods for Scalable Multi-value Electricity Network Expansion Planning." arXiv:2404.01255. Apr 2024. | https://arxiv.org/abs/2404.01255 | Primary source: this repo; full text verified |
| A3 | Sreekumar, Degleris, Rajagopal. "Large-Scale Network Utility Maximization via GPU-Accelerated Proximal Message Passing." arXiv:2509.10722. 2025. | https://arxiv.org/abs/2509.10722 | Primary source: this repo |
| A4 | Degleris, El Gamal, Rajagopal. "Scalable and Interactive Electricity Grid Expansion Planning." arXiv:2410.13055. Oct 2024. | https://arxiv.org/abs/2410.13055 | Primary source; warm-start 100× speedup result |

---

## B. JEPA Family Papers

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| B1 | Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)." CVPR 2023. arXiv:2301.08243. | https://arxiv.org/abs/2301.08243 | High: CVPR peer-reviewed; Meta AI |
| B2 | Bardes et al. "V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video." Meta AI Tech Report 2024. arXiv:2404.08471. | https://arxiv.org/abs/2404.08471 | High: Meta AI official |
| B3 | Assran et al. (30 authors). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985. Jun 2025. | https://arxiv.org/abs/2506.09985 | High: Meta AI; V-JEPA-2-AC robot planning results verified |
| B4 | Balestriero & LeCun. "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics." arXiv:2511.08544. Nov 2025. | https://arxiv.org/abs/2511.08544 | High: arXiv preprint; SIGReg mechanism verified |
| B5 | Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels." arXiv:2603.19312. Mar 2026. | https://arxiv.org/abs/2603.19312 | High: arXiv preprint; code at github.com/lucas-maes/le-wm |
| B6 | Sobal, Zhang, Cho, Balestriero, Rudner, LeCun. "Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models (PLDM)." arXiv:2502.14819. NeurIPS 2025. | https://arxiv.org/abs/2502.14819 | High: NeurIPS peer-reviewed |
| B7 | Terver, Yang, Ponce, Bardes, LeCun. "What Drives Success in Physical Planning with JEPA World Models?" arXiv:2512.24497. Dec 2025, rev May 2026. | https://arxiv.org/abs/2512.24497 | Medium-High: preprint, includes DROID real-robot results |
| B8 | Destrade, Bounou, Le Lidec, Ponce, LeCun. "Value-Guided Action Planning with JEPA World Models." arXiv:2601.00844. Dec 2025. | https://arxiv.org/abs/2601.00844 | Medium-High: preprint; full abstract verified; latent-distance = cost-to-go shaping |
| B9 | Masip, Swinnen, Hu, Detry, Tuytelaars. "FF-JEPA: Long-Horizon Planning in World Models with Latent Planners." arXiv:2606.09311. Jun 2026. | https://arxiv.org/html/2606.09311v1 | Medium: preprint; full abstract verified; hierarchical subgoal + forward model |
| B12 | Ennadir, Golkar, Sarra. "Joint Embeddings Go Temporal." arXiv:2509.25449. NeurIPS 2024 Workshop "Time Series in the Age of Large Models." | https://arxiv.org/abs/2509.25449 | High: NeurIPS workshop; full abstract verified; JEPA for time series; matches/surpasses SOTA |
| B13 | Skenderi, Li, Tang, Cristani. "Graph-level Representation Learning with Joint-Embedding Predictive Architectures." TMLR. arXiv:2309.36014. | https://arxiv.org/abs/2309.36014 | High: TMLR peer-reviewed; full abstract verified; masked subgraph prediction; code public |
| B10 | Meta AI blog: "V-JEPA 2: World model benchmarks." Jun 2025. | https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/ | High: official Meta AI blog |
| B11 | Meta AI blog: "I-JEPA: AI model learns the world more like humans do." Jun 2023. | https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/ | High: official Meta AI blog |

---

## C. AMI Labs / LeCun's Startup

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| C1 | TechCrunch: "Who's behind AMI Labs, Yann LeCun's world-model startup?" Jan 23, 2026. | https://techcrunch.com/2026/01/23/whos-behind-ami-labs-yann-lecuns-world-model-startup/ | High: TechCrunch primary reporting |
| C2 | TechCrunch: "Yann LeCun's AMI Labs raises $1.03 billion to build world models." Mar 9, 2026. | https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/ | High: TechCrunch primary reporting |

---

## D. Grid Foundation Models

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| D1 | Microsoft Research blog. "GridSFM: A new, small foundation model for the electric grid." May 13, 2026. | https://www.microsoft.com/en-us/research/blog/gridsfm-a-new-small-foundation-model-for-the-electric-grid/ | High: official Microsoft Research blog |
| D2 | Microsoft Research. GridFM project page. | https://www.microsoft.com/en-us/research/project/gridfm/ | High: official |
| D3 | GitHub: microsoft/GridSFM | https://github.com/microsoft/GridSFM | High: official; MIT licensed; architecture verified from source code (model.py, blocks.py, dc_prior.py, hodge_pe.py) |
| D4 | HuggingFace: microsoft/GridSFM_US_power_grid | https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid | High: official data release |
| D5 | Britto Mattos Lima et al. "Building Power Grid Models from Open Data: A Complete Pipeline from OpenStreetMap to Optimal Power Flow." arXiv:2605.04289. May 2026. | https://arxiv.org/abs/2605.04289 | High: companion paper to GridSFM |
| D6 | ESG News: "Microsoft MISO deploy AI to modernize US power grid." Jan 6, 2026. | https://esgnews.com/microsoft-miso-deploy-ai-to-modernize-us-power-grid-as-data-center-and-electrification-demand-accelerates/ | Medium: news coverage of official partnership |
| D7 | Hamann et al. "A Perspective on Foundation Models for the Electric Power Grid." arXiv:2407.09434. Jul 2024. | https://arxiv.org/html/2407.09434v1 | High: IBM Research + NREL + Argonne; published in Joule 2024 |
| D8 | "Foundation models for the electric power grid." Joule, Cell Press, 2024. | https://www.cell.com/joule/fulltext/S2542-4351(24)00470-7 | High: peer-reviewed Cell/Joule journal |
| D9 | IBM Research. "We Built GridFM 0.5." Feb 11, 2025. | https://research.ibm.com/publications/we-built-gridfm-05-from-concepts-to-our-first-working-foundation-model | High: official IBM Research publication |
| D10 | LF Energy. GridFM project page. | https://lfenergy.org/projects/gridfm/ | High: official LF Energy |
| D11 | Linux Foundation press release on OpenGridFM. | https://www.linuxfoundation.org/press/lf-energy-announces-open-source-power-grid-projects-for-ai-modeling-and-edge-applications-plus-scope-3-emissions-visibility | High: official |
| D12 | Lovett et al. (all Google DeepMind). "OPFData: Large-scale datasets for AC optimal power flow with topological perturbations." arXiv:2406.07234. Jun 2024. | https://arxiv.org/abs/2406.07234 | High: Google DeepMind; large-scale dataset paper |
| D13 | Donon et al. (Inria). "Neural networks for power flow: Graph neural solver." EPSR Vol. 189, 2020. | https://www.sciencedirect.com/science/article/abs/pii/S0378779620303515 | High: peer-reviewed EPSR journal; HAL preprint hal-02372741 |
| D14 | Varbella et al. (ETH Zürich). "PowerGraph: A power grid benchmark dataset for GNNs." NeurIPS 2024 D&B track. arXiv:2402.02827. | https://arxiv.org/abs/2402.02827 | High: NeurIPS peer-reviewed |
| D15 | Liu, Wu, Zhu. "Topology-aware Graph Neural Networks for Learning Feasible and Adaptive AC-OPF Solutions." IEEE TPWRS, 2022. arXiv:2205.10129. | https://arxiv.org/abs/2205.10129 | High: IEEE TPWRS peer-reviewed |
| D16 | Wu et al. (Stanford). "PowerGNN." arXiv:2503.22721. Mar 2025. | https://arxiv.org/html/2503.22721v1 | Medium: preprint |
| D17 | Huang et al. "Large Foundation Models for Power Systems." arXiv:2312.07044. Dec 2023. | https://arxiv.org/abs/2312.07044 | Medium: preprint; LLMs for OPF/EV/knowledge |
| D18 | Tu et al. "PowerPM: Foundation Model for Power Systems." arXiv:2408.04057. Aug 2024. | https://arxiv.org/abs/2408.04057 | Medium: preprint; temporal time-series FM |

---

## E. AC-OPF Learning Surrogates

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| E1 | Donti, Rolnick, Kolter. "DC3: A Learning Method for Optimization with Hard Constraints." ICLR 2021. arXiv:2104.12225. | https://arxiv.org/abs/2104.12225 | High: ICLR peer-reviewed; code at github.com/locuslab/DC3 |
| E2 | Pan et al. "DeepOPF: A Feasibility-Optimized Deep Neural Network Approach for DC Optimal Power Flow Problems." arXiv:1910.14448. 2020. | https://arxiv.org/abs/1910.14448 | High: published IEEE IoT Journal |
| E3 | Pan et al. "DeepOPF: AC-OPF feasibility-optimized." arXiv:2007.01002. 2022. | https://arxiv.org/abs/2007.01002 | High: published |
| E4 | Pan et al. "DeepOPF+: Towards an End-to-End DC-OPF Solution." arXiv:2009.03147. 2020. | https://arxiv.org/abs/2009.03147 | High: published |
| E5 | Pan et al. "DeepOPF-V." arXiv:2103.11793. 2021. | https://arxiv.org/abs/2103.11793 | High: published |
| E6 | Chen, Tanneau, Van Hentenryck. "End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch (E2ELR)." IEEE TPWRS 2023. arXiv:2304.11726. | https://arxiv.org/abs/2304.11726 | High: IEEE TPWRS peer-reviewed |
| E7 | Fioretto, Mak, Van Hentenryck. "Predicting AC Optimal Power Flows: Combining Deep Learning and Lagrangian Dual Methods." AAAI 2020. arXiv:1909.10461. | https://arxiv.org/abs/1909.10461 | High: AAAI peer-reviewed |
| E8 | Fioretto et al. "Lagrangian Duality for Constrained Deep Learning." ECML-PKDD 2020. arXiv:2001.09394. | https://arxiv.org/abs/2001.09394 | High: ECML-PKDD peer-reviewed |
| E9 | Park & Van Hentenryck. "Self-Supervised Primal-Dual Learning for Constrained Optimization (PDL)." AAAI 2023. arXiv:2208.09046. | https://arxiv.org/abs/2208.09046 | High: AAAI peer-reviewed |
| E10 | Pagnier, Ferrando, Dvorkin, Chertkov. "Machine Learning for Electricity Market Clearing." IREP 2022. arXiv:2205.11641. | https://arxiv.org/abs/2205.11641 | High: active set prediction → LMP derivation |
| E11 | Ferrando, Pagnier, Mieth et al. "Physics-Informed ML for NYISO." 2023. arXiv:2304.00062. | https://arxiv.org/abs/2304.00062 | High: follows up IREP 2022 with 1,814-bus NYISO |
| E12 | Liu, Wu, Zhu. "Graph Neural Networks for Learning Real-Time Prices in Electricity Market." 2021. arXiv:2106.10529. | https://arxiv.org/abs/2106.10529 | Medium-High: GNN for LMP spatial locality |
| E13 | Jami, Kardoš, Schenk, Köstler. "AI Driven Near Real-time Locational Marginal Pricing Method." 2023. arXiv:2306.10080. | https://arxiv.org/abs/2306.10080 | Medium: preprint; direct regression 5-6% LMP error, 4-5 OOM speedup |

---

## F. Warm Starts and Differentiable Optimization

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| F1 | Suri, Hilmarsson, Bose. "WARP: A Benchmark for Primal-Dual Warm-Starting of Interior-Point Solvers." arXiv:2605.05728. May 2026. | https://arxiv.org/abs/2605.05728 | High: 76% iteration reduction; critical dual warm-start finding |
| F2 | Taheri & Molzahn. "Not All Warm Starts Help: Benchmarking Primal-Dual Initializations for ACOPF Algorithms." arXiv:2606.08984. Jun 2026. | https://arxiv.org/abs/2606.08984 | High: full content verified; 19 instances 5–30,000 buses; 47.6% speedup with full primal+dual; 12/14 partial = negative speedup |
| F3 | Diehl. "Warm-Starting AC Optimal Power Flow with Graph Neural Networks." NeurIPS 2019 Climate Change AI workshop. | https://www.climatechange.ai/papers/neurips2019/1 | Medium: workshop paper; 2.8× speedup on Texas grid |
| F4 | Amos & Kolter. "OptNet: Differentiable Optimization as a Layer in Neural Networks." ICML 2017. arXiv:1703.00443. | https://arxiv.org/abs/1703.00443 | High: ICML peer-reviewed; foundational KKT backprop |
| F5 | Agrawal et al. "Differentiating through a Cone Program (diffcp)." JANO 2019. | https://github.com/cvxgrp/diffcp | High: peer-reviewed; foundational |
| F6 | Agrawal, Amos, Barratt, Boyd, Diamond, Kolter. "Differentiable Convex Optimization Layers (cvxpylayers)." NeurIPS 2019. arXiv:1910.12430. | https://arxiv.org/abs/1910.12430 | High: NeurIPS peer-reviewed |

---

## G. Compositional Generalization and N-k Contingencies

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| G1 | Arowolo & Cremer. "Towards Generalization of GNNs for AC OPF (HH-MPNN)." arXiv:2510.06860. Oct 2025, rev Apr 2026. | https://arxiv.org/abs/2510.06860 | High: detailed numerics verified; best cross-topology result |
| G2 | Yamizi et al. "OPF-HGNN: Generalizable Heterogeneous Graph Neural Networks for AC OPF." IEEE ICDCS 2024. arXiv:2403.00892. | https://arxiv.org/html/2403.00892v1 | High: IEEE ICDCS; full paper fetched; code available |
| G3 | "LG-HGNN: Local and Global Message Passing for AC-OPF Solutions." MDPI Applied Sciences. Jan 5, 2026. | https://www.mdpi.com/2571-5577/9/1/18 | High: MDPI peer-reviewed; N-1 generalization |
| G4 | Wu et al. "Universal Graph Convolutional Network (UGCN)." arXiv:2509.08672. Sep 2025. | https://arxiv.org/abs/2509.08672 | Medium: preprint; zero-shot cross-topology claims |
| G5 | Zhang, Karve, Mahadevan. "Operational Risk Quantification of Power Grids Using GNN Surrogates." arXiv:2311.03661. Nov 2023. | https://arxiv.org/abs/2311.03661 | High: published; 250-800× DC-OPF speedup |
| G6 | Zhang et al. (companion). arXiv:2405.07343. May 2024. | https://arxiv.org/html/2405.07343v1 | High: 60,000-90,000× vs. MILP solver |
| G7 | "Residual Correction Models for AC OPF Using DC OPF Solutions." arXiv:2510.16064. Oct 2025. | https://arxiv.org/abs/2510.16064 | Medium: preprint; 25% lower MSE, 13× speedup |
| G8 | "Constraint-Driven Deep Learning for N-k SC-OPF." EPSR 2024. | https://www.sciencedirect.com/science/article/pii/S0378779624005789 | High: EPSR peer-reviewed; LODF-based N-k scaling |
| G12 | Giraud, Nellikath, Vorwerk, Alowaifeer, Chatzivasileiadis. "Neural Networks for AC OPF: Improving Worst-Case Guarantees During Training." arXiv:2510.23196. PSCC 2026. | https://arxiv.org/abs/2510.23196 | High: PSCC peer-reviewed; 57–793 bus; verification-informed training; first large-scale AC-OPF constraint verification |
| G9 | Hu et al. "Fast and Reliable N-k Contingency Screening with Input-Convex Neural Networks." arXiv:2410.00796. Oct 2024. | https://arxiv.org/abs/2410.00796 | High: 10-20× speedup, zero false negatives |
| G10 | Anrrango et al. "Self-Supervised Learning of Parametric Approx for SC-DC-OPF." arXiv:2601.13486. Jan 2026. | https://arxiv.org/abs/2601.13486 | Medium: preprint |
| G11 | "Gated GNN for AC Power Flow Under Topological Uncertainty." arXiv:2507.02078. Jul 2025. | https://arxiv.org/abs/2507.02078 | Medium: preprint |
| G13 | Misra, Roald, Ng. "Learning for Constrained Optimization: Identifying Optimal Active Constraint Sets." arXiv:1802.09639. INFORMS J. Computing 34(1):463–480, 2022. | https://arxiv.org/abs/1802.09639 | High: peer-reviewed INFORMS; 15 networks 3–1951 buses; only 3 active sets for IEEE 118-bus |
| G14 | Deka & Misra. "Learning for DC-OPF: Classifying active sets using neural nets." IEEE PowerTech 2019. arXiv:1902.05607. | https://arxiv.org/abs/1902.05607 | High: IEEE peer-reviewed; across PGLib-OPF, 4 test cases had >3 active sets; neural net classifier approach |
| G15 | Ventura Nadal & Chevalier. "Scalable Bilevel Optimization for Generating Maximally Representative OPF Datasets (RAMBO)." arXiv:2304.10912. 2023. | https://arxiv.org/abs/2304.10912 | Medium-High: boundary sampling finds 48-53 vs. 0-37 active sets for 118-bus; uniform sampling misses rare regimes |
| G16 | Joswig-Jones, Baker & Zamzam. "OPF-Learn: An Open-Source Framework for Creating Representative OPF Datasets." IEEE ISGT 2022. arXiv:2111.01228. | https://arxiv.org/abs/2111.01228 | High: IEEE peer-reviewed; NREL/OPFLearn.jl; maximizes active set variety in training data |
| G17 | Stratigakos, Pineda, Morales & Kariniotakis. "Interpretable Machine Learning for DC Optimal Power Flow with Feasibility Guarantees." IEEE Trans. Power Systems 39(3):5126–5137, 2024. | https://hal.science/hal-04038380v4 | High: IEEE TPWRS peer-reviewed; prescriptive decision trees encoding binding constraints |
| G18 | Feng, Wu, Lin, You. "Graph World Model." arXiv:2507.10539. Jul 2025. | https://arxiv.org/abs/2507.10539 | Medium: preprint; GNN-based world model for graph-structured state; not tested on power grids |

---

## H. Multi-Period and Expansion Planning with Learning

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| H1 | Kim, Kim, Kim. "MPA-DNN: Projection-Aware Unsupervised Learning for Multi-Period DC-OPF." arXiv:2510.09349. Oct 2025. | https://arxiv.org/abs/2510.09349 | Medium: preprint; multi-period feasibility projection |
| H2 | "RL-Based Transmission Expansion Framework." arXiv:2602.19421. Feb 2026. | https://arxiv.org/html/2602.19421v1 | Medium: preprint; MADDPG 6% cost reduction |
| H3 | Brenner et al. "Learning-Assisted Stochastic Capacity Expansion Planning." arXiv:2401.10451. Jan 2024. | https://arxiv.org/abs/2401.10451 | Medium: preprint; 3.8% cost savings |
| H4 | "Meta-Learning Approach to OPF Under Topology Reconfigurations." arXiv:2012.11524. Dec 2020. | https://arxiv.org/abs/2012.11524 | Medium: MAML for OPF topology amortization |

---

## I. Feasibility Restoration Layers

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| I1 | Liang, Chen, Low. "Homeomorphic Projection to Ensure NN Solution Feasibility." JMLR Vol. 25, 2024. | http://jmlr.org/papers/v25/23-1577.html | High: JMLR peer-reviewed; provable feasibility for non-convex AC-OPF |
| I2 | "Refining GNN Predictions Using Flow Matching for OPF." arXiv:2512.11127. Dec 2025. | https://arxiv.org/abs/2512.11127 | Medium-High: 0.07% cost gap, 100% feasibility on IEEE 30-bus |
| I3 | "A Hard-Constrained NN Learning Framework for DC-to-AC OPF." arXiv:2602.06255. Feb 2026. | https://arxiv.org/html/2602.06255 | Medium-High: 40× speedup, 10⁻⁴ violation, PEGASE-9241 |
| I4 | "Learning to Pursue AC OPF Solutions with Feasibility Guarantees." arXiv:2505.22399. May 2025. | https://arxiv.org/html/2505.22399v2 | Medium: preprint; safe gradient flow on 93-node distribution system |

---

## J. Visual World Models (DINO-WM, LeWM)

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| J1 | Zhou, Pan, LeCun, Pinto. "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning." ICML 2025. arXiv:2411.04983. | https://arxiv.org/abs/2411.04983 | High: ICML peer-reviewed; visual only; frozen DINOv2; not applicable to power grids |
| J2 | Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel." arXiv:2603.19312. Mar 2026, v3 Jun 2026. | https://arxiv.org/abs/2603.19312 | High: v3 updated; code at github.com/lucas-maes/le-wm; pixel-only; principle transferable to GNN |

---

---

## K. Iteration 4 New Sources

| # | Citation | URL | Credibility |
|---|----------|-----|-------------|
| K1 | Giraud, Nellikkath, Vorwerk, Alowaifeer, Chatzivasileiadis. "Neural Networks for AC OPF: Improving Worst-Case Guarantees during Training." PSCC 2026. arXiv:2510.23196. Oct 2025. | https://arxiv.org/abs/2510.23196 | High: PSCC peer-reviewed; alpha-CROWN; ≥50% worst-case reduction; 793-bus systems |
| K2 | Tekeler, Zhong, Zhang, Chevalier. "Fast and Certified Bounding of SC-DCOPF via Interval Bound Propagation." arXiv:2511.15624. Nov 2025. | https://arxiv.org/abs/2511.15624 | Medium-High: preprint; IBP for SC-DCOPF; certified gaps <3.98%; scales to 8,316-bus |
| K3 | Nellikkath, Tanneau, Van Hentenryck, Chatzivasileiadis. "Scalable Exact Verification of Optimization Proxies for AC OPF." arXiv:2405.06109. May 2024. | https://arxiv.org/abs/2405.06109 | High: GPU MIP; >1000 buses; order-of-magnitude scale improvement |
| K4 | Nellikkath & Chatzivasileiadis. "Minimizing Worst-Case Violations of Neural Networks for AC Optimal Power Flow." arXiv:2212.10930. 2022. | https://arxiv.org/abs/2212.10930 | High: 39–162 buses; differentiable worst-case training loop |
| K5 | Venzke, Qu, Low, Chatzivasileiadis. "Learning Optimal Power Flow: Worst-Case Guarantees for Neural Network Solutions." arXiv:2006.11029. 2020. | https://arxiv.org/abs/2006.11029 | High: first MILP worst-case guarantee framework for NN-OPF; ≤300 buses |
| K6 | Kim, Kim, Kim. "MPA-DNN: Projection-Aware Unsupervised Learning for Multi-Period DC-OPF." arXiv:2510.09349. Oct 2025. | https://arxiv.org/abs/2510.09349 | Medium-High: preprint; full HTML verified; SOC hard constraint via lower-triangular projection; <0.024% gap; 39-bus 24h |
| K7 | Garcia, LoGiudice, Parker, Bent. "Transient Stability-Constrained OPF: Neural Network Surrogate Models and Pricing Stability." arXiv:2502.01844. Feb 2025. | https://arxiv.org/abs/2502.01844 | Medium-High: preprint; LMPs from KKT of NN-augmented OPF; discriminatory + uniform pricing; LANL/Texas A&M |
| K8 | He et al. "MTS-JEPA: Multi-resolution JEPA for Multivariate Time Series Anomaly Detection." arXiv:2602.04643. Feb 2026. | https://arxiv.org/abs/2602.04643 | Medium: preprint; JEPA applied to multivariate time series; anomaly detection; not power grids |
| K9 | Feng et al. "Learning to Accelerate Distributed ADMM Using Graph Neural Networks." arXiv:2509.05288. Sep 2025. | https://arxiv.org/html/2509.05288v1 | Medium-High: preprint; ADMM-GNN unrolling; directly relevant to zap ADMMLayer acceleration |
| K10 | Chen, Zhao, Tanneau, Van Hentenryck. "Compact Optimality Verification for Optimization Proxies." ICML 2024. arXiv:2405.21023. | https://arxiv.org/abs/2405.21023 | High: ICML peer-reviewed; MIP-based verification with gradient heuristics; large-scale DC-OPF |
| K11 | Sreekumar, Degleris, Rajagopal. "Large-Scale Network Utility Maximization via GPU-Accelerated Proximal Message Passing." arXiv:2509.10722. 2025. (NUMax ruling) | https://arxiv.org/abs/2509.10722 | High: primary source (zap repo); NUMax is convex resource allocation on transport networks, NOT power grid / JEPA related |

---

## Notes on Credibility Assessment

- **High**: peer-reviewed venue (ICLR, NeurIPS, AAAI, IEEE TPWRS, EPSR, JMLR, Joule/Cell) or official organization blog/repo with primary data
- **Medium-High**: well-established preprint with verified numerics; from credible institution
- **Medium**: preprint without full-paper verification; single source; claims from abstract only
