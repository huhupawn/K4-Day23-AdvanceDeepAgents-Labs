# Comprehensive Survey on World Models in Artificial Intelligence

## TL;DR
- World models enable AI agents and neural networks to simulate, predict, and reason about environment dynamics internally without relying solely on real-world trial and error [1][2].
- Architectures range from compact Recurrent State Space Models (RSSM) and search-based planners to Joint-Embedding Predictive Architectures (JEPA) and large-scale video generative simulators [3][4][5].
- Recent breakthroughs between 2023 and 2025 leverage spatiotemporal tokenization, latent action discovery, and diffusion transformers to build interactive virtual environments and physical simulators like Genie 3 and Sora 2 [6][4][7].
- Critical challenges persist, including compounding errors in long-horizon rollouts, data-coverage-driven hallucination, and the sim-to-real embodiment gap [8][9][2].

## Background
The conceptual foundation of world models traces back to cognitive science and cybernetics. Kenneth Craik hypothesized in his 1943 work *The Nature of Explanation* that the fundamental feature of neural machinery is its capacity to construct a "small-scale model" of external reality, allowing organisms to test alternative actions and evaluate consequences internally before physical execution [10][11]. This conceptual lineage was prefigured in the 19th-century writings of Hermann von Helmholtz and Ernst Mach, where Mach argued that physics performs data compression to produce compact, economical representations of sensory observations [12]. In artificial intelligence and reinforcement learning, David Ha and Jürgen Schmidhuber formalized recurrent neural network world models in 2018, demonstrating that an agent can learn internal spatial and temporal representations of an environment and evolve its policy entirely within a hallucinated dream environment [1][13]. Mathematically and theoretically, world models formalize transition functions and predictive state dynamics, bridging control theory, model-based reinforcement learning, and modern generative deep learning [3].

## Latent Dynamics and Search-Based Planning Approaches
Model-based reinforcement learning approaches have evolved through distinct architectural families designed to handle high-dimensional observations and long-horizon planning. Latent-dynamics models, epitomized by the Dreamer series (DreamerV1, DreamerV2, and DreamerV3), rely on Recurrent State Space Models (RSSMs) that combine deterministic recurrent units with stochastic categorical variables to learn compressed representations of environment dynamics [14][3]. DreamerV3 demonstrated domain-general robustness across more than 150 diverse benchmarks—including Atari, DeepMind Control, Crafter, and sparse-reward challenges like collecting diamonds in Minecraft—using a single fixed hyperparameter set [14]. 

In contrast, search-based planning approaches such as MuZero integrate learned dynamics models directly with Monte Carlo Tree Search (MCTS), eliminating the requirement to know the true underlying rules of the environment [15][3]. MuZero recursively predicts three core quantities from latent states and actions—future reward, value, and policy logits—and is trained end-to-end to match target transitions across complex board games (Go, Chess, Shogi) and continuous control benchmarks [15]. While RSSM-based models excel at continuous policy imagination and compact state compression, search-based planning models utilize explicit tree search over learned action-transition graphs to achieve strategic depth.

## Representation-Learning and Generative Video Simulators
Complementing latent-dynamics and search-based methods, representation-learning and generative video architectures tackle world simulation through self-supervised prediction and spatiotemporal tokenization [3]. Joint-Embedding Predictive Architectures (JEPA), such as V-JEPA, learn robust abstract visual representations by predicting missing regions directly in representation space rather than pixel space [16]. By utilizing a predictor network conditioned on spatiotemporal masks and a stop-gradient mechanism on the target encoder, JEPA architectures avoid the severe computational overhead and representation collapse associated with pixel-level reconstruction [16].

Concurrently, generative interactive environments such as Genie and its successors represent a paradigm shift towards unlabelled Internet video pretraining [6]. Genie introduced an action-conditioned video world model trained on over 200,000 hours of unlabelled gaming videos, combining a spatiotemporal tokenizer, an autoregressive Transformer dynamics model, and an unsupervised latent action model to discover discrete actions without human action annotations [6]. Recent comprehensive surveys highlight that these generative frameworks unify latent space reasoning with high-fidelity video rendering, paving the way for open-domain generalist simulators [3][17].

## Recent Advances in Scalable Multimodal World Models (2023–2025)
Between 2023 and 2025, world models scaled dramatically in parameter count, resolution, and interactivity, driven by Diffusion Transformers and large-scale video pretraining [18][5]. OpenAI's Sora and Sora 2 established foundational benchmarks for video generative world models by training jointly on variable-duration videos and images using spacetime patches [5][7]. Sora 2 enhanced physical accuracy, realism, and synchronized audio-visual generation, faithfully modeling physical failure states (such as a basketball bouncing off a backboard when missing a shot) rather than morphing reality to achieve success [7].

Concurrently, Google DeepMind's Genie 3 advanced real-time interactive simulation, generating dynamic environments navigable at 24 frames per second at 720p resolution with promptable text-based world events and robust short-term visual memory [4]. Hybrid architectures like Pandora and PAN further integrated pretrained large language models with video diffusion decoders to support free-text actions and long-horizon open-domain simulation across driving, robotics, and virtual gaming worlds [19][20].

## Trends and Open Problems
Despite rapid scaling and impressive visual fidelity, contemporary world models encounter severe fundamental limitations. Autoregressive generative models and decision-coupled systems suffer from compounding multi-step prediction errors over long horizons, leading to kinematic drift, semantic blurring, and the "Hallucinated Value Hypothesis" where agents optimize against systematic model errors [21][22]. Recent studies reveal that hallucination in world models is primarily a data-coverage problem concentrated in low-coverage regions of the state-action space, rather than a purely architectural failure [8]. Runtime predictors and coverage-aware training have been introduced to mitigate perceptual hallucination and scene divergence [8].

Furthermore, investigations into physical law generalization demonstrate that video generation models learn in a "case-based" retrieval manner rather than discovering universal physical laws, exhibiting poor out-of-distribution (OOD) generalization when confronted with novel physical scenarios [23]. In robotics and embodied AI, models like XEWorld reveal that action-conditioned world models often function as 2D visual pattern matchers rather than true physical simulators, struggling with cross-embodiment generalization and suffering catastrophic forgetting when adapting to novel robot kinematics [9]. Overcoming the sim-to-real gap, eliminating agentic hallucination, and establishing rigorous uncertainty-aware evaluation protocols remain critical open challenges for future research [21][2].

## References
[1] Recurrent World Models Facilitate Policy Evolution. web. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] World Models as an Intermediary between Agents and the Real World. web. https://arxiv.org/pdf/2602.00785v1 (2026-02-03)
[3] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. arxiv. https://arxiv.org/abs/2606.00133 (2026-05-28)
[4] Genie 3: A new frontier for world models. web. https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ (2025-08-05)
[5] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[6] Genie: Generative Interactive Environments. arxiv. https://arxiv.org/abs/2402.15391 (2024-02-23)
[7] Sora 2 is here. web. https://openai.com/index/sora-2/ (2025-09-30)
[8] Hallucination in World Models is Predictable and Preventable. web. https://arxiv.org/html/2606.27326 (2026-06-25)
[9] XEWorld: Can Action-Conditioned World Models Generalize to Unseen Robot Embodiments?. web. https://arxiv.org/pdf/2608.05799.pdf (2026-08-06)
[10] The Nature of Explanation (Chapter 5: Hypothesis on the Nature of Thought). web. https://markhuckvale.com/research/hp/Craik_NOE_Chapter_5.pdf (1943-01-01)
[11] Kenneth Craik Wikipedia Entry on Control Theory & Servomechanisms. web. https://en.wikipedia.org/wiki/Kenneth_Craik (2026-10-02)
[12] The Prehistory of the Idea that Thinking is Modelling. web. https://link.springer.com/article/10.1007/s42087-024-00460-z (2025-01-14)
[13] Recurrent World Models Facilitate Policy Evolution. web. https://proceedings.neurips.cc/paper_files/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf (2018-12-04)
[14] Mastering Diverse Control Tasks through World Models. arxiv. https://arxiv.org/abs/2301.04104 (2023-01-10)
[15] MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. arxiv. https://arxiv.org/abs/1911.08265 (2019-11-19)
[16] Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. arxiv. https://arxiv.org/abs/2301.08243 (2023-01-20)
[17] Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond. hf-search. https://huggingface.co/papers/2405.03520 (2024-05-06)
[18] Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. hf-search. https://huggingface.co/papers/2402.17177 (2024-02-27)
[19] Pandora: Towards General World Model with Natural Language Actions and Video States. web. https://arxiv.org/pdf/2406.09455v1 (2024-06-09)
[20] PAN: A World Model for General, Interactable, and Long-Horizon World Simulation. web. https://arxiv.org/html/2511.09057v2 (2025-11-09)
[21] World Models: The Safety Perspective. web. https://arxiv.org/pdf/2411.07690 (2024-11-12)
[22] Safety, Security, and Cognitive Risks of World Models. web. https://arxiv.org/pdf/2604.01346v1 (2026-04-01)
[23] How Far is Video Generation from World Model: A Physical Law Perspective. web. https://raw.githubusercontent.com/mlresearch/v267/main/assets/kang25g/kang25g.pdf (2025-01-01)
