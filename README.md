# AvataRL+SAC: An Experimental Exploration in Hybrid RL for Language Models

**Experimental Research Project**

This repository documents an ongoing experiment into training language models from scratch using a novel, hybrid reinforcement learning approach. The core of this exploration is to investigate if a secondary, meta-learning agent can iteratively improve a primary language model's architecture during training, and to do so in a computationally feasible manner.

## 1. The Core Technical Challenge

The foundational `avataRL` project demonstrated that a language model could be trained using next-token prediction guided by a reinforcement learning signal. However, this approach appeared to hit a performance plateau.

A logical next step is to introduce a "meta-learning" agent that doesn't just predict the next token, but actively modifies the model's internal machinery to improve its learning capability. The most straightforward way to evaluate such a "surgical" modification would be to:
1.  Clone the entire multi-hundred-megabyte language model.
2.  Apply the surgical change to the clone.
3.  Evaluate the clone's performance.
4.  Discard the clone.

Performing this process for every potential action by a meta-learning agent would be computationally prohibitive, requiring immense GPU memory and time, rendering the experiment impractical on standard hardware.

## 2. Our Experimental Approach: In-Place Attention Surgery

This experiment investigates a more efficient method that bypasses the need for full model cloning. The core of this work, implemented in `hybrid.py`, is a set of technical explorations designed to make this dual-agent approach tractable.

### `SurgicalTheater`: A Lightweight Evaluation Context

The central technical solution explored here is a Python context manager named `SurgicalTheater`. Its function is to drastically reduce the overhead of evaluating a surgical action. Instead of cloning the entire model (~280MB), it operates as follows:
1.  Before a surgical evaluation, it saves a copy of only the weights of the target attention layers (~2.1MB).
2.  It allows the "surgeon" agent to apply its modification directly to the shared, in-place model.
3.  The modified model is used for a quick performance evaluation.
4.  Upon exiting the context, the original attention weights are restored, effectively undoing the surgery.

Initial observations suggest this method is significantly more efficient, reducing the memory footprint of each evaluation by approximately 98% and making a high frequency of surgical evaluations feasible.

### Three-Phase Experimental Design

To isolate the variables and observe the effects of this approach, a three-phase curriculum is used:
*   **Phase 1 - Bootstrapping**: The model is trained using only the baseline GRPO (next-token prediction) algorithm. This establishes a performance baseline and confirms the previously observed "stagnation" point.
*   **Phase 2 - Exclusive Surgery**: The primary GRPO agent is paused. A Soft Actor-Critic (SAC) agent is trained exclusively, its sole objective being to learn how to perform beneficial surgeries. Its reward is based on the quality of longer text generations from the surgically-modified model.
*   **Phase 3 - Joint Exploration**: Both agents are activated. This phase explores the complex dynamics as the GRPO agent fine-tunes the next-token policy while the SAC surgeon simultaneously attempts to optimize the model's underlying attention structure.

## 3. Initial Observations

The first full run of this experiment has been completed. The detailed results and charts are available in `first_run_report.md`. Key initial observations include:
*   The SAC surgeon agent appeared to learn a stable policy during Phase 2, evidenced by the convergence of its action magnitude. This may suggest that the `SurgicalTheater` approach provides a stable enough environment for the agent to learn.
*   During the joint exploration in Phase 3, the primary model's next-token accuracy trended upwards, potentially breaking past the performance plateau observed at the end of Phase 1.
*   The interaction between the two agents in the final phase resulted in significant volatility in the accuracy metrics, indicating a complex dynamic that requires further exploration.

These are preliminary findings from a single run. Further experimentation is required to validate these observations.

## 4. How to Run the Experiment

### Prerequisites
*   Python 3.x
*   PyTorch
*   A `wandb` (Weights & Biases) account for logging.
*   Required packages: `torch`, `numpy`, `wandb`, `tqdm`, `requests`, `transformers`

You can install the Python dependencies using pip:
```bash
pip install torch numpy wandb tqdm requests transformers
```

### Execution
1.  **Log in to W&B**:
    ```bash
    wandb login
    ```
2.  **Run the script**:
    ```bash
    python hybrid.py
    ```
    The script will automatically download the required dataset, initialize the models, and begin the three-phase training experiment.

## 5. License
This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details. 