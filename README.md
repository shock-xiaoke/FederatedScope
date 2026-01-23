# FedHera: Towards Drift-Resilient Federated Fine-tuning with Heterogeneous Resources

## Introduction
FedHera is a federated tuning framework for large language models under heterogeneous client resources and tasks. It extends FederatedScope with aggregation, rank allocation, and adaptive parameter-efficient fine-tuning components that support both reasoning and generative workloads.

## Requirements and Dependencies
- Recommended: Python >= 3.10, PyTorch >= 2.x. CUDA-enabled GPU is optional but recommended for speed.
- Install dependencies from the repository root:
  ```bash
  pip install -r requirements.txt
  ```
- HuggingFace access: some models (e.g., Llama series) may be gated. Make sure you have access and are authenticated (e.g., via `huggingface-cli login`). You may also configure `HF_HOME` or `TRANSFORMERS_CACHE` for a custom cache directory.

## Project Structure

The project is organized as follows:

-   `main.py`: The main entry point for running federated training and evaluation. It handles argument parsing and orchestrates the overall workflow.
-   `requirements.txt`: Lists the Python dependencies required to run the project.
-   `environment.yml`: Conda environment file for reproducible dependency management.

-   **`data/`**: This directory is intended to store the datasets used for training and evaluation. The structure within this directory typically follows `data/<dataset_name>/<num_clients>/`, containing the data splits for each client.
-   **`fed_utils/`**: Core components for the federated learning framework.
    -   `adaptive_peft.py`: Utilities for adaptive Parameter-Efficient Fine-Tuning (PEFT) to handle heterogeneous clients.
    -   `client.py`: Implements the client-side logic for training and evaluation.
    -   `client_participation_scheduling.py`: Contains strategies for sampling clients across communication rounds.
    -   `model_aggregation.py`: Server-side logic for aggregating model updates from clients, including FedHera variants.
    -   `rank_allocator.py`: Manages rank allocation for heterogeneous LoRA configurations.
-   **`templates/`**: Stores various prompt templates used for generative tasks (e.g., `alpaca.json`).
-   **`utils/`**: A collection of helper scripts and utility functions.
    -   `preprocess_fedhera_data.py`: A script to prepare and partition datasets into federated client splits.
    -   `callbacks.py`: Callbacks for different stages of the training process.
    -   `prompter.py`: Helper for constructing prompts from templates.


## Data Preparation
Example: generate 20 clients with up to 10,000 WinoGrande examples.
```bash
python utils/preprocess_fedhera_data.py --task winogrande --hf_dataset winogrande --hf_config winogrande_xl --output_root ./data/winogrande --num_clients 20 --max_examples 10000
```

## Running Examples
(Adjust caches and authentication as needed. Full hyperparameters are listed in the paper appendix; flag descriptions are in `main.py`.)

- Reasoning task (WinoGrande + Llama-2-7b):
  ```bash
  python main.py --aggregation fedhera --hetero_mode setting_A --global_model meta-llama/Llama-2-7b-hf --data_path ./data/winogrande --use_atw --eval_protocol auto --eval_answer_only_loss
  ```

- Generative task (Alpaca + Llama-3.2-3B-Instruct, unbiased server aggregation):
  ```bash
  python main.py --aggregation fedhera --hetero_mode setting_A --global_model meta-llama/Llama-3.2-3B-Instruct --data_path ./data/alpaca --use_atw --fedhera_server_agg unbiased
  ```

## Citation
Placeholder for double-blind review; add the final bibliographic entry after acceptance.

## License
This project is distributed under the Apache-2.0 License (see `LICENSE`). License notices from upstream components are retained.

## Acknowledgement
Based on FederatedScope (Apache-2.0).
