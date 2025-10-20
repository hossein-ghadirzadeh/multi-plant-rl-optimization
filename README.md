# A Comparative Study of Q-Learning and Policy Gradient for Garden Resource Management

This project provides a comparative study between two Reinforcement Learning (RL) algorithms: **Q-Learning** and **Policy Gradient**. The comparison is conducted within a custom-built, multi-plant garden simulation environment. The primary objective is to evaluate the performance of these agents in learning an optimal strategy for managing limited resources (water) to maximize rewards (harvests).

### Key Features

- **Custom Environment**: A simulated environment named `MultiPlantGardenEnv` where game rules, plant growth, and resource consumption are defined.
- **Two RL Agents**: A complete implementation of two agents:
  1.  `SimpleQLearningAgent`: A classic, table-based approach.
  2.  `PolicyGradientAgent`: A modern, neural network-based approach using PyTorch.
- **Centralized Configuration**: All important project parameters are centralized in a `config.py` file for easy management.
- **Automated Analysis & Visualization**: Helper scripts for analyzing results, plotting learning curves, and visually comparing agent behavior.
- **Comparative Animation Output**: The final output is a video animation that compares the agents' strategies at the beginning and end of their training.

### Project Structure

The project is organized in a modular structure as follows:

```bash
src/
│
├── agents/
│ └── q_learning_agent.py # Q-Learning agent implementation
│ └── policy_gradient_agent.py # Policy Gradient agent implementation
│
├── envs/
│ └── garden_env.py # Garden simulation environment definition
│
├── train/
│ └── train_agents.py # Main training loop logic
│
├── utils/
│ ├── plotting.py # Functions for plotting rewards
│ └── growth_analysis.py # Functions for analyzing plant growth
│
├── visualization/
│ └── animate_garden.py # Script to generate the final animation
│
├── config.py # Central configuration file for hyperparameters
└── main.py # Main entry point for the application
```

---

## Getting Started

### 1. Requirements

- Python 3.8+
- numpy
- matplotlib
- torch
- tqdm

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### 2. Configuration

You can easily modify the main parameters of the experiment via the `src/config.py` file. For example, to change the number of plants in the garden, simply edit the `NUM_PLANTS` and `MAX_WATER_RESOURCES` variables:

```python
# In config.py
NUM_PLANTS = 3 # Change the number of plants to 3
MAX_WATER_RESOURCES = 3 # Change the amount of water available to 3
```

### 3. Train

The entire process of training, analysis, and output generation is handled by a single command. Navigate into the `src` directory and run the main script:

```bash
cd src
python main.py
```

The program will start training the agents, which may take a few minutes. After completion, the outputs will be generated automatically.

---

### Outputs

After a successful run, the following outputs will be generated:

1.  **Learning Curve Plot**: An image file named `plot_<N>plants.png` will be saved in the `src/plots/` directory, comparing the learning progress of the two agents.
2.  **Comparative Animation**: A video file (or GIF) named `garden_growth_<N>.gif` will be created in the `src/plots/` directory, visually demonstrating the agents' behavior at the start and end of training.

---

### Authors and Contact

This project was created by a team of students from Jönköping University's School of Engineering (JTH) for the Reinforcement Learning Course (TFSS25).

For questions, feedback, or collaborations, please feel free to reach out to any of the authors or open an issue on the project's repository.

| Name                    | Email Address            |
| ----------------------- | ------------------------ |
| **Hossein Ghadirzadeh** | `ghmo23az@student.ju.se` |
| **Noor Alsaadi**        | `alno21fp@student.ju.se` |
| **Koray Duzgun**        | `duko23hj@student.ju.se` |
| **Mohamad Alkhaled**    | `almo24jy@student.ju.se` |

**Jönköping University, School of Engineering (JTH)**<br>
_October 2025_
