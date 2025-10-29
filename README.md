# Connect 4 RL Solving 🎮🧠

The goal of the repo is to try different RL algorithms to play connect4 (model free). This is an extension to my Roboticc arm pplaying connect4 project, that explores more deeply the different algorithms that can solve the task. 

You can train your own models with customed parameters for each RL algorithm directly in the UI. The purpose of this is to offer an easy way for people to explore RL algorithms, and add their own, following the same structure as DQN example. 

This project is in developpement. For the moment, the algorithms avaible are:
- Double DQN


## 🌟 Core components

- **/scripts/Env.py**: The Connect4 environnement
- **/scripts/Train.py**: Training scripts for algorithms
- **/scripts/Connect4.py**: The basic rules for the connect4 (useful for the env)
- **/scripts/rl_algorithms**: The folder containing the avaible algorithms (in which you can create yours).
- **/graphics**: The folder containing the graphics part
- **Experience Replay**: Efficient learning through stored game experiences
- **Epsilon-Greedy Strategy**: Balanced exploration vs exploitation
- **Model Persistence**: Save and load trained models
- **GUI Interface**: Interactive game interface using Kivy

## 🏗️ Architecture

### Core Components

| File | Description |
|------|-------------|
| `DQN2.py` | Enhanced Deep Q-Network implementation with TensorFlow/Keras |
| `Train2.py` | Training pipeline for multi-agent self-play |
| `Connect4.py` | Core Connect 4 game logic and board management |
| `env.py` | Game environment following OpenAI Gym-style interface |
| `connect4InterfaceNoRobot.py` | GUI interface for human vs AI gameplay |

### Neural Network Architecture

```
Input Layer (42 neurons) 
    ↓
One-Hot Encoding (42 → 126 features)
    ↓ 
Dense Layer (n_neurons × 3)
    ↓
Batch Normalization → ReLU → Dropout (0.2)
    ↓
[Hidden Layers] × (n_layers - 2)
    ↓
Dense Layer (n_neurons)
    ↓
Batch Normalization → ReLU
    ↓
Output Layer (42 neurons, softmax)
```

## 🚀 Getting Started

### Prerequisites

```bash
# Required packages
tensorflow>=2.16.1
numpy
kivy  # For GUI interface
collections  # For replay buffer
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd connect_4_dqn
   ```

2. **Set up Python environment**
   ```bash
   # Create conda environment (recommended)
   conda create -n connect4_dqn python=3.10
   conda activate connect4_dqn
   
   # Install dependencies
   pip install tensorflow numpy kivy
   ```

3. **Create models directory**
   ```bash
   mkdir -p models
   ```

### Quick Start

#### 1. Test the DQN Implementation
```bash
python DQN2.py
```

#### 2. Train AI Agents
```bash
python Train2.py
```

#### 3. Play Against AI
```bash
python connect4InterfaceNoRobot.py
```

## 🎯 Usage

### Training Configuration

```python
# Example training setup
trainer = Train(
    model_name="my_model",
    learning_rate=0.5e-3,
    discount_factor=0.98,
    eps=0.5,  # Initial epsilon for exploration
    reset=False  # Set to True to start fresh training
)

# Train for n games
trainer.train_n_games(1000)
```

### DQN Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_layers` | 2 | Number of hidden layers |
| `n_neurons` | 32 | Neurons per hidden layer |
| `learning_rate` | 1e-2 | Adam optimizer learning rate |
| `gamma` | 1e-1 | Discount factor for future rewards |
| `eps` | 0.9 | Initial epsilon for ε-greedy strategy |
| `batch_size` | 32 | Experience replay batch size |

## 🧠 Deep Q-Learning Details

### 🎯 Learning Process Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Game State    │───▶│   DQN Agent     │───▶│     Action      │
│                 │    │                 │    │                 │
│  [ 0 0 0 0 0 0 0] │    │  Neural Network │    │   Column: 3     │
│  [ 0 0 0 0 0 0 0] │    │                 │    │                 │
│  [ 0 0 0 0 0 0 0] │    │ Q-Values for    │    │                 │
│  [ 0 0 0 1 0 0 0] │    │ each column     │    │                 │
│  [ 0 0 2 1 0 0 0] │    │                 │    │                 │
│  [ 0 1 2 1 2 0 0] │    │ [0.1,0.3,0.2,   │    │                 │
└─────────────────┘    │  0.8,0.1,0.2,0.1]│    └─────────────────┘
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Experience    │
                       │     Storage     │
                       │                 │
                       │ (state, action, │
                       │ reward, next_   │
                       │ state, done)    │
                       └─────────────────┘
```

### 🎮 Self-Play Training Cycle

#### Step 1: Initial Game State
```
Connect 4 Board (Empty):
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │ ← Row 5
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │ ← Row 4
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │ ← Row 3
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │ ← Row 2
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │ ← Row 1
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │ ← Row 0
└───┴───┴───┴───┴───┴───┴───┘
  0   1   2   3   4   5   6   ← Columns
```

#### Step 2: Player 1 (Red) Makes Move
```
Agent 1 Decision Process:
┌─────────────────┐
│ Current State:  │
│ [0,0,0,0,0,0,0, │ ← Flattened board (42 elements)
│  0,0,0,0,0,0,0, │
│  0,0,0,0,0,0,0, │
│  0,0,0,0,0,0,0, │
│  0,0,0,0,0,0,0, │
│  0,0,0,0,0,0,0] │
└─────────────────┘
         │
         ▼ (Neural Network Processing)
┌─────────────────┐
│ Q-Values:       │
│ Col 0: 0.12     │ ← Low probability
│ Col 1: 0.08     │
│ Col 2: 0.15     │
│ Col 3: 0.25     │ ◄── HIGHEST! Choose this
│ Col 4: 0.18     │
│ Col 5: 0.11     │
│ Col 6: 0.11     │
└─────────────────┘
         │
         ▼ (Action: Drop in Column 3)
         
Result Board:
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │ 1 │   │   │   │ ← Player 1 piece
└───┴───┴───┴───┴───┴───┴───┘
  0   1   2   3   4   5   6
```

#### Step 3: Player 2 (Yellow) Responds
```
Agent 2 sees updated board and decides:
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │ 2 │   │   │   │ ← Player 2 blocks
└───┴───┴───┴───┴───┴───┴───┘
                │
                ▼ (Strategic blocking move)
```

#### Step 4: Learning from Outcomes
```
Game Progression Example:

Move 1:           Move 5:           Final State:
┌─────────┐      ┌─────────┐       ┌─────────┐
│    1    │      │ 2   1   │       │ 2 1 1 2 │ ← Player 1 WINS!
│    2    │ ───▶ │ 1 2 2 1 │ ───▶  │ 1 2 2 1 │   (4 in a row)
└─────────┘      │ 2 1 1 2 │       │ 2 1 1 2 │
                 └─────────┘       └─────────┘

Experience Storage:
┌─────────────────────────────────────────────────────────┐
│ State₁ → Action₁ → Reward₁ → State₂ → Done             │
│ [0,0,0,1,0,0,0...] → 3 → +1.0 → [final] → True        │ ← Win!
│                                                         │
│ State₁ → Action₁ → Reward₁ → State₂ → Done             │
│ [0,0,0,2,0,0,0...] → 3 → -1.0 → [final] → True        │ ← Loss!
└─────────────────────────────────────────────────────────┘
```

### 🔄 Experience Replay & Learning

```
Training Batch (Random Sample from Memory):
┌─────────────────────────────────────────────────────────────┐
│ Experience 1: [state] → action: 3 → reward: +1.0 → [next]  │
│ Experience 2: [state] → action: 1 → reward: -0.1 → [next]  │
│ Experience 3: [state] → action: 4 → reward: +0.5 → [next]  │
│ Experience 4: [state] → action: 2 → reward: -1.0 → [next]  │
│                           ...                               │
│ Experience 32: [state] → action: 6 → reward: +0.0 → [next] │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (Batch Learning)
┌─────────────────────────────────────────────────────────────┐
│                    Q-Learning Update                        │
│                                                             │
│ Target Q-Value = Reward + γ × max(Q_next_state)           │
│                                                             │
│ Current Q-Value = Neural_Network(current_state)[action]     │
│                                                             │
│ Loss = MSE(Target Q-Value, Current Q-Value)                │
│                                                             │
│ Backpropagation: Update network weights to minimize loss   │
└─────────────────────────────────────────────────────────────┘
```

### 🎯 Epsilon-Greedy Strategy Evolution

```
Training Progress:

Episode 1 (ε = 0.9):           Episode 500 (ε = 0.3):         Episode 1000 (ε = 0.1):
┌─────────────────┐           ┌─────────────────┐            ┌─────────────────┐
│ 90% Random      │────────▶ │ 30% Random      │─────────▶  │ 10% Random      │
│ 10% Best Action │           │ 70% Best Action │            │ 90% Best Action │
│                 │           │                 │            │                 │
│ Exploration     │           │ Balanced        │            │ Exploitation    │
│ Heavy Learning  │           │ Learning        │            │ Optimal Play    │
└─────────────────┘           └─────────────────┘            └─────────────────┘

Random Move Example:          Neural Network Move:           Expert Move:
┌───┬───┬───┬───┬───┬───┬───┐ ┌───┬───┬───┬───┬───┬───┬───┐  ┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │ │   │   │   │   │   │   │   │  │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │ │   │   │   │   │   │   │   │  │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │ │   │   │   │   │   │   │   │  │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │ 2 │   │   │   │ │   │   │   │ 2 │   │   │   │  │   │   │   │ 2 │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │ 1 │   │   │   │ │   │   │   │ 1 │   │   │   │  │   │   │   │ 1 │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┼───┤
│   │ 1 │   │ 2 │   │   │   │ │   │   │ 1 │ 2 │   │   │   │  │   │   │   │ 2 │ 1 │   │   │
└───┴───┴───┴───┴───┴───┴───┘ └───┴───┴───┴───┴───┴───┴───┘  └───┴───┴───┴───┴───┴───┴───┘
  Random move: Col 1 (bad)     Smart move: Col 2 (good)      Expert: Col 4 (blocks win!)
```

### 🧮 Neural Network Processing Flow

```
Input Processing:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Connect 4 Board │───▶│   One-Hot       │───▶│   Flattened     │
│                 │    │   Encoding      │    │   Vector        │
│ [0,1,2,0,1,2,0, │    │                 │    │                 │
│  0,0,0,0,0,0,0, │    │ 0→[1,0,0]      │    │ [1,0,0,0,1,0,   │
│  0,0,0,0,0,0,0, │    │ 1→[0,1,0]      │    │  0,0,1,1,0,0,   │
│  0,0,0,0,0,0,0, │    │ 2→[0,0,1]      │    │  0,1,0,0,0,0,   │
│  0,0,0,0,0,0,0, │    │                 │    │  ...           │
│  0,0,0,0,0,0,0] │    │                 │    │  126 features] │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       42 values              3D encoding           126 features

Hidden Layer Processing:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Dense Layer 1   │───▶│ Batch Norm +    │───▶│ Dense Layer 2   │
│                 │    │ ReLU + Dropout  │    │                 │
│ 126 → 96 nodes  │    │                 │    │ 96 → 32 nodes   │
│                 │    │ ┌─────────────┐ │    │                 │
│ W₁ × input + b₁ │    │ │ 20% dropout │ │    │ W₂ × h₁ + b₂   │
│                 │    │ │ (training)  │ │    │                 │
└─────────────────┘    │ └─────────────┘ │    └─────────────────┘
                       └─────────────────┘

Output Generation:
┌─────────────────┐    ┌─────────────────┐
│ Final Dense     │───▶│ Q-Values for    │
│                 │    │ Each Action     │
│ 32 → 7 nodes    │    │                 │
│                 │    │ [Q₀, Q₁, Q₂,   │
│ Softmax/Linear  │    │  Q₃, Q₄, Q₅,   │
│ Activation      │    │  Q₆]            │
│                 │    │                 │
│                 │    │ Action = argmax │
└─────────────────┘    └─────────────────┘
```

### State Representation
- **Board State**: 6×7 grid flattened to 42-element vector
- **Encoding**: 0 (empty), 1 (player 1), 2 (player 2)
- **One-Hot**: Each position expanded to 3-dimensional one-hot vector

### Action Space
- **Actions**: 7 possible column choices (0-6)
- **Invalid Moves**: Handled by environment with negative rewards

### Reward System
- **Win**: +1 reward
- **Loss**: -1 reward  
- **Draw**: 0 reward
- **Invalid Move**: Negative penalty
- **Ongoing**: Small step penalty to encourage faster wins

### Training Process
1. **Self-Play**: Two DQN agents play against each other
2. **Experience Collection**: Store (state, action, reward, next_state, done) tuples
3. **Replay Buffer**: Maintain buffer of recent experiences
4. **Batch Learning**: Sample random batches for training
5. **Target Network**: Separate target network for stable learning

## 📊 Model Performance

### Training Metrics
- **Episode Rewards**: Track cumulative rewards per game
- **Win Rate**: Percentage of games won vs random/previous versions
- **Loss Convergence**: Monitor training loss reduction
- **Epsilon Decay**: Exploration rate reduction over time

### Evaluation
```python
# Evaluate trained model
dqn = DQN("trained_model")
state = np.array([0] * 42)  # Empty board
action_probs = dqn.model.predict(state[np.newaxis])[0]
best_action = np.argmax(action_probs)
```

## 🎮 Game Interface

The GUI interface (`connect4InterfaceNoRobot.py`) provides:
- **Interactive Board**: Click to drop pieces
- **AI Opponent**: Play against trained DQN
- **Visual Feedback**: Real-time game state updates
- **Score Tracking**: Win/loss statistics

## 🔧 Troubleshooting

### Common Issues

**1. TensorFlow Import Errors**
```bash
# Use tf.keras instead of separate keras import
import tensorflow as tf
# All keras functionality via tf.keras.*
```

**2. Model Loading Issues**
```bash
# Ensure custom layers are registered
custom_objects = {'OneHotLayer': OneHotLayer}
model = tf.keras.models.load_model(path, custom_objects=custom_objects)
```

**3. GPU Setup (Optional)**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Performance Tips
- **CPU Training**: Works well for Connect 4 complexity
- **Batch Size**: Increase for faster training (if memory allows)
- **Learning Rate**: Lower for more stable convergence
- **Replay Buffer**: Larger buffer for more diverse experiences

## 📈 Future Improvements

- [ ] **Advanced Architectures**: Convolutional layers for spatial awareness
- [ ] **Tournament Play**: Multi-agent tournaments for robust evaluation  
- [ ] **Opening Book**: Pre-computed optimal opening moves
- [ ] **Alpha-Beta Integration**: Hybrid AI with traditional game tree search
- [ ] **Web Interface**: Browser-based gameplay
- [ ] **Model Compression**: Smaller models for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepMind**: Original DQN paper and methodology
- **OpenAI Gym**: Environment interface inspiration
- **TensorFlow/Keras**: Deep learning framework
- **Connect 4 Community**: Game rules and strategy insights

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **Project Link**: [https://github.com/yourusername/connect_4_dqn](https://github.com/yourusername/connect_4_dqn)

---

*Built with ❤️ and lots of ☕ for the love of AI and classic games*
