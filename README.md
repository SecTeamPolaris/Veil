# Veil: Online Traffic Camouflage via Deep Reinforcement Learning  
This repository contains the official implementation of **Veil**, an asymmetric packet-block (PB)-level traffic defender designed to evade encrypted network traffic analyzers. The work is based on the paper *"Online Traffic Camouflage against Network Analyzers via Deep Reinforcement Learning"* (TNSM submission, file: `TNSM__Online_Traffic_Camouflage_against_Network_Analyzers_via_Deep_Reinforcement_Learning__R1_-2.pdf`).  


## Overview  
Veil addresses key limitations of existing symmetric traffic defenses (e.g., impractical decoder deployment, high latency) by leveraging:  
- **Asymmetric Architecture**: Deployable on a single network node (no decoder required) while adhering to TCP/IP constraints.  
- **Packet-Block (PB) Perturbation**: Reconstructs traffic at the PB level (sequences of packets delimited by ACKs) to maintain protocol compliance.  
- **Deep Reinforcement Learning**: Uses class-specific BiLSTM-DQNs to model statistical feature distributions, enabling both **Targeted Defense (TD)** and **Untargeted Defense (UD)**.  


## Repository Structure  
```
Veil/
├── model_def.py        # Core model definitions (EnvSimulator, BiLSTM-DQN, Action Processor)
├── train_veil.py       # Training pipeline (PCAP parsing, feature extraction, model training)
├── test_veil.py        # Testing pipeline (traffic perturbation, TD/UD mode, PCAP generation)
├── traffic_data/       # Directory for training PCAPs (one subdir per traffic class)
│   ├── class_0/        # Example: Class 0 traffic PCAPs
│   ├── class_1/        # Example: Class 1 traffic PCAPs
│   └── ...
└── veil_models/        # Auto-generated directory for saved models (output after training)
    ├── env_sim.pth     # Trained Environment Simulator
    ├── scaler.pkl      # Feature scaler (for preprocessing)
    ├── feature_selector.pkl  # Top-20 feature selector
    └── dqn_class_X.pth # Trained DQN for class X (X = 0,1,...)
```


## Prerequisites  
Install dependencies via `pip`:  
```bash
pip install torch==2.0.0 scapy==2.5.0 scikit-learn==1.3.0 joblib==1.3.2 pandas==2.0.3 numpy==1.25.2
```  
- **PyTorch**: For model training/inference (supports CPU/GPU; GPU recommended for faster training).  
- **Scapy**: For PCAP parsing and packet reconstruction.  
- **Scikit-learn**: For feature scaling and selection.  


## Training Guide  
### Step 1: Prepare Training Data  
1. Create a `traffic_data` directory in the root folder.  
2. For each traffic class (e.g., "WeChat", "TIM", "HTTP"), create a subdirectory under `traffic_data` (e.g., `traffic_data/wechat`, `traffic_data/tim`).  
3. Place **PCAP files** (containing valid TCP flows) into their respective class subdirectories.  

   Example structure:  
   ```
   traffic_data/
   ├── wechat/
   │   ├── wechat_flow1.pcap
   │   ├── wechat_flow2.pcap
   │   └── ...
   ├── tim/
   │   ├── tim_flow1.pcap
   │   ├── tim_flow2.pcap
   │   └── ...
   └── http/
       ├── http_flow1.pcap
       └── ...
   ```

### Step 2: Run Training  
Execute `train_veil.py` to start training. The script automatically:  
1. Parses PCAPs into PBs (split by ACK packets).  
2. Extracts Top-20 statistical features (e.g., packet length distribution, inter-arrival time).  
3. Trains the **Environment Simulator** (a DNN that mimics attacker traffic analyzers).  
4. Trains a **class-specific BiLSTM-DQN** for each traffic class.  

```bash
python train_veil.py
```  

### Training Output  
All trained models and preprocessing tools are saved to the `veil_models` directory:  
- `env_sim.pth`: Trained Environment Simulator (provides reward signals for DQN training).  
- `scaler.pkl` / `feature_selector.pkl`: Feature preprocessing tools (reused in testing).  
- `dqn_class_X.pth`: Trained DQN for traffic class `X` (e.g., `dqn_class_0.pth` for "wechat").  


## Testing Guide  
### Step 1: Prepare Test Data  
- Prepare a single PCAP file (e.g., `input_session.pcap`) containing **one TCP traffic session** (the traffic to be perturbed).  


### Step 2: Configure Testing Mode  
Edit `test_veil.py` to set the defense mode and target class:  
| Parameter         | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `defense_mode`     | Set to `"TD"` (Targeted Defense) or `"UD"` (Untargeted Defense).             |
| `target_class`     | For TD mode: Specify the target class index (matches training class subdirs).|
| `input_pcap`       | Path to the test PCAP (e.g., `"input_session.pcap"`).                        |
| `output_pcap`      | Path for the perturbed output PCAP (e.g., `"output_perturbed.pcap"`).        |

Example configuration:  
```python
defense_mode = "TD"  # Use Targeted Defense
target_class = 1     # Perturb traffic to mimic class 1 ("tim")
input_pcap = "input_session.pcap"
output_pcap = "output_perturbed.pcap"
```


### Step 3: Run Testing  
Execute `test_veil.py` to generate the perturbed PCAP:  
```bash
python test_veil.py
```  

### Testing Logic  
1. **Parse Input PCAP**: Converts the test session into PBs using the same logic as training.  
2. **Select DQN Template**:  
   - **TD Mode**: Loads the pre-trained DQN for the specified `target_class`.  
   - **UD Mode**: Predicts the original class of the test traffic, then randomly selects a non-original class DQN.  
3. **PB Reconstruction**: Uses the DQN to reconstruct each PB (adjusts packet length/inter-arrival time) while maintaining TCP/IP compliance.  
4. **Generate Output PCAP**: Sorts perturbed packets by time and saves to `output_pcap`.  


## Key Features  
| Feature                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **TCP/IP Compliance**    | Reconstructs packets with valid SEQ/ACK numbers and checksums (no connection drops). |
| **Targeted Defense**     | Perturbs traffic to be misclassified as a specific target class.            |
| **Untargeted Defense**   | Perturbs traffic to escape its original class (no target specified).         |
| **Real-Time Efficiency** | Asymmetric design eliminates decoder latency; suitable for live networks.    |



## Notes  
- **PCAP Requirements**: Training/testing PCAPs must contain **valid TCP flows** (with ACK packets) to enable PB segmentation.  
- **Class Indexing**: Class indices in `dqn_class_X.pth` match the alphabetical order of subdirectories in `traffic_data`.  
- **GPU Acceleration**: Add `.cuda()` to model definitions in `model_def.py` for GPU training (e.g., `self.dqn = BiLSTM_DQN(...).cuda()`).
