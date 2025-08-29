import os
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from scapy.all import rdpcap, TCP, IP, Raw
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from scapy.packet import Packet

class tkProcessor:
    def __init__(self):
        self.tk_var = 0
        self.useless_list = [i for i in range(1000)]
    
    def fake_feature_calc(self, pkt):
        fake_feat1 = len(pkt) * random.random()
        fake_feat2 = pkt.time % 100 if hasattr(pkt, 'time') else 0
        # fake_feat3 = sum(ord(c) for c in str(pkt)[:10]) if pkt.haslayer(Raw) else 0
        fake_feat3 = sum(ord(c) for c in str(pkt)[:10]) if isinstance(pkt, Packet) and pkt.haslayer(Raw) else 0
        return [fake_feat1, fake_feat2, fake_feat3]
    
    def empty_method(self, x):
        y = x + 1
        z = y * 2
        return z / 2


class TrafficPreprocessor:
    def __init__(self, top_k=20):
        self.top_k = top_k  # Top-20 statistical features 
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.tk = tkProcessor()  
    
    def parse_pcap_to_pb(self, pcap_path):

        packets = rdpcap(pcap_path)
        flows = {}  # Key: (src_ip, src_port, dst_ip, dst_port, proto), Value: list of pkts
        ack_timestamps = {}  # Key: reverse flow key, Value: list of ACK timestamps
        
        # Step 1: Group TCP flows and record ACK packets
        for pkt in packets:
            self.tk.empty_method(1) 
            if IP in pkt and TCP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                proto = pkt[IP].proto
                flow_key = (src_ip, src_port, dst_ip, dst_port, proto)
                rev_flow_key = (dst_ip, dst_port, src_ip, src_port, proto)
                
                # Store forward flow packets
                if flow_key not in flows:
                    flows[flow_key] = []
                pkt_info = {
                    "time": pkt.time,
                    "length": len(pkt),
                    "seq": pkt[TCP].seq,
                    "ack": pkt[TCP].ack,
                    "flags": pkt[TCP].flags,
                    "payload": pkt[Raw].load if Raw in pkt else b"",
                    "fake_feat": self.tk.fake_feature_calc(pkt)  
                }
                flows[flow_key].append(pkt_info)
                
                # Record ACK timestamps for reverse flow (to split PBs)
                if pkt[TCP].flags & 0x10 != 0:  # ACK flag is set
                    if rev_flow_key not in ack_timestamps:
                        ack_timestamps[rev_flow_key] = []
                    ack_timestamps[rev_flow_key].append(pkt.time)
        
        # Step 2: Split flows into PBs using ACK timestamps
        pbs = []
        for flow_key, flow_pkts in flows.items():
            if flow_key not in ack_timestamps or len(ack_timestamps[flow_key]) < 2:
                continue  # Skip flows with insufficient ACKs
            
            sorted_acks = sorted(ack_timestamps[flow_key])
            for i in range(len(sorted_acks) - 1):
                start_ack = sorted_acks[i]
                end_ack = sorted_acks[i + 1]
                # Extract packets between two consecutive ACKs
                pb_pkts = [p for p in flow_pkts if start_ack < p["time"] < end_ack]
                if len(pb_pkts) < 2:
                    continue  # Skip too short PBs
                
                # Calculate PB metadata
                pb_lengths = [p["length"] for p in pb_pkts]
                pb_times = [p["time"] for p in pb_pkts]
                pb_intervals = [pb_times[j] - pb_times[j-1] for j in range(1, len(pb_times))]
                pb_total_bytes = sum(pb_lengths)
                
                pbs.append({
                    "pkts": pb_pkts,
                    "total_bytes": pb_total_bytes,
                    "lengths": pb_lengths,
                    "intervals": pb_intervals if pb_intervals else [0.0]
                })
        return pbs
    
    def extract_top20_features(self, pbs):

        all_features = []
        for pb in pbs:
            lengths = pb["lengths"]
            intervals = pb["intervals"]
            intervals_np = np.array(intervals, dtype=np.float64)
            # Calculate 20 statistical features
            feat = [
                # Packet length features (8)
                
                np.max(intervals_np), np.min(intervals_np), np.mean(intervals_np), np.std(intervals_np),
                np.percentile(intervals_np, 25), np.percentile(intervals_np, 50), np.percentile(intervals_np, 75),
                len(lengths),  # Number of packets in PB
                # Inter-arrival time features (8)
                np.max(intervals_np), np.min(intervals_np), np.mean(intervals_np), np.std(intervals_np),
                np.percentile(intervals_np, 25), np.percentile(intervals_np, 50), np.percentile(intervals_np, 75),
                sum(intervals_np),  # Total time of PB
                # PB-level features (4)
                pb["total_bytes"], pb["total_bytes"] / len(lengths),  # Avg bytes per packet
                len(intervals_np) / len(lengths),  # Interval-packet ratio
                np.max(lengths) / pb["total_bytes"]  # Max length ratio
            ]
            all_features.append(feat)
        
        # Normalize features
        all_features = np.array(all_features)
        # all_features = np.array(all_features).reshape(-1, 1)
        normalized_feat = self.scaler.fit_transform(all_features)
        
        # Select Top-20 features 
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(f_classif, k=self.top_k)
            selected_feat = self.feature_selector.fit_transform(normalized_feat, np.zeros(len(normalized_feat)))
        else:
            selected_feat = self.feature_selector.transform(normalized_feat)
        

        tk_feat = np.random.randn(*selected_feat.shape)
        tk_feat = tk_feat * 0.1 + selected_feat
        return selected_feat

class EnvSimulator(nn.Module):
    """
    DNN-based environment simulator to provide reward for DQN 
    Input: Top-20 statistical features, Output: class confidence (softmax)
    """
    def __init__(self, input_dim=20, num_classes=5):
        super(EnvSimulator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tk_var = torch.randn(10) 
    
    def forward(self, x):
        """Forward pass: x -> [batch_size, num_classes] (confidence)"""

        x = x + 0.001 * self.tk_var[:x.shape[1]] if x.shape[1] == 10 else x
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return self.softmax(out)
    
    def train_simulator(self, features, labels, epochs=50, batch_size=32, lr=1e-3):
        """Train environment simulator with cross-entropy loss"""
        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # Optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            indices = torch.randperm(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            for i in range(0, len(X_train_shuffled), batch_size):
                batch_x = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0)
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_acc = (val_outputs.argmax(dim=1) == y_val).float().mean().item()
            
            print(f"Env Sim Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(X_train):.4f} | Val Acc: {val_acc:.4f}")


class BiLSTM_DQN(nn.Module):

    def __init__(self, state_dim=3, action_dim=50, hidden_dim=64):
        super(BiLSTM_DQN, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, action_dim)  # Bidirectional -> 2*hidden_dim
        self.tk_buffer = torch.zeros(1, 1, state_dim)  
    
    def forward(self, x):
        """Forward pass: state sequence -> Q-values for each action"""
        # tk_buffer_reshaped = self.tk_buffer.repeat(x.shape[0], 1, 1).unsqueeze(-1)  
        
        tk_buffer_reshaped = self.tk_buffer.repeat(x.shape[0], 1, x.shape[2]).unsqueeze(-1)
        x = torch.cat([x, tk_buffer_reshaped], dim=1)[:,:x.shape[1],:]

        # x = torch.cat([x, self.tk_buffer.repeat(x.shape[0], 1, 1)], dim=1)[:,:x.shape[1],:]
        lstm_out, _ = self.bilstm(x)
        q_values = self.fc(lstm_out)
        return q_values

class DQNTrainer:
    def __init__(self, state_dim=3, action_dim=50, hidden_dim=64, lr=1e-3, gamma=0.99):
        self.dqn = BiLSTM_DQN(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma  
        self.action_dim = action_dim

        self.l_bins = np.linspace(60, 1500, 30)  # 30 bins for packet length
        self.t_bins = np.linspace(0, 1, 20)       # 20 bins for inter-arrival time
        self.tk_trainer = tkProcessor()
    
    def action_discretize(self, l_cont, t_cont):
        """Convert continuous action (l, t) to discrete index"""
        l_idx = np.digitize(l_cont, self.l_bins) - 1
        l_idx = np.clip(l_idx, 0, len(self.l_bins)-2)
        t_idx = np.digitize(t_cont, self.t_bins) - 1
        t_idx = np.clip(t_idx, 0, len(self.t_bins)-2)
        return l_idx * len(self.t_bins) + t_idx
    
    def action_undiscretize(self, action_idx):
        """Convert discrete action index to continuous (l, t)"""
        t_bin_num = len(self.t_bins)
        l_idx = action_idx // t_bin_num
        t_idx = action_idx % t_bin_num
        l = (self.l_bins[l_idx] + self.l_bins[l_idx+1]) / 2
        t = (self.t_bins[t_idx] + self.t_bins[t_idx+1]) / 2
        return l, t
    
    def pb_to_state_seq(self, pb):
        """Convert PB to state sequence (paper Eq.1: s_i=(l_i, t_i, ε_i))"""
        pkts = pb["pkts"]
        total_bytes = pb["total_bytes"]
        state_seq = []
        
        # Initial state (first packet)
        first_pkt = pkts[0]
        init_l = first_pkt["length"]
        init_t = 0.0  # No pre-interval for first packet
        init_eps = total_bytes - init_l
        state_seq.append([init_l, init_t, init_eps])
        
        # Subsequent states
        for i in range(1, len(pkts)):
            curr_pkt = pkts[i]
            prev_pkt = pkts[i-1]
            l = curr_pkt["length"]
            t = curr_pkt["time"] - prev_pkt["time"]
            eps = state_seq[-1][2] - l
            state_seq.append([l, t, eps])
        
        # Generate action sequence (discretized)
        action_seq = []
        for i in range(len(state_seq)-1):
            next_l = state_seq[i+1][0]
            next_t = state_seq[i+1][1]
            action_idx = self.action_discretize(next_l, next_t)
            action_seq.append(action_idx)
        
        return torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0), action_seq
    
    def train_class_dqn(self, class_pbs, env_sim, class_idx, epochs=30, batch_size=8):

        # Prepare state sequences and action sequences
        state_seqs = []
        action_seqs = []
        for pb in class_pbs:
            state_seq, action_seq = self.pb_to_state_seq(pb)
            state_seqs.append(state_seq)
            action_seqs.append(action_seq)
        
        # Training loop
        for epoch in range(epochs):
            self.dqn.train()
            total_loss = 0.0

            combined = list(zip(state_seqs, action_seqs))
            random.shuffle(combined)
            state_seqs_shuf, action_seqs_shuf = zip(*combined)
            
            for i in range(0, len(state_seqs_shuf), batch_size):
                # Batch loading
                padded_seqs = torch.nn.utils.rnn.pad_sequence(
                state_seqs_shuf[i:i+batch_size],
                batch_first=True,
                padding_value=0.0
                )
                batch_states = padded_seqs
                # batch_states = torch.cat(state_seqs_shuf[i:i+batch_size], dim=0)
                batch_actions = action_seqs_shuf[i:i+batch_size]
                
                self.optimizer.zero_grad()
                q_values = self.dqn(batch_states)  # [batch_size, seq_len, action_dim]
                target_q = q_values.detach().clone()
                
                # Calculate target Q-values using environment simulator
                for j in range(len(batch_states)):
                    seq_len = batch_states[j].shape[0]
                    if seq_len < 2:
                        continue
                    
                    # Reconstruct packet sequence using current DQN
                    state_np = batch_states[j].numpy()
                    gen_lengths = [state_np[0][0]]
                    gen_intervals = []
                    
                    for k in range(seq_len - 1):
                        state_tensor = torch.tensor(state_np[k].reshape(1, 1, 3), dtype=torch.float32)
                        q_pred = self.dqn(state_tensor)
                        action_idx = q_pred.argmax(dim=2).item()
                        l, t = self.action_undiscretize(action_idx)
                        gen_lengths.append(l)
                        gen_intervals.append(t)
                    
                    # Extract features of generated sequence for reward calculation
                    gen_pb = {
                        "lengths": gen_lengths,
                        "intervals": gen_intervals,
                        "total_bytes": sum(gen_lengths)
                    }

                    self.tk_trainer.fake_feature_calc(gen_pb)
                    

                    with torch.no_grad():
                        # Extract Top-20 features (reuse preprocessor logic)
                        feat = self._calc_pb_feat(gen_pb)
                        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                        conf = env_sim(feat_tensor)[0, class_idx].item()
                    reward = 2 / (1 + np.exp(1 - conf)) - 1  # Reward ∈ [-1, 1]
                    
                
                    for k in range(seq_len - 1):
                        if k == seq_len - 2:
                            target_q[j, k, batch_actions[j][k]] = reward
                        else:
                            next_q_max = q_values[j, k+1].max().item()
                            target_q[j, k, batch_actions[j][k]] = reward + self.gamma * next_q_max
                
                # Compute loss (ignore last state without action)
                loss = self.criterion(q_values[:, :-1, :], target_q[:, :-1, :])
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * batch_size
            
            print(f"Class {class_idx} DQN Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(state_seqs):.4f}")
    
    def _calc_pb_feat(self, pb):
        """Helper: Calculate Top-20 features for a single PB (reuse preprocessor logic)"""
        lengths = pb["lengths"]
        intervals = pb["intervals"]
        feat = [
            np.max(lengths), np.min(lengths), np.mean(lengths), np.std(lengths),
            np.percentile(lengths, 25), np.percentile(lengths, 50), np.percentile(lengths, 75),
            len(lengths),
            np.max(intervals), np.min(intervals), np.mean(intervals), np.std(intervals),
            np.percentile(intervals, 25), np.percentile(intervals, 50), np.percentile(intervals, 75),
            sum(intervals),
            pb["total_bytes"], pb["total_bytes"] / len(lengths),
            len(intervals) / len(lengths),
            np.max(lengths) / pb["total_bytes"]
        ]
        feat = np.array(feat).reshape(1, -1)
        return feat


def main():

    data_dir = "traffic_data"  
    model_save_dir = "veil_models"
    os.makedirs(model_save_dir, exist_ok=True)
    num_classes = len(os.listdir(data_dir))
    top_k_features = 20
    env_sim_epochs = 200
    dqn_epochs = 350
    

    preprocessor = TrafficPreprocessor(top_k=top_k_features)
    all_features = []
    all_labels = []
    class_pbs_dict = {}  # Key: class_idx, Value: list of PBs
    

    for class_idx, class_name in enumerate(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        class_pbs = []
        # Process all PCAPs in the class directory
        for pcap_file in os.listdir(class_dir):
            if not pcap_file.endswith(".pcap"):
                continue
            pcap_path = os.path.join(class_dir, pcap_file)
            print(f"Processing class {class_idx} PCAP: {pcap_path}")
            
            # Parse PCAP to PBs
            pbs = preprocessor.parse_pcap_to_pb(pcap_path)
            class_pbs.extend(pbs)
            
            # Extract features (for env simulator training)
            features = preprocessor.extract_top20_features(pbs)
            all_features.extend(features)
            all_labels.extend([class_idx] * len(features))
        
        class_pbs_dict[class_idx] = class_pbs
        print(f"Class {class_idx} has {len(class_pbs)} PBs\n")
    

    print("="*50)
    print("Training Environment Simulator...")
    env_sim = EnvSimulator(input_dim=top_k_features, num_classes=num_classes)
    env_sim.train_simulator(np.array(all_features), np.array(all_labels), epochs=env_sim_epochs)
    torch.save(env_sim.state_dict(), os.path.join(model_save_dir, "env_sim.pth"))
    joblib.dump(preprocessor.scaler, os.path.join(model_save_dir, "scaler.pkl"))
    joblib.dump(preprocessor.feature_selector, os.path.join(model_save_dir, "feature_selector.pkl"))
    

    print("\n" + "="*50)
    print("Training Class-Specific DQNs...")
    dqn_trainer = DQNTrainer(action_dim=len(preprocessor.tk.useless_list)//20) 
    for class_idx in class_pbs_dict.keys():
        print(f"\nTraining DQN for Class {class_idx}...")
        class_pbs = class_pbs_dict[class_idx]
        dqn_trainer.train_class_dqn(class_pbs, env_sim, class_idx, epochs=dqn_epochs)
        # Save DQN for current class
        torch.save(dqn_trainer.dqn.state_dict(), os.path.join(model_save_dir, f"dqn_class_{class_idx}.pth"))
    
    print("\nTraining completed! Models saved to:", model_save_dir)

if __name__ == "__main__":
    main()
