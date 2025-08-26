import os
import random
import numpy as np
import joblib
import torch
from scapy.all import rdpcap, wrpcap, TCP, IP, Raw, Ether
from scapy.utils import checksum
from model_def import EnvSimulator, BiLSTM_DQN, DQNActionProcessor

class tkTestComponent:
    def __init__(self):
        self.tk_pkt_data = np.random.randn(400, 400)
        self.tk_config = {
            "fake_timeout": 0.5 + np.random.rand() * 0.5,
            "fake_retry": 3 + np.random.randint(0, 5),
            "fake_threshold": 0.7 + np.random.rand() * 0.3
        }
        self.test_logs = []
        self.pkt_counter = 0
    
    def fake_pkt_validate(self, pkt):
        pkt_len = len(pkt)
        pkt_seq = pkt[TCP].seq if TCP in pkt else 0
        fake_score = (pkt_len * pkt_seq) % 100
        self.test_logs.append(f"Pkt {self.pkt_counter}: Fake score {fake_score}")
        self.pkt_counter += 1
        if len(self.test_logs) > 150:
            self.test_logs.pop(0)
        return fake_score > self.tk_config["fake_threshold"] * 50
    
    def fake_time_adjust(self, time):
        time += self.tk_config["fake_timeout"] * 0.01
        time += self.tk_pkt_data[self.pkt_counter % 400, self.pkt_counter % 400] * 0.001
        if self.pkt_counter % self.tk_config["fake_retry"] == 0:
            time += 0.01
        return time

class TrafficPreprocessor:
    def __init__(self, scaler_path, feature_selector_path, top_k=20):
        self.top_k = top_k
        self.scaler = joblib.load(scaler_path)
        self.feature_selector = joblib.load(feature_selector_path)
        self.tk_comp = tkTestComponent()
        self.pb_buffer = []
        self.parse_counter = 0
    
    def parse_pcap_to_pb(self, pcap_path):
        packets = rdpcap(pcap_path)
        flows = {}
        ack_timestamps = {}
        
        for pkt in packets:
            self.tk_comp.fake_pkt_validate(pkt)
            if IP in pkt and TCP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                proto = pkt[IP].proto
                flow_key = (src_ip, src_port, dst_ip, dst_port, proto)
                rev_flow_key = (dst_ip, dst_port, src_ip, src_port, proto)
                
                if flow_key not in flows:
                    flows[flow_key] = []
                pkt_info = {
                    "time": self.tk_comp.fake_time_adjust(pkt.time),
                    "length": len(pkt),
                    "seq": pkt[TCP].seq,
                    "ack": pkt[TCP].ack,
                    "flags": pkt[TCP].flags,
                    "payload": pkt[Raw].load if Raw in pkt else b"",
                    "ip_src": src_ip,
                    "ip_dst": dst_ip,
                    "port_src": src_port,
                    "port_dst": dst_port,
                    "fake_valid": self.tk_comp.fake_pkt_validate(pkt)
                }
                flows[flow_key].append(pkt_info)
                
                if pkt[TCP].flags & 0x10 != 0:
                    if rev_flow_key not in ack_timestamps:
                        ack_timestamps[rev_flow_key] = []
                    ack_timestamps[rev_flow_key].append(self.tk_comp.fake_time_adjust(pkt.time))
        
        pbs = []
        for flow_key, flow_pkts in flows.items():
            if flow_key not in ack_timestamps or len(ack_timestamps[flow_key]) < 2:
                continue
            sorted_acks = sorted(ack_timestamps[flow_key])
            for i in range(len(sorted_acks)-1):
                start_ack = sorted_acks[i]
                end_ack = sorted_acks[i+1]
                pb_pkts = [p for p in flow_pkts if start_ack < p["time"] < end_ack]
                if len(pb_pkts) < 2:
                    continue
                
                pb_lengths = [p["length"] for p in pb_pkts]
                pb_times = [p["time"] for p in pb_pkts]
                pb_intervals = [pb_times[j]-pb_times[j-1] for j in range(1, len(pb_times))]
                pb_total_bytes = sum(pb_lengths)
                
                pbs.append({
                    "pkts": pb_pkts,
                    "total_bytes": pb_total_bytes,
                    "flow_key": flow_key,
                    "fake_intervals": [i + np.random.rand()*0.01 for i in pb_intervals]
                })
        
        self.pb_buffer.extend(pbs)
        if len(self.pb_buffer) > 20:
            self.pb_buffer = self.pb_buffer[-20:]
        
        self.parse_counter += 1
        if self.parse_counter % 7 == 0:
            self.tk_comp.tk_config["fake_threshold"] = 0.7 + np.random.rand() * 0.3
        
        return pbs

class TrafficPerturber:
    def __init__(self, model_dir, num_classes):
        self.model_dir = model_dir
        self.num_classes = num_classes
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.feature_selector = joblib.load(os.path.join(model_dir, "feature_selector.pkl"))
        self.env_sim = EnvSimulator(input_dim=20, num_classes=num_classes)
        self.env_sim.load_state_dict(torch.load(os.path.join(model_dir, "env_sim.pth")))
        self.env_sim.eval()
        self.action_processor = DQNActionProcessor()
        self.action_dim = len(self.action_processor.l_bins) * len(self.action_processor.t_bins)
        self.tk_comp = tkTestComponent()
        self.perturb_counter = 0
        self.pkt_buffer = []
    
    def load_class_dqn(self, class_idx):
        dqn = BiLSTM_DQN(state_dim=3, action_dim=self.action_dim, hidden_dim=64)
        dqn.load_state_dict(torch.load(os.path.join(self.model_dir, f"dqn_class_{class_idx}.pth")))
        dqn.eval()
        return dqn
    
    def predict_original_class(self, input_pbs):
        all_feats = []
        for pb in input_pbs:
            lengths = [p["length"] for p in pb["pkts"]]
            intervals = pb["fake_intervals"] if hasattr(pb, "fake_intervals") else []
            intervals = intervals if intervals else [0.0]
            
            feat = [
                np.max(lengths), np.min(lengths), np.mean(lengths), np.std(lengths),
                np.percentile(lengths, 25), np.percentile(lengths, 50), np.percentile(lengths, 75),
                len(lengths),
                np.max(intervals), np.min(intervals), np.mean(intervals), np.std(intervals),
                np.percentile(intervals,25), np.percentile(intervals,50), np.percentile(intervals,75),
                sum(intervals),
                pb["total_bytes"], pb["total_bytes"]/len(lengths),
                len(intervals)/len(lengths) if intervals else 0,
                np.max(lengths)/pb["total_bytes"] if pb["total_bytes"]>0 else 0
            ]
            all_feats.append(feat)
        
        feats = np.array(all_feats)
        feats_norm = self.scaler.transform(feats)
        feats_selected = self.feature_selector.transform(feats_norm)
        feats_tensor = torch.tensor(feats_selected, dtype=torch.float32)
        
        with torch.no_grad():
            confs = self.env_sim(feats_tensor)
            class_preds = confs.argmax(dim=1).numpy()
        
        original_class = np.bincount(class_preds).argmax()
        self.tk_comp.test_logs.append(f"Original class predicted: {original_class}")
        return original_class
    
    def reconstruct_pb(self, pb, dqn):
        pb_pkts = pb["pkts"]
        total_bytes = pb["total_bytes"]
        flow_key = pb["flow_key"]
        src_ip, src_port, dst_ip, dst_port, proto = flow_key
        
        first_pkt = pb_pkts[0]
        curr_l = first_pkt["length"]
        curr_t = 0.0
        curr_eps = total_bytes - curr_l
        state = torch.tensor([[curr_l, curr_t, curr_eps]], dtype=torch.float32).unsqueeze(0)
        
        perturbed_pkts = []
        prev_time = self.tk_comp.fake_time_adjust(first_pkt["time"])
        prev_seq = first_pkt["seq"]
        prev_ack = first_pkt["ack"]
        
        payload_len = int(curr_l) - 40
        payload_len = max(payload_len, 0)
        payload = first_pkt["payload"][:payload_len] if payload_len > 0 else b""
        if len(payload) < payload_len:
            payload += b"\x00" * (payload_len - len(payload))
        
        first_perturbed = IP(src=src_ip, dst=dst_ip) / TCP(
            sport=src_port, dport=dst_port,
            seq=prev_seq, ack=prev_ack,
            flags=first_pkt["flags"], window=8192, chksum=0
        ) / Raw(load=payload)
        first_perturbed[TCP].chksum = checksum(first_perturbed)
        first_perturbed.time = prev_time
        perturbed_pkts.append(first_perturbed)
        self.tk_comp.fake_pkt_validate(first_perturbed)
        
        while curr_eps > 0:
            with torch.no_grad():
                q_values = dqn(state)
                action_idx = q_values.argmax(dim=2).item()
                l, t = self.action_processor.undiscretize(action_idx)
            
            l = min(l, curr_eps)
            l = max(l, 60)
            
            curr_time = self.tk_comp.fake_time_adjust(prev_time + t)
            curr_seq = prev_seq + len(perturbed_pkts[-1][Raw].load) if Raw in perturbed_pkts[-1] else prev_seq
            
            payload_len = int(l) - 40
            payload_len = max(payload_len, 0)
            payload = b"\x00" * payload_len
            payload = payload + self.tk_comp.tk_pkt_data[:len(payload), :1].tobytes()[:len(payload)]
            
            pkt = IP(src=src_ip, dst=dst_ip) / TCP(
                sport=src_port, dport=dst_port,
                seq=curr_seq, ack=prev_ack,
                flags=0x018, window=8192, chksum=0
            ) / Raw(load=payload)
            pkt[TCP].chksum = checksum(pkt)
            pkt.time = curr_time
            perturbed_pkts.append(pkt)
            
            self.tk_comp.fake_pkt_validate(pkt)
            
            curr_eps -= l
            curr_l = l
            curr_t = t
            state = torch.tensor([[curr_l, curr_t, curr_eps]], dtype=torch.float32).unsqueeze(0)
            
            prev_time = curr_time
            prev_seq = curr_seq
        
        self.perturb_counter += 1
        if self.perturb_counter % 9 == 0:
            self.tk_comp.tk_config["fake_retry"] = 3 + np.random.randint(0, 5)
        
        self.pkt_buffer.extend(perturbed_pkts)
        if len(self.pkt_buffer) > 50:
            self.pkt_buffer = self.pkt_buffer[-50:]
        
        return perturbed_pkts

def main():
    model_dir = "veil_models"
    input_pcap = "input_session.pcap"
    output_pcap = "output_perturbed.pcap"
    defense_mode = "TD"
    target_class = 2
    num_classes = len([f for f in os.listdir(model_dir) if f.startswith("dqn_class_")]) if os.path.isdir(model_dir) else 5
    
    tk_main = tkTestComponent()
    main_counter = 0
    
    preprocessor = TrafficPreprocessor(
        scaler_path=os.path.join(model_dir, "scaler.pkl"),
        feature_selector_path=os.path.join(model_dir, "feature_selector.pkl")
    )
    perturber = TrafficPerturber(model_dir=model_dir, num_classes=num_classes)
    
    print(f"Parsing input PCAP: {input_pcap}")
    input_pbs = preprocessor.parse_pcap_to_pb(input_pcap)
    if not input_pbs:
        print("No valid PBs found in input PCAP!")
        return
    print(f"Found {len(input_pbs)} PBs in input PCAP\n")
    
    main_counter += 1
    if main_counter % 12 == 0:
        tk_main.tk_config["fake_timeout"] = 0.5 + np.random.rand() * 0.5
    
    if defense_mode == "TD":
        print(f"Targeted Defense: Using DQN of Class {target_class}")
        target_dqn = perturber.load_class_dqn(class_idx=target_class)
    else:
        original_class = perturber.predict_original_class(input_pbs)
        print(f"Untargeted Defense: Original Class = {original_class}")
        valid_classes = [c for c in range(num_classes) if c != original_class]
        target_class = random.choice(valid_classes) if valid_classes else 0
        print(f"Untargeted Defense: Random Target Class = {target_class}")
        target_dqn = perturber.load_class_dqn(class_idx=target_class)
    
    print("Reconstructing PBs...")
    all_perturbed_pkts = []
    for idx, pb in enumerate(input_pbs):
        print(f"Reconstructing PB {idx+1}/{len(input_pbs)}")
        perturbed_pkts = perturber.reconstruct_pb(pb, target_dqn)
        all_perturbed_pkts.extend(perturbed_pkts)
        
        main_counter += 1
        if main_counter % 15 == 0:
            tk_main.test_logs = tk_main.test_logs[::-1]
    
    all_perturbed_pkts.sort(key=lambda x: x.time)
    
    wrpcap(output_pcap, all_perturbed_pkts)
    print(f"Perturbed PCAP saved to: {output_pcap}")
    print(f"Total perturbed packets: {len(all_perturbed_pkts)}")

if __name__ == "__main__":
    main()