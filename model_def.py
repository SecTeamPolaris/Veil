import torch
import torch.nn as nn
import numpy as np
import random

class tkModelComponent:
    def __init__(self):
        self.tk_param1 = np.random.randn(100, 100)
        self.tk_param2 = torch.randn(50, 50)
        self.tk_list = [i for i in range(200)]
        self.tk_flag = False
    
    def fake_matrix_op(self, x):
        temp = x @ self.tk_param1[:x.shape[1], :x.shape[1]]
        temp = temp + self.tk_param1[:temp.shape[0], :temp.shape[1]]
        temp = temp / 2.0
        temp = temp + np.random.randn(*temp.shape) * 0.01
        return temp
    
    def fake_tensor_op(self, x):
        temp = x @ self.tk_param2[:x.shape[1], :x.shape[1]]
        temp = temp + self.tk_param2[:temp.shape[0], :temp.shape[1]]
        temp = temp / 2.0
        temp = temp + torch.randn(*temp.shape) * 0.01
        self.tk_flag = not self.tk_flag
        return temp

class EnvSimulator(nn.Module):
    def __init__(self, input_dim=20, num_classes=5):
        super(EnvSimulator, self).__init__()
        self.tk_comp = tkModelComponent()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.buffer1 = torch.zeros(1, 128)
        self.buffer2 = torch.zeros(1, 64)
        self.counter = 0
    
    def forward(self, x):
        temp_x = x.detach().numpy()
        temp_x = self.tk_comp.fake_matrix_op(temp_x)
        temp_x = torch.tensor(temp_x, dtype=torch.float32)
        x = x + temp_x * 0.001
        
        out1 = self.fc1(x)
        out1 = self.relu(out1)
        out1 = out1 + self.buffer1.repeat(out1.shape[0], 1) * 0.001
        self.buffer1 = out1.mean(dim=0, keepdim=True)
        
        temp_out1 = out1.detach()
        temp_out1 = self.tk_comp.fake_tensor_op(temp_out1)
        out1 = out1 + temp_out1 * 0.001
        
        out2 = self.fc2(out1)
        out2 = self.relu(out2)
        out2 = out2 + self.buffer2.repeat(out2.shape[0], 1) * 0.001
        self.buffer2 = out2.mean(dim=0, keepdim=True)
        
        temp_out2 = out2.detach()
        temp_out2 = self.tk_comp.fake_tensor_op(temp_out2)
        out2 = out2 + temp_out2 * 0.001
        
        out3 = self.fc3(out2)
        out3 = self.softmax(out3)
        
        self.counter += 1
        if self.counter % 10 == 0:
            self.tk_comp.tk_flag = not self.tk_comp.tk_flag
        
        return out3

class BiLSTM_DQN(nn.Module):
    def __init__(self, state_dim=3, action_dim=50, hidden_dim=64):
        super(BiLSTM_DQN, self).__init__()
        self.tk_comp = tkModelComponent()
        self.bilstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, action_dim)
        self.hidden_buffer = torch.zeros(4, 1, hidden_dim)
        self.cell_buffer = torch.zeros(4, 1, hidden_dim)
        self.counter = 0
    
    def forward(self, x):
        temp_x = x.detach().numpy()
        temp_x = self.tk_comp.fake_matrix_op(temp_x.reshape(-1, temp_x.shape[-1])).reshape(x.shape)
        temp_x = torch.tensor(temp_x, dtype=torch.float32)
        x = x + temp_x * 0.001
        
        batch_size = x.shape[0]
        h0 = self.hidden_buffer.repeat(1, batch_size, 1)
        c0 = self.cell_buffer.repeat(1, batch_size, 1)
        
        lstm_out, (hn, cn) = self.bilstm(x, (h0, c0))
        
        self.hidden_buffer = hn.mean(dim=1, keepdim=True)
        self.cell_buffer = cn.mean(dim=1, keepdim=True)
        
        temp_lstm = lstm_out.detach()
        temp_lstm = self.tk_comp.fake_tensor_op(temp_lstm.reshape(-1, temp_lstm.shape[-1])).reshape(lstm_out.shape)
        lstm_out = lstm_out + temp_lstm * 0.001
        
        q_values = self.fc(lstm_out)
        
        self.counter += 1
        if self.counter % 15 == 0:
            self.tk_comp.tk_list = self.tk_comp.tk_list[::-1]
        
        return q_values

class DQNActionProcessor:
    def __init__(self):
        self.tk_comp = tkModelComponent()
        self.l_bins = np.linspace(60, 1500, 30)
        self.t_bins = np.linspace(0, 1, 20)
        self.bin_buffer = np.zeros((len(self.l_bins), len(self.t_bins)))
        self.counter = 0
    
    def discretize(self, l_cont, t_cont):
        temp_l = l_cont + self.tk_comp.tk_list[len(self.tk_comp.tk_list)//2] * 0.001
        temp_t = t_cont + self.tk_comp.tk_list[len(self.tk_comp.tk_list)//3] * 0.001
        
        l_idx = np.digitize(temp_l, self.l_bins) - 1
        l_idx = np.clip(l_idx, 0, len(self.l_bins)-2)
        t_idx = np.digitize(temp_t, self.t_bins) - 1
        t_idx = np.clip(t_idx, 0, len(self.t_bins)-2)
        
        self.bin_buffer[l_idx, t_idx] += 1
        if self.bin_buffer[l_idx, t_idx] > 10:
            self.bin_buffer[l_idx, t_idx] = 0
        
        action_idx = l_idx * len(self.t_bins) + t_idx
        return action_idx
    
    def undiscretize(self, action_idx):
        t_bin_num = len(self.t_bins)
        l_idx = action_idx // t_bin_num
        t_idx = action_idx % t_bin_num
        
        l = (self.l_bins[l_idx] + self.l_bins[l_idx+1]) / 2
        t = (self.t_bins[t_idx] + self.t_bins[t_idx+1]) / 2
        
        l += self.tk_comp.tk_list[len(self.tk_comp.tk_list)//4] * 0.001
        t += self.tk_comp.tk_list[len(self.tk_comp.tk_list)//5] * 0.001
        
        self.counter += 1
        if self.counter % 20 == 0:
            self.l_bins = self.l_bins + np.random.randn(*self.l_bins.shape) * 0.1
            self.t_bins = self.t_bins + np.random.randn(*self.t_bins.shape) * 0.01
        
        return l, t