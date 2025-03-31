import os
import shutil
from scapy.all import *
from scapy.utils import PcapReader
import random
import copy as cp
import numpy as np

# Define the protocols to exclude
excluded_protocols = ['FTPS', 'SFTP']
data_base = 'data'
target_data_base = 'targeted_data'
undirected_data = 'data_targeted_undir'
directed_data = 'data_targeted_bidir'

# Construct full paths
undirected_data_path = os.path.join(target_data_base, undirected_data)
directed_data_path = os.path.join(target_data_base, directed_data)

# Create directories if they do not exist
os.makedirs(undirected_data_path, exist_ok=True)
os.makedirs(directed_data_path, exist_ok=True)

# Iterate over the flows
for src_key, src_flows in targeted_flows_dir.items():
    if src_key in excluded_protocols:
        continue

    src_flows_path_undir = os.path.join(undirected_data_path, src_key)
    src_flows_path_dir = os.path.join(directed_data_path, src_key)

    os.makedirs(src_flows_path_undir, exist_ok=True)
    os.makedirs(src_flows_path_dir, exist_ok=True)

    for dst_key, flow_pairs in src_flows.items():
        if dst_key in excluded_protocols:
            continue

        src_dir = os.path.join(data_base, src_key)
        dst_dir_undir = os.path.join(src_flows_path_undir, dst_key)
        dst_dir_dir = os.path.join(src_flows_path_dir, dst_key)

        # Skip if the file already exists
        if os.path.isfile(os.path.join(dst_dir_undir, file)):
            continue

        try:
            original_flow = rdpcap(os.path.join(src_dir, file))

            # Extract IP and MAC addresses
            ip_info = None
            for packet in original_flow:
                if packet['TCP'].flags == 'S':
                    ip_info = (packet['IP'].src, packet['IP'].dst,
                               packet['TCP'].sport, packet['TCP'].dport,
                               packet['Ether'].src, packet['Ether'].dst)
                    break
                elif packet['TCP'].flags == 'SA':
                    ip_info = (packet['IP'].dst, packet['IP'].src,
                               packet['TCP'].dport, packet['TCP'].sport,
                               packet['Ether'].dst, packet['Ether'].src)
                    break

            if ip_info is None:
                raise ValueError('IP information not found')

            # Process the flow
            process_flow(original_flow, ip_info, flow_pairs, dst_dir_undir, dst_dir_dir)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Helper function to process the flow
def process_flow(flow, ip_info, blocks, undir_path, dir_path):
    src_ip, dst_ip, src_port, dst_port, src_mac, dst_mac = ip_info
    reconstructed_blocks = []
    for i in range(len(blocks) - 1):
        current_end, next_start = blocks[i][-2], blocks[i+1][1]
        reconstructed_blocks.extend([(blocks[i], 1), ((current_end + 1, next_start - 1), -1)])
    reconstructed_blocks.extend([(blocks[-1], 1), ((blocks[-1][-2] + 1, len(flow) - 1), -1)])

    # Process each block
    for block_pair in reconstructed_blocks:
        block = block_pair[0]
        if block_pair[1] == 1:
            # Positive direction
            process_block_positive(flow, block, src_ip, dst_ip, src_port, dst_port, src_mac, dst_mac, undir_path, dir_path)
        elif block_pair[1] == -1:
            # Negative direction
            process_block_negative(flow, block, src_ip, dst_ip, src_port, dst_port, src_mac, dst_mac, dir_path)

# Helper function to process positive direction blocks
def process_block_positive(flow, block, src_ip, dst_ip, src_port, dst_port, src_mac, dst_mac, undir_path, dir_path):
    # Implementation of positive direction block processing
    pass

# Helper function to process negative direction blocks
def process_block_negative(flow, block, src_ip, dst_ip, src_port, dst_port, src_mac, dst_mac, dir_path):
    # Implementation of negative direction block processing
    pass