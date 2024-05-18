#!/usr/bin/python3

from bcc import BPF
import time
import numpy as np
import sys
import gateway
from ctypes import *


def int_to_ip_little_endian(ip_integer):
    # 将小端序的无符号整数转换为点分十进制形式的IP地址
    if not 0 <= ip_integer <= 0xFFFFFFFF:
        raise ValueError("无效的IP地址整数")

    ip_str = ".".join(str((ip_integer >> (8 * i)) & 255) for i in range(4))
    return ip_str

def ip_to_int_little_endian(ip_address):
    # 将点分十进制形式的IP地址转换为小端序的无符号整数
    ip_parts = list(map(int, ip_address.split(".")))
    if len(ip_parts) != 4 or not all(0 <= part <= 255 for part in ip_parts):
        raise ValueError("无效的IP地址格式")

    ip_integer = sum((ip_parts[i] << (8 * i)) for i in range(4))
    return ip_integer



childrenLeft = np.fromfile("rf/childLeft.bin")
childrenRight = np.fromfile("rf/childrenRight.bin")
feature = np.fromfile("rf/feature.bin")
threshold = np.fromfile("rf/threshold.bin", dtype=int)
value = np.fromfile("rf/value.bin")
size = np.fromfile("rf/size.bin")

decide_tree_map = "BPF_ARRAY(child_left, s32," + str(childrenLeft.shape[0]) + ");\n" + \
                  "BPF_ARRAY(child_right, s32," + str(childrenRight.shape[0]) + ");\n" + \
                  "BPF_ARRAY(feature, s32," + str(feature.shape[0]) + ");\n" + \
                  "BPF_ARRAY(threshold, u64," + str(threshold.shape[0]) + ");\n" + \
                  "BPF_ARRAY(value, u32," + str(value.shape[0]) + ");\n" + \
                  "BPF_ARRAY(size, u32," + str(size.shape[0]) + ");\n" + \
                  "BPF_HASH(block_list, u32, u32," + '10000' + ");\n"

with open('xdp.c', 'r', encoding='utf-8') as f:
    program = f.read()
iface = sys.argv[1]
b = BPF(text=decide_tree_map + program)

child_left_table = b.get_table("child_left")
child_right_table = b.get_table("child_right")
feature_table = b.get_table("feature")
threshold_table = b.get_table("threshold")
value_table = b.get_table("value")
size_table = b.get_table("size")
statistic_table = b.get_table("statistic")
exception_table = b.get_table("exception_table")
flow_table = b.get_table("flow_table")
block_list = b.get_table("block_list")
for i in range(childrenLeft.shape[0]):
    child_left_table[i] = child_left_table.Leaf(childrenLeft[i].astype(int))

for i in range(childrenRight.shape[0]):
    child_right_table[i] = child_right_table.Leaf(childrenRight[i].astype(int))

for i in range(feature.shape[0]):
    feature_table[i] = feature_table.Leaf(feature[i].astype(int))

for i in range(threshold.shape[0]):
    threshold_table[i] = threshold_table.Leaf(threshold[i])

for i in range(value.shape[0]):
    value_table[i] = value_table.Leaf(value[i].astype(int))

for i in range(size.shape[0]):
    size_table[i] = value_table.Leaf(size[i].astype(int))

fn = b.load_func("my_program", BPF.XDP)
b.attach_xdp(iface, fn, 0)

print("hit CTRL+C to stop")
pkt_cnt = 0
pkt = 0
flow_cnt = 0
flow = 0
tcp_cnt = 0
tcp = 0
udp_cnt = 0
udp = 0
interval = 5
ddos = {}
ip = {}
while True:
    try:
        statistic_table = b.get_table("statistic")
        for k in statistic_table.keys():
            val = statistic_table.sum(k).value
            i = k.value
            if i == 0:
                pkt = (val - pkt_cnt) / interval
                print("pkt/s:", pkt)
                pkt_cnt = val
            if i == 1:
                tcp = (val - tcp_cnt) / interval
                print("tcp/s:", tcp)
                tcp_cnt = val
            if i == 2:
                udp = (val - udp_cnt) / interval
                print("udp/s:", udp)
                udp_cnt = val
            if i == 3:
                flow = (val - flow_cnt) / interval
                print("flow/s:", flow)
                flow_cnt = val
        for k, v in exception_table.items():
            src = k.src
            if block_list.get(c_uint(src)) is not None:
                continue
            if src in ddos:
                ddos[src] += v.value
            else:
                ddos[src] = v.value
        for k, v in ddos.items():
            print(int_to_ip_little_endian(k), ":", v)
            if v >= 1:
                block_list[c_uint(k)] = c_uint(1)
        gateway.update(pkt, tcp, udp, flow, ddos)
        time.sleep(interval)
    except KeyboardInterrupt:
        print("Removing filter from device")
        break
b.remove_xdp(iface, 0)
