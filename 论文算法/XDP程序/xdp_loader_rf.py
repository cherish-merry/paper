#!/usr/bin/python3

from bcc import BPF, utils
import time
import numpy as np
import sys


def print_stats():
    for k in statistic_table.keys():
        val = statistic_table.sum(k).value
        i = k.value
        if i == 0:
            print("packet_num:", val)
        if i == 1:
            print("tcp:", val)
        if i == 2:
            print("udp:", val)
        if i == 3:
            print("flow:", val)
        if i == 4:
            print("flow_end:", val)
        if i == 5:
            print("exception:", val)
    time.sleep(1)
    for k, v in exception_table.items():
        print(k, ":", v)
    print("----------------------------")


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
                  "BPF_ARRAY(size, u32," + str(size.shape[0]) + ");\n"

with open('xdp_rf.c', 'r', encoding='utf-8') as f:
    program = f.read()
iface = sys.argv[1]
num_cpus = len(utils.get_online_cpus())
b = BPF(text=decide_tree_map + program,
        cflags=['-DMAX_CPUS=%s' % str(num_cpus), '-DTREES=%s' % str(27),
                '-DMAX_DEPTH=%s' % str(27)])

child_left_table = b.get_table("child_left")
child_right_table = b.get_table("child_right")
feature_table = b.get_table("feature")
threshold_table = b.get_table("threshold")
value_table = b.get_table("value")
size_table = b.get_table("size")
statistic_table = b.get_table("statistic")
exception_table = b.get_table("exception_table")
flow_table = b.get_table("flow_table")

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
while True:
    try:
        # time.sleep(1)
        print_stats()
    except KeyboardInterrupt:
        print("Removing filter from device")
        break
b.remove_xdp(iface, 0)
