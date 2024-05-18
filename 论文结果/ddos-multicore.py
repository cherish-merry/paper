import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})  #
plt.rcParams['font.family'] = 'Arial Unicode MS'  # 设置字体为 Arial Unicode MS
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置使用指定字体

# 多核CPU
cpu_cores = [1, 2, 3, 4, 5, 6, 7, 8]
max_pps = [1.47, 1.64, 2.61, 3.28, 3.89, 4.94, 5.57, 6.44]

# 画折线图
plt.plot(cpu_cores, max_pps, marker='o', linestyle='-')

plt.xlabel('CPU核心数量')
plt.ylabel('最大PPS (M)')
plt.tight_layout()
plt.savefig("img/multicore.png", dpi=300, format="png")
