import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'font.size': 12})  #
plt.rcParams['font.family'] = 'Arial Unicode MS'  # 设置字体为 Arial Unicode MS
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置使用指定字体

# CPU占用率和丢包率

color = {
    "A": (78 / 255, 171 / 255, 144 / 255),
    "B": (142 / 255, 182 / 255, 156 / 255),
    "C": (237 / 255, 221 / 255, 195 / 255),
    "D": (238 / 255, 191 / 255, 109 / 255),
    "E": (217 / 255, 79 / 255, 51 / 255),
    "F": (131 / 255, 64 / 255, 38 / 255)
}

cg = [color["A"], color["D"], color["F"], color["C"]]


def cpu_usage(ax):
    pps = ("0.1", "0.2", "0.5", "1")
    penguin_means = {
        'Userspace': (100, 100, 100, 100),
        'Snort': (63.4, 100, 100, 100),
        "XDP": (6.2, 14.2, 36.1, 70.3),
        "Baseline": (4.9, 10.6, 25.3, 51.6),
    }
    x = np.arange(len(pps))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    ax.grid(axis='y', alpha=0.5, zorder=1)
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=cg[multiplier], zorder=2)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Cpu Usage (%)')
    ax.set_xlabel('Packets Per Second (M) \n (a) CPU占用率')
    # ax.set_title('Benchmark Performance')
    ax.set_xticks(x + width * 1.5, pps)
    ax.legend(loc='upper left', ncol=4)
    ax.set_ylim(0, 120)


def drop_rate(ax):
    pps = ("0.1", "0.2", "0.5", "1")
    penguin_means = {
        'Userspace': (27.8, 60.2, 86.8, 94.9),
        'Snort': (0, 43.9, 85.4, 92.8),
        "XDP": (0, 0, 0, 0),
        "Baseline": (0, 0, 0, 0),
    }

    x = np.arange(len(pps))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    ax.grid(axis='y', alpha=0.5, zorder=1)
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=cg[multiplier], zorder=2)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Drop Rate (%)')
    ax.set_xlabel('Packets Per Second (M) \n (b) 丢包率, Baseline和XDP场景下未丢包')
    ax.set_xticks(x + width * 1.5, pps)
    ax.legend(loc='upper left', ncol=4)
    ax.set_ylim(0, 120)


if __name__ == '__main__':
    fig = plt.figure(tight_layout=True, figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, :])
    cpu_usage(ax1)
    ax2 = fig.add_subplot(gs[1, 0])
    drop_rate(ax2)
    plt.savefig("img/ddos-performance.png", dpi=500, format="png")
    # plt.show()
