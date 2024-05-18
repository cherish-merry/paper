import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib

# DDoS攻击缓解

matplotlib.rcParams.update({'font.size': 12})  #

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
    pps = ("0.5", "1", "2", "5")
    penguin_means = {
        'Baseline': (35.6, 51.6, 100, 100),
        'Iptable': (27.2, 42.6, 100, 100),
        "XDP": (18.2, 32.1, 42.3, 91.6),
    }
    x = np.arange(len(pps))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    ax.grid(axis='y', alpha=0.5, zorder=1)
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        ax.bar(x + offset, measurement, width, label=attribute, color=cg[multiplier], zorder=2)
        multiplier += 1

    ax.set_ylabel('Cpu Usage (%)')
    ax.set_xticks(x + width * 1.5, pps)
    ax.legend(loc='upper left', ncol=3)
    ax.set_ylim(0, 120)


if __name__ == '__main__':
    fig = plt.figure(tight_layout=True, figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlabel('Packets Per Second (M)')
    cpu_usage(ax1)

    plt.savefig("img/ddos-mitigate.png", dpi=500, format="png")
    # plt.show()
