import matplotlib.pyplot as plt
from matplotlib import gridspec
# 最大深度影响

color = {
    "A": (78 / 255, 171 / 255, 144 / 255),
    "B": (142 / 255, 182 / 255, 156 / 255),
    "C": (237 / 255, 221 / 255, 195 / 255),
    "D": (238 / 255, 191 / 255, 109 / 255),
    "E": (217 / 255, 79 / 255, 51 / 255),
    "F": (131 / 255, 64 / 255, 38 / 255)
}

if __name__ == '__main__':
    max_depth = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    dt_leafs = [31, 55, 84, 116, 158, 210, 273, 337, 402, 463, 537, 612, 687, 762, 835, 901]
    rdt_leafs = [30, 53, 84, 121, 160, 210, 263, 314, 382, 426, 455, 540, 569, 603, 654, 694]
    mdt_leafs = [28, 54, 76, 95, 130, 169, 196, 241, 248, 325, 333, 348, 353, 451, 488, 496]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Tree Leafs')
    ax.plot(max_depth, dt_leafs, label="DT", color=color["A"], marker='o')
    ax.plot(max_depth, rdt_leafs, label="DT(RF)", color=color["D"], marker='s')
    ax.plot(max_depth, mdt_leafs, color=color["F"], marker='x', label="DT(MLP)")
    ax.legend()

    # plt.savefig("img/kd-loss.png", dpi=500, format="png")
    plt.show()
