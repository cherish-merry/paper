import matplotlib.pyplot as plt
import numpy as np

# data from https://allisonhorst.github.io/palmerpenguins/

species = (
    "0.1",
    "0.2",
    "0.5",
    "1"
)

color = {
    "A": (78 / 255, 171 / 255, 144 / 255),
    "B": (142 / 255, 182 / 255, 156 / 255),
    "C": (237 / 255, 221 / 255, 195 / 255),
    "D": (238 / 255, 191 / 255, 109 / 255),
    "E": (217 / 255, 79 / 255, 51 / 255),
    "F": (131 / 255, 64 / 255, 38 / 255)
}

cg = [color["A"], color["D"], color["F"]]


weight_counts = {
    "Baseline": np.array([4.9, 10.6, 25.3, 51.6]),
    "XDP Feature Extraction": np.array([5.4 - 4.9, 12.4 - 10.6, 34.4 - 25.3, 68.2 - 51.6]),
    "XDP DDoS Detection": np.array([6.2 - 5.4, 14.2 - 12.4, 36.1 - 34.4, 70.3 - 68.2]),
}

width = 0.5
idx = 0
fig = plt.figure(tight_layout=True, figsize=(8, 6))
ax = fig.add_subplot()

bottom = np.zeros(len(weight_counts["Baseline"]))

for label, weight_count in weight_counts.items():
    p = ax.bar(species, weight_count, width, label=label, bottom=bottom, color = cg[idx])
    bottom += weight_count
    idx += 1

ax.set_xlabel('Packets Per Second (M)')
ax.set_title("CPU Usage Ablation Experiment")
ax.legend(loc="upper left")
plt.savefig("img/ddos-ablation.png", dpi=500, format="png")
plt.show()
