import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'font.size': 14})  #
plt.rcParams['font.family'] = 'Arial Unicode MS'  # 设置字体为 Arial Unicode MS
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置使用指定字体

# CPU 占用

color = {
    "A": (78 / 255, 171 / 255, 144 / 255),
    "B": (142 / 255, 182 / 255, 156 / 255),
    "C": (237 / 255, 221 / 255, 195 / 255),
    "D": (238 / 255, 191 / 255, 109 / 255),
    "E": (217 / 255, 79 / 255, 51 / 255),
    "F": (131 / 255, 64 / 255, 38 / 255)
}

cg = [color["B"], color["D"], color["E"], color["F"]]

labels = ["Userspace", "Snort", "XDP", "Baseline"]


def parse_cpu_usage_log(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    cpu_data = {"%usr": [], "%soft": [], "%idle": []}
    lines = lines[3:]
    for line in lines:
        parts = line.split()
        if len(parts) == 12 and not parts[1].endswith("CPU") and parts[1]:
            usr = float(parts[2])
            soft = float(parts[7])
            idle = float(parts[11])
            cpu_data["%usr"].append(usr)
            cpu_data["%soft"].append(soft)
            cpu_data["%idle"].append(idle)
    return cpu_data


def plot_cpu_data_subplot(ax, cpu_data, label, color):
    ax.plot(cpu_data, label=label, color=color, linestyle='-')


def plot_cpu_data(ax, data):
    for index, values in enumerate(data):
        ax.fill_between(range(len(values)), 0, values, label=labels[index], color=cg[index], alpha=1)


if __name__ == '__main__':
    us_data = parse_cpu_usage_log("cpu-data/us.log")
    snort_data = parse_cpu_usage_log("cpu-data/snort.log")
    xdp_data = parse_cpu_usage_log("cpu-data/xdp.log")
    base_data = parse_cpu_usage_log("cpu-data/base.log")

    fig = plt.figure(tight_layout=True, figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])

    plot_cpu_data(ax1, [us_data["%usr"], snort_data["%usr"], xdp_data["%usr"], base_data["%usr"]])
    plot_cpu_data(ax2, [us_data["%soft"], snort_data["%soft"], xdp_data["%soft"], base_data["%soft"]])

    # ax1.set_title('CPU Usage Detail')
    ax1.set_ylabel('%Usr')
    ax1.set_xlabel('Time (s) \n (a) 用户态CPU占用率, Baseline和XDP用户态CPU占用率为0')

    ax2.set_ylabel('%Soft')
    ax2.set_xlabel('Time (s) \n (b) 软中断CPU占用率')

    ax1.legend()
    ax2.legend()
    plt.savefig("img/ddos-cpu.png", dpi=500, format="png")
    plt.show()
