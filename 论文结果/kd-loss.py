import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

# 损失函数

color = {
    "A": (78 / 255, 171 / 255, 144 / 255),
    "B": (142 / 255, 182 / 255, 156 / 255),
    "C": (237 / 255, 221 / 255, 195 / 255),
    "D": (238 / 255, 191 / 255, 109 / 255),
    "E": (217 / 255, 79 / 255, 51 / 255),
    "F": (131 / 255, 64 / 255, 38 / 255)
}


def draw(ax, max_depth, log_loss, f1_score, title, x_label):
    ax.set_xlabel(x_label)
    ax.set_ylabel('Macro F1 Score')
    ax.plot(max_depth, f1_score, color=color["E"], label='Macro F1 Score', marker='o')
    ax.tick_params(axis='y')
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.set_ylabel('Log Loss')
    ax2.plot(max_depth, log_loss, color=color["F"], label='Log Loss', marker='s')
    ax2.tick_params(axis='y')

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax.legend(lines, labels, loc='center right')  # 设置图例位置为右中


if __name__ == '__main__':
    max_depth = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    mlp_log_loss = [0.18802103452797322, 0.14833582307006532, 0.11907355749702402, 0.1018083520649502,
                    0.08190884491601934, 0.07427858928908945, 0.07643091224944401, 0.07094491619677036,
                    0.0678486077049675, 0.0705295586439924, 0.0715068197323438, 0.07184926004752039,
                    0.06792873051922142, 0.06708391299534956, 0.07040481485607664, 0.06861812346041298]

    mlp_f1_score = [0.901, 0.93, 0.953, 0.962, 0.969, 0.972, 0.974, 0.975, 0.976, 0.976, 0.977, 0.975, 0.976, 0.976,
                    0.976, 0.976]

    # rf_log_loss = [0.18454338066571035, 0.14438029627125812, 0.11961418625111735, 0.08928917651921281,
    #                0.07394825578262898, 0.07003415793265778, 0.0600220729916865, 0.05972127163161666,
    #                0.05685015488377477, 0.056389472079063495, 0.05825777226143712, 0.057335445915992156,
    #                0.05797414443813048, 0.05934103642847167, 0.056872901167874065, 0.059532989650854254]
    # rf_f1_score = [0.902, 0.929, 0.951, 0.963, 0.969, 0.973, 0.974, 0.974, 0.976, 0.976, 0.975, 0.974, 0.976, 0.975,
    #                0.975, 0.974]

    xgb_log_loss = [0.18435291594276057, 0.14169515809331332, 0.11284844800884423, 0.09090944097087633,
                    0.07358973883511086, 0.06657470098384469, 0.061913718905445225, 0.06055755876486281,
                    0.057966257679007004, 0.058278965861662056, 0.05834776600978298, 0.058410852551668856,
                    0.058328931998710684, 0.05874155581319518, 0.05883944451296475, 0.058817558896942894]

    xgb_f1_score = [0.902, 0.93, 0.951, 0.962, 0.97, 0.972, 0.973, 0.974, 0.976, 0.975, 0.975, 0.976, 0.975, 0.976,
                    0.975, 0.975]

    fig = plt.figure(tight_layout=True, figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    draw(ax1, max_depth, mlp_log_loss, mlp_f1_score, "DT-MLP", 'Max Depth \n (a)')
    draw(ax2, max_depth, xgb_log_loss, xgb_f1_score, "DT-XGB", 'Max Depth \n (b)')
    plt.tight_layout()
    plt.savefig("img/kd-loss.png", dpi=500, format="png")
    # plt.show()
