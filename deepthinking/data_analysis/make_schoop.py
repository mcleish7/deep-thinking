""" make_schoop.py
    For generating schoopy plots

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import argparse
from datetime import datetime

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from make_table import get_table

def get_schoopy_plot_no_alpha_lines(table, error_bars=True):

    if error_bars and "test_acc_sem" in table.keys():
        print("trigged if at 1")
    print(table.head(10))
    conditions = [
        (table['model']=="dt_net_1d_width=400") & (table['alpha']==0),
        (table['model']=="dt_net_1d_width=400")& (table['alpha']==1),
        (table['model']=="dt_net_2d_width=128") & (table['alpha']==0),
        (table['model']=="dt_net_2d_width=128")& (table['alpha']==1),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==1),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==1),
        (table['model']=="feedforward_net_1d_width=400"),
        (table['model']=="feedforward_net_2d_width=128"),
        (table['model']=="feedforward_net_recall_1d_width=400"),
        (table['model']=="feedforward_net_recall_2d_width=128")
    ]
    
    values = ["dt","dt_prog","dt","dt_prog","dt_recall","dt_recall_prog","dt_recall","dt_recall_prog","ff","ff","ff","ff"]
    table["graph_name"] = np.select(conditions, values)
    table = table[(table['graph_name'] == "dt")|(table['graph_name'] == "dt_prog")|(table['graph_name'] == "dt_recall")|(table['graph_name'] == "dt_recall_prog")|(table['graph_name'] == "ff")]
    fig, ax = plt.subplots(figsize=(20, 9))
    print(table.head(10))
    models = set(table.graph_name)
    test_datas = set(table.test_data)
    alphas = set(table.alpha)

    sns.lineplot(data=table,
                 x="test_iter",
                 y="test_acc_mean",
                 hue="graph_name",
                 style="test_data" if len(test_datas) > 1 else None,
                 palette="dark",
                 dashes=True,
                 units=None,
                 legend="auto",
                 ax=ax)

    if error_bars and "test_acc_sem" in table.keys():
        print("trigged if at 2")
        for model in models:
            for test_data in test_datas:
                for alpha in alphas:
                    print("trigged innermost loop")
                    data = table[(table.graph_name == model) &
                                 (table.test_data == test_data) &
                                 (table.alpha == alpha)]
                    plt.fill_between(data.test_iter,
                                    data.test_acc_mean - data.test_acc_sem,
                                    data.test_acc_mean + data.test_acc_sem,
                                    alpha=0.1, color="k")

    tr = table.max_iters.max()  # training regime number
    ax.fill_between([0, tr], [105, 105], alpha=0.3, label="Training Regime")
    return ax

def trigger_if_plot(table, error_bars=True):
    #trying to find where losing shading
    print("in function to trigger")

    conditions = [
        (table['model']=="dt_net_1d_width=400") & (table['alpha']==0.0),
        (table['model']=="dt_net_1d_width=400")& (table['alpha']==1.0),
        (table['model']=="dt_net_2d_width=128") & (table['alpha']==0.0),
        (table['model']=="dt_net_2d_width=128")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==1.0),
    ]
    
    values = ["dt","dt_prog","dt","dt_prog","dt_recall","dt_recall_prog","dt_recall","dt_recall_prog"]
    table["model"] = np.select(conditions, values)
    #shading only shown for 0 option
    table = table[table.model != "0"]
    fig, ax = plt.subplots(figsize=(20, 9))

    models = set(table.model)
    test_datas = set(table.test_data)
    alphas = set(table.alpha)

    sns.lineplot(data=table,
                 x="test_iter",
                 y="test_acc_mean",
                 hue="model",
                 style="test_data" if len(test_datas) > 1 else None,
                 palette="dark",
                 lw=4,
                 dashes=True,
                 units=None,
                #  legend="auto",
                 ax=ax)

    # if error_bars and "test_acc_sem" in table.keys():
    #     for model in models:
    #         for test_data in test_datas:
    #             for alpha in alphas:
    #                 print("in innner most loop 5")
    #                 data = table[(table.model == model) &
    #                              (table.test_data == test_data) &
    #                              (table.alpha == alpha)]
    #                 plt.fill_between(data.test_iter,
    #                                  data.test_acc_mean - data.test_acc_sem,
    #                                  data.test_acc_mean + data.test_acc_sem,
    #                                  alpha=0.1, color="k")

    tr = table.max_iters.max()  # training regime number
    ax.fill_between([0, tr], [105, 105], alpha=0.3, label="Training Regime")
    return ax



def get_schoopy_plot(table, error_bars=True):
    if error_bars and "test_acc_sem" in table.keys():
        print("triggered 3")
    fig, ax = plt.subplots(figsize=(20, 9))

    models = set(table.model)
    test_datas = set(table.test_data)
    alphas = set(table.alpha)

    sns.lineplot(data=table,
                 x="test_iter",
                 y="test_acc_mean",
                 hue="model",
                 size="alpha",
                 sizes=(2, 8),
                 style="test_data" if len(test_datas) > 1 else None,
                 palette="dark",
                 dashes=True,
                 units=None,
                 legend="auto",
                 ax=ax)

    if error_bars and "test_acc_sem" in table.keys():
        for model in models:
            for test_data in test_datas:
                for alpha in alphas:
                    data = table[(table.model == model) &
                                 (table.test_data == test_data) &
                                 (table.alpha == alpha)]
                    plt.fill_between(data.test_iter,
                                     data.test_acc_mean - data.test_acc_sem,
                                     data.test_acc_mean + data.test_acc_sem,
                                     alpha=0.1, color="k")

    tr = table.max_iters.max()  # training regime number
    ax.fill_between([0, tr], [105, 105], alpha=0.3, label="Training Regime")
    return ax



def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("--alpha_list", type=float, nargs="+", default=None,
                        help="only plot models with alphas in given list")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--filter", type=float, default=None,
                        help="cutoff for filtering by training acc?")
    parser.add_argument("--plot_name", type=str, default=None, help="where to save image?")
    parser.add_argument("--max_iters_list", type=int, nargs="+", default=None,
                        help="only plot models with max iters in given list")
    parser.add_argument("--model_list", type=str, nargs="+", default=None,
                        help="only plot models with model name in given list")
    parser.add_argument("--width_list", type=str, nargs="+", default=None,
                        help="only plot models with widths in given list")
    parser.add_argument("--max", action="store_true", help="add max values to table?")
    parser.add_argument("--min", action="store_true", help="add min values too table?")
    parser.add_argument("--xlim", type=list, nargs="+", default=None, help="x limits for plotting")
    parser.add_argument("--ylim", type=float, nargs="+", default=None, help="y limits for plotting")
    parser.add_argument("--line_thick_alpha", type=bool, default=False, help="makes the thickness of lines in graph proporional to alpha used")
    args = parser.parse_args()

    if args.plot_name is None:
        now = datetime.now().strftime("%m%d-%H.%M")
        args.plot_name = f"schoop{now}.png"
        plot_title = "Schoopy Plot"
    else:
        plot_title = args.plot_name[:-4]
        plot_title = args.plot_name

    # get table of results
    table = get_table(args.filepath,
                      args.max,
                      args.min,
                      filter_at=args.filter,
                      max_iters_list=args.max_iters_list,
                      alpha_list=args.alpha_list,
                      width_list=args.width_list,
                      model_list=args.model_list)

    # reformat and reindex table for plotting purposes
    table.columns = table.columns.map("_".join)
    table.columns.name = None
    table = table.reset_index()
    #print(table.round(2).to_markdown())
    if args.line_thick_alpha == True:
        ax = get_schoopy_plot(table)
    else:
        # ax = trigger_if_plot(table)
        ax = get_schoopy_plot_no_alpha_lines(table)

    ax.legend(fontsize=26, loc="upper left", bbox_to_anchor=(1.0, 0.8))
    x_max = table.test_iter.max()
    x = np.arange(20, x_max + 1, 10 if (x_max <= 100) else 100)
    ax.tick_params(axis="y", labelsize=34)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=34, rotation=37)
    if args.xlim is None:
        ax.set_xlim([x.min() - 0.5, x.max() + 0.5])
        # ax.set_xlim([1.5, 100.5]) #for right figure 6
        # ax.set_xlim([1.5, 500.5]) #for figure 3
    else:
        ax.set_xlim(args.xlim)
    if args.ylim is None:
        ax.set_ylim([0, 103])
    else:
        ax.set_ylim(args.ylim)
    ax.set_xlabel("Test-Time Iterations", fontsize=34)
    ax.set_ylabel("Accuracy (%)", fontsize=34)
    ax.set_title(plot_title, fontsize=34)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    # for line in ax.get_legend().get_lines():
    #     line.set_linewidth(10.0)
    if (plot_title == "Figure-3") or (plot_title == "Maze-Anomaly"):
        print("true")
        for line in ax.get_lines():
            line.set_linewidth(4.0)
    plt.savefig(args.plot_name)
    #plt.show()


if __name__ == "__main__":
    main()
