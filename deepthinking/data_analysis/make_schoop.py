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
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from make_table import get_table

def get_schoopy_plot_no_alpha_lines(table, error_bars=True):
    print(type(table))
    if error_bars and "test_acc_sem" in table.keys():
        print("trigged if at 1")
    # print(table.head(10))
    temp  =table[(table.alpha == 1.0)]
    print("at beggining count is",len(temp.index))
    # print(temp.head(100))
    # print(table.head(5))
    conditions = [
        (table['model']=="dt_net_1d_width=400") & (table['alpha']==0.0),
        (table['model']=="dt_net_1d_width=400")& (table['alpha']==1.0),
        (table['model']=="dt_net_2d_width=128") & (table['alpha']==0.0),
        (table['model']=="dt_net_2d_width=128")& (table['alpha']==1.0),
        (table['model']=="dt_net_2d_width=512") & (table['alpha']==0.0),
        (table['model']=="dt_net_2d_width=512")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_2d_width=512")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_2d_width=512")& (table['alpha']==1.0),
        (table['model']=="feedforward_net_1d_width=400"),
        (table['model']=="feedforward_net_2d_width=128"),
        (table['model']=="feedforward_net_recall_1d_width=400"),
        (table['model']=="feedforward_net_recall_2d_width=128")
    ]
    
    values = ["dt","dt_prog","dt","dt_prog","dt","dt_prog","dt_recall","dt_recall_prog","dt_recall","dt_recall_prog","dt_recall","dt_recall_prog","ff","ff","ff","ff"]
    table["graph_name"] = np.select(conditions, values)
    table = table[(table['graph_name'] == "dt")|(table['graph_name'] == "dt_prog")|(table['graph_name'] == "dt_recall")|(table['graph_name'] == "dt_recall_prog")|(table['graph_name'] == "ff")]
    temp  =table[(table.alpha == 1.0)]
    print("in middle count is",len(temp.index))
    fig, ax = plt.subplots(figsize=(20, 9))
    # print(table.head(10))
    models = set(table.graph_name)
    test_datas = set(table.test_data)
    alphas = set(table.alpha)
    print(test_datas)
                    #  x="test_iter",
                                    #  x="max_iter",
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
        (table['model']=="dt_net_2d_width=512") & (table['alpha']==0.0),
        (table['model']=="dt_net_2d_width=512")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_1d_width=512")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_1d_width=512")& (table['alpha']==1.0),
    ]
    
    values = ["dt","dt_prog","dt","dt_prog","dt","dt_prog","dt_recall","dt_recall_prog","dt_recall","dt_recall_prog","dt_recall","dt_recall_prog"]
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

def get_schoopy_plot_alpha_colour(table, error_bars=True):
    # if error_bars and "test_acc_sem" in table.keys():
    #     print("triggered 3")
    conditions = [
        (table['model']=="dt_net_1d_width=400") & (table['alpha']==0.0),
        (table['model']=="dt_net_1d_width=400")& (table['alpha']==1.0),
        (table['model']=="dt_net_2d_width=128") & (table['alpha']==0.0),
        (table['model']=="dt_net_2d_width=128")& (table['alpha']==1.0),
        (table['model']=="dt_net_2d_width=512") & (table['alpha']==0.0),
        (table['model']=="dt_net_2d_width=512")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_1d_width=400")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_2d_width=128")& (table['alpha']==1.0),
        (table['model']=="dt_net_recall_1d_width=512")& (table['alpha']==0.0),
        (table['model']=="dt_net_recall_1d_width=512")& (table['alpha']==1.0),
    ]
    
    values = ["dt","dt_prog","dt","dt_prog","dt","dt_prog","dt_recall","dt_recall_prog","dt_recall","dt_recall_prog","dt_recall","dt_recall_prog"]
    table["model"] = np.select(conditions, values)
    fig, ax = plt.subplots(figsize=(20, 9))

    models = set(table.model)
    test_datas = set(table.test_data)
    alphas = set(table.alpha)

    sns.lineplot(data=table,
                 x="test_iter",
                 y="test_acc_mean",
                 hue="alpha",
                 linewidth = 3.0,
                 sizes=(2, 8),
                 style="test_data" if len(test_datas) > 1 else None,
                 palette='bright',#'colorblind',#sns.color_palette("tab10"),#"bright",
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

def get_schoopy_plot_colour_all_same(table, error_bars=True):
    # if error_bars and "test_acc_sem" in table.keys():
    #     print("triggered 3")
    fig, ax = plt.subplots(figsize=(20, 9))

    models = set(table.model)
    test_datas = set(table.test_data)
    alphas = set(table.alpha)

    sns.lineplot(data=table,
                 x="test_iter",
                 y="test_acc_mean",
                 hue="model",
                 linewidth = 3.0,
                 sizes=(2, 8),
                 style="test_data" if len(test_datas) > 1 else None,
                #  palette="bright",
                 dashes=True,
                 units=None,
                 legend="auto",
                 ax=ax)

    # if error_bars and "test_acc_sem" in table.keys():
    #     for model in models:
    #         for test_data in test_datas:
    #             for alpha in alphas:
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


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("--alpha_list", type=float, nargs="+", default=None,
                        help="only plot models with alphas in given list")
    parser.add_argument("--filepath", type=str, default="",help="first folders in path")
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
    parser.add_argument("--xlim", type=float, nargs="+", default=None, help="x limits for plotting")
    parser.add_argument("--ylim", type=float, nargs="+", default=None, help="y limits for plotting")
    parser.add_argument("--line_thick_alpha", type=bool, default=False, help="makes the thickness of lines in graph proporional to alpha used")
    parser.add_argument("--colour_by_alpha", type=bool, default=False, help="makes the colour of the lines relate to the alpha used")
    parser.add_argument('--folders', nargs='+', help='the folders the models are in', required=True)
    parser.add_argument("--colour_all_same", type=bool, default=False, help="colours all lines the same colour")
    args = parser.parse_args()

    if args.plot_name is None:
        now = datetime.now().strftime("%m%d-%H.%M")
        args.plot_name = f"schoop{now}.png"
        plot_title = "Schoopy Plot"
    else:
        plot_title = args.plot_name[:-4]
        plot_title = args.plot_name
    # print(args.folders)
    # print(type(args.folders))
    tablestore = []
    for file in args.folders:
        if args.filepath != "": 
            path = args.filepath +"/"+ file
        else:
            path = file
        print("path is ",path)
        # get table of results
        table = get_table(path,
                        args.max,
                        args.min,
                        filter_at=args.filter,
                        max_iters_list=args.max_iters_list,
                        alpha_list=args.alpha_list,
                        width_list=args.width_list,
                        model_list=args.model_list)
        print(table.shape)
        print(table.head())
        tablestore.append(table)
    if len(tablestore) == 0:
        raise Exception("no input files")
    table = tablestore[0]
    if len(tablestore) > 1:
        for i in range (1,len(tablestore)):
            table = table = pd.concat([table,tablestore[i]])
    print(table.shape)
    print(table.head())
    # table2 = get_table(args.filepath,
    #                 args.max,
    #                 args.min,
    #                 filter_at=args.filter,
    #                 max_iters_list=args.max_iters_list,
    #                 alpha_list=args.alpha_list,
    #                 width_list=args.width_list,
    #                 model_list=args.model_list)
    # print(table2.shape)
    # print(table2.head())
    # table = pd.concat([table,table2])
    # print(table.shape)
    # print(table.head())
    # reformat and reindex table for plotting purposes
    table.columns = table.columns.map("_".join)
    table.columns.name = None
    table = table.reset_index()
    #print(table.round(2).to_markdown())
    if args.line_thick_alpha == True:
        ax = get_schoopy_plot(table)
    elif args.colour_by_alpha == True:
        ax = get_schoopy_plot_alpha_colour(table)
    elif args.colour_all_same ==True:
        ax = get_schoopy_plot_colour_all_same(table)
    else:
        # ax = trigger_if_plot(table)
        ax = get_schoopy_plot_no_alpha_lines(table)

    # ax.legend(fontsize=26, loc="upper left", bbox_to_anchor=(1.0, 0.8))
    ax.legend(fontsize=26, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    x_max = table.test_iter.max()
    x = np.arange(20, x_max + 1, 10 if (x_max <= 100) else 100)
    ax.tick_params(axis="y", labelsize=34)
    ax.set_xticks(x)
    #to set x values use below two line and change x in th eline below to l
    # l = np.linspace(0, 200, num=5).tolist()
    # plt.xticks(ticks=l, labels=l)
    ax.set_xticklabels(x, fontsize=34, rotation=37)

    if args.xlim is None:
        # print("x mins is ",x.min())
        ax.set_xlim([x.min() - 0.5, x.max() + 0.5])
        # ax.set_xlim([1.5, 200.5])
        # ax.set_xlim([0,500.5]) #for mismatch sums
        # ax.set_xlim([1.5, 130.5]) #for chess
        # ax.set_xlim([1.5, 100.5]) #for right figure 6
        if plot_title.find("maze") != -1:
            ax.set_xlim([1.5, 1000.5]) #for mazes
        elif (plot_title.find("sum") != -1) or (plot_title.find("prefix") != -1):
            ax.set_xlim([1.5, 500.5]) #for sums
        else:
            ax.set_xlim([1.5, 130.5]) #for chess
    else:
        ax.set_xlim([x.min() - 0.5,320])
    if args.ylim is None:
        ax.set_ylim([0, 103])
        # ax.set_ylim([0, 25]) #for bad mazes
    else:
        ax.set_ylim(args.ylim)
    ax.set_xlabel("Test-Time Iterations", fontsize=34)
    ax.set_ylabel("Accuracy (%)", fontsize=34)
    ax.set_title(plot_title, fontsize=34)
    ax.set_title("Prefix Sums with 0.01 Learning Rate Factor (gamma)", fontsize=34)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    for line in ax.get_legend().get_lines():
        line.set_linewidth(10.0)
    if (plot_title == "Figure-3") or (plot_title == "Maze-Anomaly") or True:
        print("true")
        for line in ax.get_lines():
            line.set_linewidth(3.0)
    plt.savefig(args.plot_name)
    #plt.show()


if __name__ == "__main__":
    main()
