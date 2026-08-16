# 模块说明：实验统计结果的读取、聚合和可视化工具。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from pathlib import Path

# 变量说明（PROJECT_ROOT）：项目根目录的绝对路径，用于让独立脚本导入仓库模块。
PROJECT_ROOT = Path(__file__).resolve().parent

def plot_task_completion_stats(csv_path="all_earliest_steps.csv", save_path=None):
    # 变量说明（csv_path）：`plot_task_completion_stats` 读取或写入当前文件时使用的完整路径。
    # 变量说明（save_path）：当前产物写入磁盘时使用的完整路径。
    """
    Plot the mean and standard deviation of earliest step completion for each task across all rounds.
    Reads the cumulative CSV file, computes statistics, and displays a bar chart with error bars.
    :param csv_path: Path to the CSV file containing all rounds' earliest steps (default: 'all_earliest_steps.csv')
    :param save_path: If provided, saves the figure to this path instead of displaying it
    """
    

    # Read the cumulative results
    # 变量说明（df）：保存实验统计数据的 Pandas DataFrame。
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data to plot.")
        return

    # If only one row, mean and std will be scalars; convert to Series for consistent plotting
    if len(df) == 1:
        # Ensure task_means and task_std are pandas Series for consistent plotting
        # 变量说明（task_means）：`plot_task_completion_stats` 中待批量处理的任务集合。
        task_means = pd.Series(df.iloc[0], index=df.columns)
        # 变量说明（task_std）：`plot_task_completion_stats` 中当前处理的任务。
        task_std = pd.Series([0]*len(df.columns), index=df.columns)
    else:
        task_means = df.mean(axis=0)
        task_std = df.std(axis=0)

    # Ensure task_means and task_std are pandas Series for consistent plotting
    if not isinstance(task_means, pd.Series):
        task_means = pd.Series(task_means, index=df.columns)
    if not isinstance(task_std, pd.Series):
        task_std = pd.Series(task_std, index=df.columns)

    # Use .to_numpy() for values and .index for labels
    plt.figure(figsize=(10, 6))
    plt.bar(task_means.index, task_means.to_numpy(), yerr=task_std.to_numpy(), color='lightcoral')
    plt.xlabel('Tasks')
    plt.ylabel('Mean Step Completion')
    plt.title('Average Step (AS)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_task_completion_and_success(
    csv_path="all_earliest_steps.csv",
    icon_paths=None,
    save_path=None
):
    # 变量说明（csv_path）：`plot_task_completion_and_success` 读取或写入当前文件时使用的完整路径。
    # 变量说明（icon_paths）：`plot_task_completion_and_success` 中待批量处理的路径集合。
    # 变量说明（save_path）：当前产物写入磁盘时使用的完整路径。
    """
    Plot the mean, std, and success rate for each task across all rounds, with task icons above the bars.
    :param csv_path: Path to the CSV file containing all rounds' earliest steps (default: 'all_earliest_steps.csv')
    :param icon_paths: List of icon image paths for each task, in order
    :param save_path: If provided, saves the figure to this path instead of displaying it
    """
    # Set default icon paths if not provided
    if icon_paths is None:
        icon_paths = [
            'icons/wood.png', 'icons/table.png', 'icons/wood_pickaxe.png', 'icons/stone.png',
            'icons/stone_pickaxe.png', 'icons/iron.png', 'icons/coal.png', 'icons/furnace.png',
            'icons/iron_pickaxe.png', 'icons/diamond.png'
        ]

    # Read the cumulative results
    # 变量说明（df）：保存实验统计数据的 Pandas DataFrame。
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data to plot.")
        return

    # Compute mean, std, and success rate for each task
    if len(df) == 1:
        # 变量说明（task_means）：`plot_task_completion_and_success` 中待批量处理的任务集合。
        task_means = pd.Series(df.iloc[0], index=df.columns)
        # 变量说明（task_std）：`plot_task_completion_and_success` 中当前处理的任务。
        task_std = pd.Series([0]*len(df.columns), index=df.columns)
        # 变量说明（col）：当前处理的网格列索引。
        # 变量说明（success_rates）：各项任务在多轮实验中的完成比例。
        success_rates = pd.Series([100 if not pd.isna(df.iloc[0][col]) else 0 for col in df.columns], index=df.columns)
    else:
        task_means = df.mean(axis=0)
        task_std = df.std(axis=0)
        # Success rate: percentage of non-NaN values per column
        success_rates = df.notna().sum(axis=0) / len(df) * 100

    # Ensure all stats are pandas Series for consistent plotting
    if not isinstance(task_means, pd.Series):
        task_means = pd.Series(task_means, index=df.columns)
    if not isinstance(task_std, pd.Series):
        task_std = pd.Series(task_std, index=df.columns)
    if not isinstance(success_rates, pd.Series):
        success_rates = pd.Series(success_rates, index=df.columns)

    # Set up the plot
    # 变量说明（ax1）：对比图中用于绘制第一组指标的坐标轴。
    # 变量说明（fig）：Matplotlib 创建的整张图表画布。
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for mean step completion with std as error bars
    ax1.bar(task_means.index, task_means.to_numpy(), yerr=task_std.to_numpy(), capsize=5, color='lightcoral', label='Mean Step Completion')
    ax1.set_xlabel('Tasks')
    ax1.set_ylabel('Average Steps (AS)', color='lightcoral')
    ax1.tick_params(axis='y', labelcolor='lightcoral')
    ax1.set_xticklabels(task_means.index, rotation=45, ha='right')

    # Line plot for success rate on secondary y-axis
    # 变量说明（ax2）：对比图中用于绘制第二组指标的坐标轴。
    ax2 = ax1.twinx()
    ax2.plot(success_rates.index, success_rates.to_numpy(), color='mediumseagreen', marker='o', label='Success Rate')
    ax2.set_ylabel('Success Rate (%)', color='mediumseagreen')
    ax2.tick_params(axis='y', labelcolor='mediumseagreen')
    ax2.set_ylim(0, 110)

    # Add icons above each bar
    # 变量说明（zoom_ratios）：不同视图或图片分别计算出的缩放比例列表。
    zoom_ratios = [0.4, 0.4, 0.8, 0.4, 0.8, 0.4, 0.4, 0.4, 0.8, 0.4]
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    # 变量说明（task）：`plot_task_completion_and_success` 中当前处理的任务。
    # 变量说明（zoom_ratio）：把世界观察缩放到目标画布时使用的比例。
    for i, (task, zoom_ratio) in enumerate(zip(task_means.index, zoom_ratios)):
        if i < len(icon_paths):
            try:
                # 变量说明（img）：当前加载或正在处理的图像数组。
                img = mpimg.imread(icon_paths[i])
                # 变量说明（imagebox）：Matplotlib 中用于承载物品图标的 OffsetImage 对象。
                imagebox = OffsetImage(img, zoom=zoom_ratio)
                # 变量说明（ab）：在 Matplotlib 坐标轴上放置物品图标的 AnnotationBbox 对象。
                ab = AnnotationBbox(imagebox, (i, task_means[task]+50), frameon=False)
                ax1.add_artist(ab)
            # 变量说明（e）：当前循环正在处理的事件、边或异常元素。
            except Exception as e:
                print(f"Could not load icon for {task}: {e}")

    plt.title('Average Step Completion Time and Success Rate for Each Task')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


class Plotter:
    """
    A class for plotting experiment results with icons above each task.
    """
    def __init__(self, picture_names=None, colors=None, figsize=(10, 4)):
        # Set default icon paths if not provided
        # 变量说明（picture_names）：按时间顺序排列的观察帧文件名列表。
        # 变量说明（colors）：`__init__` 中待批量处理的颜色集合。
        # 变量说明（figsize）：Matplotlib 图表的宽高尺寸，单位为英寸。
        if picture_names is None:
            picture_names = [
                'icons/wood.png', 'icons/table.png', 'icons/wood_pickaxe.png', 'icons/stone.png',
                'icons/stone_pickaxe.png', 'icons/iron.png', 'icons/coal.png', 'icons/furnace.png',
                'icons/iron_pickaxe.png', 'icons/diamond.png'
            ]
        # Set default colors if not provided
        if colors is None:
            colors = [
                ('royalblue', 'lightskyblue'),
                ('firebrick', 'lightcoral'),
                ('darkgoldenrod', 'gold'),
                ('mediumseagreen', 'lightgreen')
            ]
        # 变量说明（self.picture_names）：按时间顺序排列的观察帧文件名列表。
        self.picture_names = picture_names
        # 变量说明（self.colors）：`__init__` 中待批量处理的颜色集合。
        self.colors = colors
        # 变量说明（self.figsize）：Matplotlib 图表的宽高尺寸，单位为英寸。
        self.figsize = figsize

    def plot(self, data_list, legend_labels=None, save_path=None):
        # 变量说明（data_list）：从多个实验文件读取、等待聚合的数据表列表。
        # 变量说明（legend_labels）：`plot` 中待批量处理的标签集合。
        """
        Read CSV files and plot mean and 95% confidence interval for multiple experiment groups, with icons above each task.
        :param csv_files: List of CSV file paths, each for one experiment group (rows: rounds, columns: tasks)
        :param legend_labels: List of legend labels for each group
        """
        # 变量说明（dfs）：从多个 CSV 文件加载的 Pandas DataFrame 列表。
        dfs = []
        # 变量说明（data）：与当前节点、文件或处理步骤关联的结构化数据。
        for data in data_list:
            if isinstance(data, str):
                # 变量说明（df）：保存实验统计数据的 Pandas DataFrame。
                df = pd.read_csv(data)
            else:
                df = data
            # Drop rows and columns that are completely empty
            df = df.dropna(how='all').dropna(axis=1, how='all')
            # If the first column is not numeric (e.g., an index or label), drop it
            if not np.issubdtype(df.iloc[:,0].dtype, np.number):
                try:
                    df = df.iloc[:, 1:]
                except Exception:
                    pass
            # Try to convert all data to float, non-convertible values become NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            dfs.append(df)

        if legend_labels is None:
            # 变量说明（i）：当前序列或网格遍历使用的索引。
            legend_labels = [f'Group {i+1}' for i in range(len(dfs))]

        # 变量说明（means_all）：所有实验输入对应的指标均值数组列表。
        means_all = []
        # 变量说明（ci_lower_all）：所有输入实验统计合并后的置信区间下界列表。
        ci_lower_all = []
        # 变量说明（ci_upper_all）：所有输入实验统计合并后的置信区间上界列表。
        ci_upper_all = []

        # Calculate mean and 95% confidence interval for each group
        for df in dfs:
            # 变量说明（means）：当前指标在不同轮次上的均值数组。
            means = df.mean(axis=0)
            # 变量说明（stds）：当前指标在不同轮次上的标准差数组。
            stds = df.std(axis=0)
            # 变量说明（n）：当前算法上下文中的元素数量或循环上限。
            n = df.shape[0]
            # 变量说明（ci_lower）：当前指标置信区间的下界数组。
            ci_lower = means - 1.96 * stds / np.sqrt(n)
            # 变量说明（ci_upper）：当前指标置信区间的上界数组。
            ci_upper = means + 1.96 * stds / np.sqrt(n)
            means_all.append(means)
            ci_lower_all.append(ci_lower)
            ci_upper_all.append(ci_upper)

        # For icon placement: get the max mean+ci_upper for each task
        # 变量说明（means_max）：用于统一图表纵轴范围的最大均值。
        means_max = np.max([ci_upper for ci_upper in ci_upper_all], axis=0)

        # 变量说明（x_values）：绘图时用作横轴的步骤编号数组。
        x_values = np.arange(len(dfs[0].columns))
        plt.figure(figsize=self.figsize)
        # Plot each group's mean and confidence interval
        # 变量说明（color）：`plot` 中当前处理的颜色。
        # 变量说明（label）：`plot` 中当前处理的标签。
        for means, ci_lower, ci_upper, label, color in zip(means_all, ci_lower_all, ci_upper_all, legend_labels, self.colors):
            plt.plot(x_values, means, '-o', label=label, color=color[0])
            plt.fill_between(x_values, ci_lower, ci_upper, color=color[1], alpha=0.3)

        # Add icons above each task
        # 变量说明（img_path）：`plot` 读取或写入当前文件时使用的完整路径。
        for i, img_path in enumerate(self.picture_names[:len(x_values)]):
            try:
                # 变量说明（full_path）：`plot` 读取或写入当前文件时使用的完整路径。
                full_path = PROJECT_ROOT / img_path
                # 变量说明（img）：当前加载或正在处理的图像数组。
                img = mpimg.imread(full_path)
                # 变量说明（imagebox）：Matplotlib 中用于承载物品图标的 OffsetImage 对象。
                imagebox = OffsetImage(img, zoom=0.3)  # Adjust zoom as needed
                # 变量说明（ab）：在 Matplotlib 坐标轴上放置物品图标的 AnnotationBbox 对象。
                ab = AnnotationBbox(imagebox, (x_values[i], means_max[i]), frameon=False, zorder=10)
                plt.gca().add_artist(ab)
            # 变量说明（e）：当前循环正在处理的事件、边或异常元素。
            except Exception as e:
                print(f"Could not load icon for {dfs[0].columns[i]}: {e}")

        plt.xticks(x_values, dfs[0].columns, rotation=45, ha='right')
        plt.ylabel('Average Steps (AS)')
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        # 主运行器传入实验目录；其他旧调用未传路径时保持原默认行为。
        save_path = save_path or 'results/result_fig.png'
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
