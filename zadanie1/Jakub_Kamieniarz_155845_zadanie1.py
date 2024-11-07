#Jakub Kamieniarz 155845

import matplotlib.pyplot as plt
import pandas as pd

FILES = ["1ers.csv", "1crs.csv", "2crs.csv", "1c.csv", "2c.csv"]
LABELS = ['1-Evol-RS', '1-Coev-RS', '2-Coev-RS', '1-Coev', '2-Coev']
COLORS = ['blue', 'green', 'red', 'black', 'magenta']
MARKERS = ['o', 'v', 'D', 's', 'd']

def load_data(files):
    return {file[:-4]: pd.read_csv(file) for file in files}

def calculate_averages(data):
    for df in data.values():
        df["average"] = df.drop(['generation', 'effort'], axis=1).mean(axis=1)

def plot_effort_average(ax, data):
    for i, df in enumerate(data.values()):
        ax.plot(df.effort / 1000, df.average * 100, 
                label=LABELS[i], 
                marker=MARKERS[i], 
                color=COLORS[i],
                markevery=25, 
                markersize=7,
                markeredgecolor='black', 
                markeredgewidth=0.5,
                linewidth=0.9)
        
    ax.grid(linestyle='dotted', linewidth=1, dashes=(1, 5))
    ax.tick_params(axis='x', direction='in') 
    ax.tick_params(axis='y', direction='in')
    ax.legend(loc='lower right', numpoints=2, fontsize=10)
    ax.set_xlabel(r'Rozegranych gier ($\times 1000$)')
    ax.set_ylabel(r'Odsetek wygranych gier [$\%$]')
    ax.set_xlim(0, 500)
    ax.set_ylim(60, 100)
    ax1_up = ax.twiny()
    ax1_up.set_xlabel('Pokolenie')
    ax1_up.set_xticks(range(0, 201, 40))
    ax1_up.tick_params(axis='x', direction='in')

def plot_boxplot(ax, data):
    boxes = [df.drop(['generation', 'effort', 'average'], axis=1).values[-1] * 100 for df in data.values()]
    
    ax.boxplot(
        boxes,
        notch=True,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='blue', markeredgecolor='black', markersize=5),
        boxprops=dict(color='blue', linestyle="solid", linewidth=1.2),
        whiskerprops=dict(color='blue', linestyle='--', linewidth=1.5, dashes=(5, 6)),
        medianprops=dict(color='red', linestyle="-", linewidth=1.3),
        flierprops=dict(marker='+', markersize=7, markerfacecolor='blue', markeredgecolor='blue'),
        capprops=dict(color='black', linewidth=1.5)
    )

    ax.yaxis.tick_right()
    ax.set_ylim(60, 100)
    ax.set_xticklabels(LABELS, rotation=15, fontsize=10)
    ax.set_yticks(range(60, 101, 5))
    ax.grid(linestyle='dotted', linewidth=1, dashes=(1, 5))
    ax.tick_params(axis='x', direction='in') 
    ax.tick_params(top=True, bottom=True, direction="in")

def main():
    data = load_data(FILES)
    calculate_averages(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    plot_effort_average(ax1, data)

    plot_boxplot(ax2, data)

    plt.savefig("Jakub Kamieniarz 155845.png")

if __name__ == "__main__":
    main()
