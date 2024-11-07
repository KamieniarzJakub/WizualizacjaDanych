import matplotlib.pyplot as plt
import pandas as pd

def main():
    data = {}
    files = ["1ers.csv", "1crs.csv", "2crs.csv", "1c.csv", "2c.csv",]
    labels = ['1-Evol-RS', '1-Coev-RS', '2-Coev-RS', '1-Coev', '2-Coev']
    colors = ['blue', 'green', 'red', 'black', 'magenta']
    markers = ['o', 'v', 'D', 's', 'd']
    for file in files:
        data[file[:-4]] = pd.read_csv(file)
    # Calculate the mean over runs for each generation
    for df in data.values():
        df["average"] = df.drop(['generation', 'effort'], axis=1).mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(7)

    for i, df in enumerate(data.values()):
        ax1.plot(df.effort / 1000, df.average * 100, 
            label=labels[i], 
            marker=markers[i], 
            color=colors[i],
            markevery=25, 
            markersize=7,
            markeredgecolor='black', 
            markeredgewidth=0.5,
            linewidth=0.9
        )

    ax1.grid(linestyle='dotted', linewidth=1, dashes=(1, 5))
    ax1.tick_params(axis='x', direction='in')  # Znaczniki na osi X do środka
    ax1.tick_params(axis='y', direction='in')  # Znaczniki na osi Y do środka
    ax1.legend(loc='lower right', numpoints=2, fontsize=10)
    ax1.set_xlabel(r'Rozegranych gier ($\times 1000$)')
    ax1.set_ylabel(r'Odsetek wygranych gier [$\%$]')
    ax1.set_xlim(0, 500)
    ax1.set_ylim(60, 100)

    ax1_up = ax1.twiny()
    ax1_up.set_xlabel('Pokolenie')
    ax1_up.set_xticks(range(0, 201, 40))



    boxes = [df.drop(['generation', 'effort', 'average'], axis=1).values[-1] * 100 for df in data.values()]

    ax2.boxplot(
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

    ax2.yaxis.tick_right()
    ax2.grid()
    ax2.set_ylim(60, 100)
    ax2.set_xticklabels(labels, rotation=15, fontdict={'fontsize': 10})
    ax2.set_yticks(range(60, 101, 5))
    
    ax2.grid(linestyle='dotted', linewidth=1, dashes=(1, 5))
    ax2.tick_params(axis='x', direction='in')  # Znaczniki na osi X do środka
    ax2.tick_params(top=True, bottom=True, direction="in")


    
    plt.savefig("myplot.png")

if __name__ == "__main__":
    main()