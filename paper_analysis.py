import os
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import matplotlib.cm as cm
import numpy as np
from datetime import timedelta

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('ggplot')
cmap = cm.get_cmap('Blues')
secs = 3600
beams = ["1e14", "1e15", "1e16"]

def load_data():
    data_files = []

    for root, dirs, files in sorted(os.walk("./icpp_training_results")):
        if "stopping" in root:
            for i in files:
                f = os.path.join(root, i)
                data_files.append(f)

    return data_files

def calculate_and_plot_epochs_savings(files):
    no_stop_1e14, no_stop_1e15, no_stop_1e16 = [], [], []
    stop_1e14, stop_1e15, stop_1e16 = [], [], []
    stop_1e14_len, stop_1e15_len, stop_1e16_len = [], [], []
   
    for f in files: 
        if "no" in f:
            df = pd.read_csv(f)
            num_epochs_no_stopping = len(df.index) # non-stopped file

            if "1e14" in f:
                no_stop_1e14.append(df) 
            elif "1e15" in f:
                no_stop_1e15.append(df)
            elif "1e16" in f:
                no_stop_1e16.append(df)

        beam = f.split('/')[-1].split('_')[1].strip()
        gpus = f.split('/')[-1].split('_')[0].strip()
        stop = f.split('/')[-1].strip()
        if "stopping" in stop:
            df = pd.read_csv(f)
            print(f"Num epochs completed for {gpus} {beam} with PENGUIN:", len(df.index))

            if "1e14" in stop:
                stop_1e14.append(df) 
                stop_1e14_len.append(len(df.index))
            elif "1e15" in stop:
                stop_1e15.append(df) 
                stop_1e15_len.append(len(df.index))
            elif "1e16" in stop:
                stop_1e16.append(df) 
                stop_1e16_len.append(len(df.index))

    print()
    avg_epochs_completed = [sum(stop_1e14_len)/len(stop_1e14_len), sum(stop_1e15_len)/len(stop_1e15_len), sum(stop_1e16_len)/len(stop_1e16_len)]
    for i, b in enumerate(beams):
        print(f"Avg. epochs completed for {b}: {avg_epochs_completed[i]}")
        percent_epochs_saved = 1-(avg_epochs_completed[i]/num_epochs_no_stopping)
        print(f"Avg. percent epochs saved for {b}: {percent_epochs_saved*100}\n")

    plot_epochs_savings(beams, num_epochs_no_stopping, stop_1e14_len, stop_1e15_len, stop_1e16_len)

    return [no_stop_1e14, no_stop_1e15, no_stop_1e16], [stop_1e14, stop_1e15, stop_1e16]

def plot_epochs_savings(beams, num_epochs_no_stop, stop_1e14, stop_1e15, stop_1e16):
    width = 0.2
    multiplier = 0

    epochs_run = {
        'Baseline': np.repeat(num_epochs_no_stop, 3),
        '1GPU with PENGUIN': (stop_1e14[0], stop_1e15[0], stop_1e16[0]),
        '4GPU with PENGUIN': (stop_1e14[1], stop_1e15[1], stop_1e16[1]),
    }

    x = np.arange(len(beams))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(12, 8))
    colors = cm.Blues(np.linspace(0.4, 0.9, 3))

    for index, (label, num_epochs) in enumerate(epochs_run.items()):
        print(label, num_epochs)
        offset = width * multiplier
        rects = ax.barh(x + offset, num_epochs, width, label=label, color=colors[index]) 
        labels = [str(round((1-(num_epochs[i]/num_epochs_no_stop))*100, 1))+'%' for i in range(len(num_epochs))]
        ax.bar_label(rects, labels=labels ,padding=2)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Epochs Completed', fontsize=20)
    ax.set_ylabel('Beam Intensity', fontsize=20)

    ax.set_title('Percent Epochs Saved', fontsize=25)
    ax.set_yticks(x + width, beams)
    ax.legend(loc='upper right',fontsize=12)

    plt.savefig('figures/epochs_saved.png')
    plt.clf()
    return

def get_run_time(row):
    days = int(row.split('-')[0])
    hours = int(row.split('-')[1].split('_')[0])
    minutes = int(row.split('-')[1].split('_')[1])
    seconds = int(row.split('-')[1].split('_')[2])
    time = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return time

def get_run_seconds(row):
    days = int(row.split('-')[0])
    hours = int(row.split('-')[1].split('_')[0])
    minutes = int(row.split('-')[1].split('_')[1])
    seconds = int(row.split('-')[1].split('_')[2])
    time = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return time.total_seconds()

def print_wall_times(no_stop, stop):
    for n, beam in enumerate(no_stop): # Non-stop DataFrames for 1e14, 1e15, 1e16
        for i, df in enumerate(beam):
            if i % 2 == 0: # 1 GPU
                try:
                    gpu1_time = df['epoch_times'].sum()/secs
                    print(f"1 GPU: {gpu1_time}")
                except:
                    print("Non stop 1 GPU: these runs don't have epoch times")
            else: # 4 GPU
                try:
                    gpu4_time = df['epoch_times'].sum()/secs
                    print(f"4 GPUS: {gpu4_time}")
                except:
                    print("Non stop 4 GPU: these runs don't have epoch times")
    print()
    for n, val in enumerate(stop): # Stopped DataFrames for 1e14, 1e15, 1e16
        for i, df in enumerate(val):
            if i % 2 == 0: # 1 GPU
                try:
                    gpu1_time = df['epoch_times'].sum()/secs
                    print(f"1 GPU: {gpu1_time}")
                except:
                    print("Stop 1 GPU: these runs don't have epoch times")
            else: # 4 GPU
                try:
                    gpu4_time = df['epoch_times'].sum()/secs
                    print(f"4 GPUS: {gpu4_time}")
                except:
                    print("Stop 4 GPU: these runs don't have epoch times")
    return

def save_and_plot_times():
    gpu=[1,1,1,1,1,1,4,4,4,4,4,4]
    stopping=[False, False, False, True, True, True, False, False, False, True, True, True]
    dataset=['14','15','16','14','15','16','14','15','16','14','15','16']
    time=['2-02_01_42','2-03_52_23','2-00_38_20','1-22_33_04','1-12_05_38','1-08_18_16','0-14_32_18','0-14_54_32','0-13_53_12','0-12_03_46','0-09_10_05','0-09_27_38']

    time_df = pd.DataFrame()
    time_df['gpu'] = gpu
    time_df['dataset'] = dataset
    time_df['stopping'] = stopping
    time_df['time'] = time
    time_df['total_time'] = time_df['time'].apply(get_run_time)
    time_df['total_seconds'] = time_df['time'].apply(get_run_seconds)

    time_df.to_csv('./icpp_training_results/time_to_run.csv', index=False)
    plot_times()
    return 

def plot_times():
    time_df = pd.read_csv('./icpp_training_results/time_to_run.csv')

    # DataFrame filtering conditions
    b14, b15, b16 = (time_df['dataset']==14), (time_df['dataset']==15), (time_df['dataset']==16)
    no_stop, stop = (time_df['stopping']==False), (time_df['stopping']==True)
    gpu1, gpu4 = (time_df['gpu']==1), (time_df['gpu']==4)
    time_format = 'total_seconds'

    # Filtering data
    no_stop_1e14 = np.array([time_df[b14 & no_stop & gpu1][time_format].item(), time_df[b14 & no_stop & gpu4][time_format].item()])/secs
    no_stop_1e15 = np.array([time_df[b15 & no_stop & gpu1][time_format].item(), time_df[b15 & no_stop & gpu4][time_format].item()])/secs
    no_stop_1e16 = np.array([time_df[b16 & no_stop & gpu1][time_format].item(), time_df[b16 & no_stop & gpu4][time_format].item()])/secs

    stop_1e14 = np.array([time_df[b14 & stop & gpu1][time_format].item(), time_df[b14 & stop & gpu4][time_format].item()])/secs
    stop_1e15 = np.array([time_df[b15 & stop & gpu1][time_format].item(), time_df[b15 & stop & gpu4][time_format].item()])/secs
    stop_1e16 = np.array([time_df[b16 & stop & gpu1][time_format].item(), time_df[b16 & stop & gpu4][time_format].item()])/secs

    epochs_run = {
        '1GPU without PENGUIN': (no_stop_1e14[0], no_stop_1e15[0], no_stop_1e16[0]),
        '1GPU with PENGUIN': (stop_1e14[0], stop_1e15[0], stop_1e16[0]),
        '4GPU without PENGUIN': (no_stop_1e14[1], no_stop_1e15[1], no_stop_1e16[1]),
        '4GPU with PENGUIN': (stop_1e14[1], stop_1e15[1], stop_1e16[1]),
    }

    x = np.arange(len(beams))  # the label locations
    width = 0.17  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(12, 8))
    colors = cm.Blues(np.linspace(0.4, 0.9, 4))

    print()
    for index, (label, times) in enumerate(epochs_run.items()):
        print(label, times)
        offset = width * multiplier
        rects = ax.bar(x + offset, times, width, label=label, color=colors[index]) #, color= cmap(np.linspace(0, 1, 3))
        labels = [round(time, 2) for time in times]
        ax.bar_label(rects, labels=labels ,padding=2)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Run Time (hours)', fontsize=20)
    ax.set_xlabel('Beam Intensity', fontsize=20)

    ax.set_title('Wall Hours Spent', fontsize=25)
    ax.set_xticks(x + width, beams)
    ax.legend(loc='upper right', fontsize=12)

    plt.savefig('figures/time_saved.png')

    return

def one_line_per_arch_penguin(arch_df):
    sorted_by_epoch = arch_df.sort_values('epoch', ascending=False)
    if sorted_by_epoch['converged'].iloc[0] == True:
        to_return = sorted_by_epoch['predictions'].iloc[0]
    else:
        to_return = sorted_by_epoch['val_accs'].iloc[0]
    return pd.Series({'final_acc':to_return, 'flops':arch_df['flops'].iloc[0], 'converged':sorted_by_epoch['converged'].iloc[0]})

def one_line_per_arch(arch_df):
    sorted_by_epoch = arch_df.sort_values('epoch', ascending=False)
    to_return = sorted_by_epoch['val_accs'].iloc[0]
    return pd.Series({'final_acc':to_return, 'flops':arch_df['flops'].iloc[0], 'converged':sorted_by_epoch['converged'].iloc[0]})

def is_pareto_efficient_simple(unadjusted_costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """

    # this assumes minimization, so we need to invert accuracy to make bigger, better
    # we should look more closely at this
    costs = np.zeros(unadjusted_costs.shape, dtype=float)
    costs[:,1] = unadjusted_costs[:,1]
    costs[:,0] = unadjusted_costs[:,0] * -1
    
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def make_graphic(one_line, pareto_optimals, title="FLOPS vs. Val Accuracy per Architecture", gens=10, children=10):
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 8), dpi=160)

    as_numpy = one_line.to_numpy()

    plt.xlabel('FLOPS', fontsize=20)
    plt.ylabel('Validation Accuracy', fontsize=20)
    plt.title(title, fontsize=24)
    plt.ylim(85, 100.2)

    colors = cm.Blues(np.linspace(0, 1, gens))
    size=100

    labels = list()
    start, end, ng = 0, 0, 0
    for g in range(0, gens * children):
        if g == 0 or g % gens == 0:
            start = g
            end = start + (children - 1)
            labels.append(f"Generation {g // gens}")
        
            ngen = as_numpy[start:end + 1]
            plt.scatter(as_numpy[start:end+1, 2], as_numpy[start:end+1, 1], color=colors[g // gens], s=size, label=f"Generation {g // gens}", zorder=3)
    
    accs = pareto_optimals['final_acc'].to_numpy()
    flops = pareto_optimals['flops'].to_numpy()
    labels.append('Pareto Optimal')
    plt.scatter(flops, accs, s=250, marker='o', color='tab:orange', facecolor='None', linewidths=1.2, label='Pareto Optimal', zorder=10)
    plt.legend(labels, loc='lower right', fontsize=15)
    plt.tight_layout()

    plt.savefig('figures/'+title.replace(" ", "_")+'.png')
    plt.clf()
    return 

def plot_paretos(no_stop, stop):

    for n, beam in enumerate(no_stop): # Non-stop DataFrames for 1e14, 1e15, 1e16
        if n == 0: # 1e14
            tb = "Low Beam"
        elif n == 1: # 1e15
            tb = "Medium Beam"
        elif n == 2: # 1e16
            tb = "High Beam"
            
        for i, df in enumerate(beam):
            if i % 2 == 0: # 1 GPU
                title = f'1GPU with {tb} without PENGUIN'
            else:
                title = f'4GPU with {tb} without PENGUIN'

            by_arch = df.groupby('arch')
            one_line = by_arch.apply(one_line_per_arch).sort_values('arch', ascending=True).reset_index()
            sorted_by_fitness = one_line.sort_values('final_acc', ascending=False).head()
            costs = one_line[['final_acc', 'flops']].to_numpy()
            pareto_optimals = is_pareto_efficient_simple(costs)
            one_line['pareto_optimal'] = pareto_optimals
            pareto_optimal_arches = one_line.loc[one_line['pareto_optimal'] == True]
            print(f"\nParetos For {tb}:\n {pareto_optimal_arches}")
            make_graphic(one_line, pareto_optimal_arches, title=title)

    for n, beam in enumerate(stop): # Non-stop DataFrames for 1e14, 1e15, 1e16
        if n == 0: # 1e14
            tb = "Low Beam"
        elif n == 1: # 1e15
            tb = "Medium Beam"
        elif n == 2: # 1e16
            tb = "High Beam"
            
        for i, df in enumerate(beam):
            if i % 2 == 0: # 1 GPU
                title = f'1GPU with {tb} with PENGUIN'
            else:
                title = f'4GPU with {tb} with PENGUIN'

            by_arch = df.groupby('arch')
            one_line = by_arch.apply(one_line_per_arch).sort_values('arch', ascending=True).reset_index()
            sorted_by_fitness = one_line.sort_values('final_acc', ascending=False).head()
            costs = one_line[['final_acc', 'flops']].to_numpy()
            pareto_optimals = is_pareto_efficient_simple(costs)
            one_line['pareto_optimal'] = pareto_optimals
            pareto_optimal_arches = one_line.loc[one_line['pareto_optimal'] == True]
            print(f"\nParetos For {tb}:\n {pareto_optimal_arches}")
            make_graphic(one_line, pareto_optimal_arches, title=title)   
    
    return

if __name__=="__main__":
    files = load_data()
    no_stop, stop = calculate_and_plot_epochs_savings(files)
    print_wall_times(no_stop, stop)
    save_and_plot_times()

    plot_paretos(no_stop, stop)