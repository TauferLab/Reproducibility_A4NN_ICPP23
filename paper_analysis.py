import os
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import matplotlib.cm as cm
import numpy as np
from datetime import timedelta
from matplotlib import font_manager as fm

# Settings to use for all plots
plt.style.use('ggplot')
figsize=(12,8)
size=200

cmap = cm.get_cmap('viridis')
secs = 3600
beams = ["Low", "Medium", "High"]
tfont = fm.FontProperties(family='STIXGeneral', weight='bold', size=30)
lfont = fm.FontProperties(family='STIXGeneral', math_fontfamily="stix", size=25, weight="bold")
tick_font = fm.FontProperties(family='STIXGeneral', math_fontfamily="stix", size=20)

def load_data():
    data_files = []

    for root, dirs, files in sorted(os.walk("./icpp_training_results")):
        if "stopping" in root:
            for i in files:
                f = os.path.join(root, i)
                data_files.append(f)

    return data_files

def plot_epochs_savings(beams, num_epochs_no_stop, stop_1e14, stop_1e15, stop_1e16):
    epochs_run = {
        'Baseline': np.repeat(num_epochs_no_stop, 3),
        '1GPU with PENGUIN': (stop_1e14[0], stop_1e15[0], stop_1e16[0]),
        '4GPU with PENGUIN': (stop_1e14[1], stop_1e15[1], stop_1e16[1]),
    }

    x = np.arange(len(beams))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=figsize)
    colors = ["xkcd:grey", "xkcd:orange", "xkcd:blue"]

    for index, (label, num_epochs) in enumerate(epochs_run.items()):
        print(label, num_epochs)
        offset = width * multiplier
        rects = ax.barh(x + offset, num_epochs, width, label=label, color=colors[index]) 
        labels = [str(round((1-(num_epochs[i]/num_epochs_no_stop))*100, 1))+'%' for i in range(len(num_epochs))]
        ax.bar_label(rects, labels=labels ,padding=2, font=tick_font)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Epochs Completed', font=lfont)
    ax.xaxis.label.set_color('black')
    ax.set_ylabel('Beam Intensity', font=lfont)
    ax.yaxis.label.set_color('black')
    ax.set_xlim(0, 2700)

    ax.set_title('Percent Epochs Saved', font=tfont)
    ax.set_yticks(x + width, beams)
    xticks = [int(x) for x in ax.get_xticks()]
    ax.set_xticklabels(xticks, color='black', font=tick_font)
    ax.set_yticklabels(beams, color='black', font=tick_font)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 1, 0]

    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper center", bbox_to_anchor=(0.5, -0.1),  ncol=3, prop=tick_font)

    plt.savefig('figures/figure10.png')
    plt.close()
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
        '1GPU': (no_stop_1e14[0], no_stop_1e15[0], no_stop_1e16[0]),
        '1GPU + PENGUIN': (stop_1e14[0], stop_1e15[0], stop_1e16[0]),
        '4GPU': (no_stop_1e14[1], no_stop_1e15[1], no_stop_1e16[1]),
        '4GPU + PENGUIN': (stop_1e14[1], stop_1e15[1], stop_1e16[1]),
    }

    x = np.arange(len(beams))  # the label locations
    width = 0.19  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=figsize)
    colors = ["xkcd:pale orange", "xkcd:orange", "xkcd:sky", "xkcd:blue"]

    print()
    for index, (label, times) in enumerate(epochs_run.items()):
        print(label, times)
        offset = width * multiplier
        rects = ax.bar(x + offset, times, width, label=label, color=colors[index]) 
        labels = [round(time, 2) for time in times]
        ax.bar_label(rects, labels=labels ,padding=2, font=tick_font)
        multiplier += 1

    # Labels, title, x-axis tick labels, etc.
    ax.set_ylabel('Run Time (hours)', font=lfont)
    ax.xaxis.label.set_color('black')
    ax.set_xlabel('Beam Intensity', font=lfont)
    ax.yaxis.label.set_color('black')

    ax.set_title('Wall Hours Spent', font=tfont)
    ax.set_xticks(x + width, beams)
    ax.set_xticklabels(beams, color='black', font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color='black', font=tick_font)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1),  ncol=4, prop=tick_font)
    plt.savefig('figures/time_saved.png')
    plt.close()
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

def get_epoch_converged(arch_df):
    sorted_by_epoch = arch_df.sort_values('epoch')
    try:
        converged = sorted_by_epoch[sorted_by_epoch.converged==True].iloc[0]
        return pd.Series({'epoch_converged': converged['epoch'], 'final_fitness': sorted_by_epoch['val_accs'].iloc[-1], 'prediction':converged['predictions']})
    except:
        return pd.Series({'epoch_converged': np.inf, 'final_fitness': sorted_by_epoch['val_accs'].iloc[-1], 'prediction':None})

def all_paretos(data_files):
    all_paretos = pd.DataFrame()
    for i in data_files:  
        df = pd.read_csv(i)
        by_arch = df.groupby('arch')

        if "1e14" in i:
            beam = "Low"
        elif "1e15" in i:
            beam = "Medium"
        elif "1e16" in i:  
            beam = "High"    

        if "no" in i:  
            stop = False
            one_line = by_arch.apply(one_line_per_arch).sort_values('arch', ascending=True).reset_index()  
        elif "stopping" in i:  
            stop = True  
            one_line = by_arch.apply(one_line_per_arch_penguin).sort_values('arch', ascending=True).reset_index()

        if "1gpu" in i:  
            gpu = 1  
        elif "4gpu" in i:  
            gpu = 4  

        print(one_line.head)
        sorted_by_fitness = one_line.sort_values('final_acc', ascending=False).head()
        costs = one_line[['final_acc', 'flops']].to_numpy()
        pareto_optimals = is_pareto_efficient_simple(costs) 
        one_line['pareto_optimal'] = pareto_optimals
        pareto_optimal_arches = one_line.loc[one_line['pareto_optimal'] == True].copy()

        pareto_optimal_arches['beam'] = beam
        pareto_optimal_arches['gpus'] = gpu
        pareto_optimal_arches['stop'] = stop

        all_paretos = pd.concat([all_paretos, pareto_optimal_arches], ignore_index=True)

    return all_paretos

def make_figure2():
    df= pd.read_csv('icpp_training_results/1gpu/no_stopping/1gpu_1e15_training_data.csv')
    by_arch = df.groupby('arch')
    where_converged = by_arch.apply(get_epoch_converged).reset_index()  
    print(where_converged.sort_values('epoch_converged', ascending=True).iloc[20:40])
    percent_converged = len(where_converged[where_converged.epoch_converged!=np.inf].index) / len(where_converged.index) * 100
    print("Percent Converged is: ", percent_converged)
    mean = where_converged[where_converged.epoch_converged!=np.inf].epoch_converged.mean()
    print("Avg epoch converged (for those converged) is: ", mean)
    print("converged arches and their earliest convergence:")
    only_converged_arches = pd.DataFrame(where_converged.loc[where_converged.epoch_converged!=np.inf,:])
    print(only_converged_arches.head())

    arch_44 = df[df['arch']==38]
    fig, ax = plt.subplots(layout='constrained', figsize=(12,5), dpi=200)

    a, b, c = 100.2, 1.15, 17

    x = np.linspace(0.5,26,200)
    y = a-b**(c-x)

    epochs, fitness = arch_44['epoch'].to_list(), arch_44['val_accs'].to_list()
    prediction, actual = 99.614529, 99.943311

    ax.plot(x,y,color="#2C4D96",label="Predictive Model", linewidth=3)

    ax.scatter(epochs, fitness, c='black', marker='o', facecolor='None', label='Observed Accuracy')
    ax.vlines(12,90,101,colors='#990000',linestyles=":",linewidth=4, label="Prediction Converged")
    ax.axvspan(12, 26, color = 'k', alpha = 0.2, label="Avoidable Training")

    ax.annotate('Predicted Accuracy = {:.2f}'.format(prediction), xy=(0.68, 0.68), xycoords='axes fraction', font=tick_font)
    ax.annotate('Actual accuracy = {:.2f}'.format(actual), xy=(0.68, 0.75), xycoords='axes fraction', font=tick_font)

    ax.set_ylim(90, 101)
    ax.set_xlim(0, 25)

    ax.set_ylabel("Accuracy (%)", font=lfont)
    ax.set_xlabel("Number of Epochs of Training", font=lfont)

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    ax.legend(loc='lower right', prop=tick_font)

    plt.savefig('figures/figure2.png')
    return

def make_figure8():
    # 1 GPU 
    fig, ax = plt.subplots(layout='constrained', figsize=figsize, dpi=160)

    plt.xlabel('FLOPS', color='black', font=lfont)
    plt.ylabel('Validation Accuracy', color='black', font=lfont)
    plt.xlim(325, 700)
    plt.ylim(87, 100.2)

    colors = ["xkcd:light orange", "xkcd:orange", "xkcd:burnt orange"]
    markers = ["o", "^", ""]
    size=200

    low_no = [93.909045, 93.293021, 97.763920]
    low_no_peng_flops = [595.47, 677.27, 597.57]
    low_pred = [98.804643, 96.645275, 99.532435]
    low_true = [93.55, 93.87, 97.833207]
    low_peng_flops = [624.04, 597.04, 661.70]

    medium_no = [99.162257, 97.564701, 98.412698]
    medium_no_peng_flops = [615.04, 533.20, 597.33]
    medium_pred = [99.559083, 95.722359, 99.937012]
    medium_true = [99.559083, 99.067775, 99.937012]
    medium_peng_flops = [599.71, 352.82, 679.70]

    high_no = [99.798438, 99.968506, 99.949609]
    high_no_peng_flops = [352.82, 432.79, 415.08]
    high_pred = [99.930713, 98.251432, 100.000000]
    high_true = [99.930713, 100.000000, 100.000000]
    high_peng_flops = [470.66, 470.66, 531.06]

    l1 = ax.scatter(low_no_peng_flops, low_no, marker= "o" , s=size, color=colors[0], label= "Low Beam" )
    l2 = ax.scatter(low_peng_flops, low_pred,  s=size, marker= "^" , color=colors[0])
    l3 = ax.scatter(low_peng_flops, low_true, s=size, marker="s", color=colors[0])

    l4 = ax.scatter(medium_no_peng_flops, medium_no, marker= "o" , s=size, color=colors[1], label= "Medium Beam" )
    l5 = ax.scatter(medium_peng_flops, medium_pred, s=size, marker= "^" , color=colors[1])
    l6 = ax.scatter(medium_peng_flops, medium_true, s=size, marker="s", color=colors[1])

    l7 = ax.scatter(high_no_peng_flops, high_no, s=size, marker= "o" , color=colors[2], label= "High Beam" )
    l8 = ax.scatter(high_peng_flops, high_pred, s=size, marker= "^" , color=colors[2])
    l9 = ax.scatter(high_peng_flops, high_true, s=size, marker="s", color=colors[2])

    l10 = ax.scatter([0], [0], s=size, marker='o', color= "black" , label= "No PENGUIN" )
    l11 = ax.scatter([0], [0], s=size, marker='^', color= "black" , label= "Predicted PENGUIN Fitness" )
    l12 = ax.scatter([0], [0], s=size, marker='s', color= "black", label = "True PENGUIN Fitness")

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    ax.legend(handles=[l1, l4, l7, l10, l11, l12], loc= "lower right" , prop=tick_font)

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    plt.savefig('figures/figure8.png')
    return

def make_figure9():
    # 4 GPUs
    fig, ax = plt.subplots(layout='constrained', figsize=figsize, dpi=160)

    plt.xlabel('FLOPS', color='black', font=lfont)
    plt.ylabel('Validation Accuracy', color='black', font=lfont)
    plt.xlim(300, 800)
    plt.ylim(92, 100.2)

    colors = ["xkcd:sky", "xkcd:cerulean blue", "xkcd:cobalt blue"]
    markers = ["o", "^", ""]
    size=200

    low_no = [97.751323, 98.425296, 93.902746]
    low_no_peng_flops = [615.2853, 712.9844, 531.0555]
    low_pred = [98.680175, 99.259251, 99.383542 ]
    low_true = [96.882086 , 97.159234, 97.889897 ]
    low_peng_flops = [517.2993, 633.2752, 679.4103]

    medium_no = [99.212648, 99.577979, 99.930713 ]
    medium_no_peng_flops = [548.7686 , 695.3055 , 777.1094 ]
    medium_pred = [98.770630, 99.911817, 99.905518]
    medium_true = [97.379693, 99.911817, 99.905518]
    medium_peng_flops = [533.2046, 704.4046, 697.6971]

    high_no = [100.000000, 100.000000, 99.962207 ]
    high_no_peng_flops = [513.0555 , 495.3423, 432.7927]
    high_pred = [100.000000, 98.704736, 99.735450]
    high_true = [100.000000, 96.403376 , 99.735450]
    high_peng_flops = [599.7112, 360.3750, 434.9419]

    l1 = ax.scatter(low_no_peng_flops, low_no, marker= "o" , s=size, color=colors[0], label= "Low Beam" )
    l2 = ax.scatter(low_peng_flops, low_pred,  s=size, marker= "^" , color=colors[0])
    l3 = ax.scatter(low_peng_flops, low_true, s=size, marker="s", color=colors[0])

    l4 = ax.scatter(medium_no_peng_flops, medium_no, marker= "o" , s=size, color=colors[1], label= "Medium Beam" )
    l5 = ax.scatter(medium_peng_flops, medium_pred, s=size, marker= "^" , color=colors[1])
    l6 = ax.scatter(medium_peng_flops, medium_true, s=size, marker="s", color=colors[1])

    l7 = ax.scatter(high_no_peng_flops, high_no, s=size, marker= "o" , color=colors[2], label= "High Beam" )
    l8 = ax.scatter(high_peng_flops, high_pred, s=size, marker= "^" , color=colors[2])
    l9 = ax.scatter(high_peng_flops, high_true, s=size, marker="s", color=colors[2])

    l10 = ax.scatter([0], [0], s=size, marker='o', color= "black" , label= "No PENGUIN" )
    l11 = ax.scatter([0], [0], s=size, marker='^', color= "black" , label= "Predicted PENGUIN Fitness" )
    l12 = ax.scatter([0], [0], s=size, marker='s', color= "black", label = "True PENGUIN Fitness")

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    ax.legend(handles=[l1, l4, l7, l10, l11, l12], loc= "lower right" , prop=tick_font)

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    plt.savefig('figures/figure9.png')
    return

def make_figure10(files):
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

def make_figure11():
    gpu1_1e14_stopping = pd.read_csv('./icpp_training_results/1gpu/stopping/1gpu_1e14_stopping_training_data.csv')
    gpu1_1e15_stopping = pd.read_csv('./icpp_training_results/1gpu/stopping/1gpu_1e15_stopping_training_data.csv')
    gpu1_1e16_stopping = pd.read_csv('./icpp_training_results/1gpu/stopping/1gpu_1e16_stopping_training_data.csv')
    gpu4_1e14_stopping = pd.read_csv('./icpp_training_results/4gpu/stopping/4gpu_1e14_stopping_training_data.csv')
    gpu4_1e15_stopping = pd.read_csv('./icpp_training_results/4gpu/stopping/4gpu_1e15_stopping_training_data.csv')
    gpu4_1e16_stopping = pd.read_csv('./icpp_training_results/4gpu/stopping/4gpu_1e16_stopping_training_data.csv')

    all_stopping = [gpu1_1e14_stopping, gpu1_1e15_stopping, gpu1_1e16_stopping, gpu4_1e14_stopping, gpu4_1e15_stopping, gpu4_1e16_stopping]
    
    sets = list()
    percents = list()

    for df in all_stopping:
        by_arch = df.groupby('arch')
        where_converged = by_arch.apply(get_epoch_converged).reset_index()
        only_converged_arches = pd.DataFrame(where_converged.loc[where_converged.epoch_converged!=np.inf,:])
        sets.append(only_converged_arches.epoch_converged.to_list())
        

        percent_converged = len(where_converged[where_converged.epoch_converged!=np.inf].index) / len(where_converged.index) * 100
        percents.append(percent_converged)
        print("percent converged is:", percent_converged)

    data_to_plot = sets
    fig, ax = plt.subplots(layout='constrained', figsize=figsize)

    ax = fig.add_axes([0,0,1,1])

    ax.set_ylim(4, 25)
    ax.set_xticklabels(['', 'Low Beam\n1 GPU', 'Medium Beam\n1 GPU', 'High Beam\n1 GPU', 'Low Beam\n4 GPU', 'Medium Beam\n4 GPU', 'High Beam\n4 GPU'], font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), font=tick_font)
    ax.set_xlabel('Test Configuration', font=lfont)
    ax.set_ylabel('Epoch Converged', font=lfont)

    bp = ax.violinplot(data_to_plot, showmeans=True)
    colors=["xkcd:orange", "xkcd:aquamarine", "xkcd:light blue", "xkcd:dark orange", "xkcd:blue green", "xkcd:blue"]

    for pc, color, percent in zip(bp['bodies'], colors, percents):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_label(str(int(percent))+'%')
        pc.set_alpha(1)

    plt.legend(title="Percent Converged", prop=tick_font, title_fontproperties=tick_font)
    plt.savefig('figures/figure11.png')
    
    return

def make_sup1(df):
    # 1 GPU - With and Without PENGUIN
    fig, ax = plt.subplots(layout='constrained', figsize=figsize, dpi=160)

    plt.xlabel('FLOPS', color='black', font=lfont)
    plt.ylabel('Validation Accuracy', color='black', font=lfont)
    plt.ylim(90, 100.2)

    colors = ["xkcd:light orange", "xkcd:orange", "xkcd:burnt orange"]
    markers = ["o", "^"]
    size=200

    low, medium, high = (df['beam'] == "Low"), (df['beam'] == "Medium"), (df['beam'] == "High")
    gpu1, gpu4 = (df['gpus'] == 1), (df['gpus'] == 4)
    no_stop, stop = (df['stop'] == False), (df['stop'] == True)

    low_acc = df.loc[low & gpu1 & no_stop]['final_acc'].to_numpy()
    low_flops = df.loc[low & gpu1 & no_stop]['flops'].to_numpy()

    low_peng_acc = df.loc[low & gpu1 & stop]['final_acc'].to_numpy()
    low_peng_flops = df.loc[low & gpu1 & stop]['flops'].to_numpy()

    med_acc = df.loc[medium & gpu1 & no_stop]['final_acc'].to_numpy()
    med_flops = df.loc[medium & gpu1 & no_stop]['flops'].to_numpy()

    med_peng_acc = df.loc[medium & gpu1 & stop]['final_acc'].to_numpy()
    med_peng_flops = df.loc[medium & gpu1 & stop]['flops'].to_numpy()

    high_acc = df.loc[high & gpu1 & no_stop]['final_acc'].to_numpy()
    high_flops = df.loc[high & gpu1 & no_stop]['flops'].to_numpy()

    high_peng_acc = df.loc[high & gpu1 & stop]['final_acc'].to_numpy()
    high_peng_flops = df.loc[high & gpu1 & stop]['flops'].to_numpy()

    l1 = ax.scatter(low_flops, low_acc, marker= "o" , s=size, color=colors[0], label= "Low Beam" )
    l2 = ax.scatter(low_peng_flops, low_peng_acc,  s=size, marker= "^" , color=colors[0])
    l3 = ax.scatter(med_flops, med_acc, marker= "o" , s=size, color=colors[1], label= "Medium Beam" )
    l4 = ax.scatter(med_peng_flops, med_peng_acc, s=size, marker= "^" , color=colors[1])
    l5 = ax.scatter(high_flops, high_acc, s=size, marker= "o" , color=colors[2], label= "High Beam" )
    l6 = ax.scatter(high_peng_flops, high_peng_acc, s=size, marker= "^" , color=colors[2])
    l7 = ax.scatter([0], [0], s=size, marker='o', color= "black" , label= "Without PENGUIN" )
    l8 = ax.scatter([0], [0], s=size, marker='^', color= "black" , label= "With PENGUIN" )

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    ax.legend(handles=[l1, l3, l5, l7, l8], loc= "lower right" , prop=tick_font)

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    plt.savefig('figures/supplemental_figure1.png')

    return

def make_sup2(df):
    # 4 GPU - With and Without PENGUIN
    fig, ax = plt.subplots(layout='constrained', figsize=figsize, dpi=160)

    plt.xlabel('FLOPS', color='black', font=lfont)
    plt.ylabel('Validation Accuracy', color='black', font=lfont)
    plt.ylim(90, 100.2)

    colors = ["xkcd:sky", "xkcd:cerulean blue", "xkcd:cobalt blue"]

    low, medium, high = (df['beam'] == "Low"), (df['beam'] == "Medium"), (df['beam'] == "High")
    gpu1, gpu4 = (df['gpus'] == 1), (df['gpus'] == 4)
    no_stop, stop = (df['stop'] == False), (df['stop'] == True)

    low_acc = df.loc[low & gpu4 & no_stop]['final_acc'].to_numpy()
    low_flops = df.loc[low & gpu4 & no_stop]['flops'].to_numpy()

    low_peng_acc = df.loc[low & gpu4 & stop]['final_acc'].to_numpy()
    low_peng_flops = df.loc[low & gpu4 & stop]['flops'].to_numpy()

    med_acc = df.loc[medium & gpu4 & no_stop]['final_acc'].to_numpy()
    med_flops = df.loc[medium & gpu4 & no_stop]['flops'].to_numpy()

    med_peng_acc = df.loc[medium & gpu4 & stop]['final_acc'].to_numpy()
    med_peng_flops = df.loc[medium & gpu4 & stop]['flops'].to_numpy()

    high_acc = df.loc[high & gpu4 & no_stop]['final_acc'].to_numpy()
    high_flops = df.loc[high & gpu4 & no_stop]['flops'].to_numpy()

    high_peng_acc = df.loc[high & gpu4 & stop]['final_acc'].to_numpy()
    high_peng_flops = df.loc[high & gpu4 & stop]['flops'].to_numpy()

    l1 = ax.scatter(low_flops, low_acc, marker= "o" , s=size, color=colors[0], label= "Low Beam" )
    l2 = ax.scatter(low_peng_flops, low_peng_acc,  s=size, marker= "^" , color=colors[0])
    l3 = ax.scatter(med_flops, med_acc, marker= "o" , s=size, color=colors[1], label= "Medium Beam" )
    l4 = ax.scatter(med_peng_flops, med_peng_acc, s=size, marker= "^" , color=colors[1])
    l5 = ax.scatter(high_flops, high_acc, s=size, marker= "o" , color=colors[2], label= "High Beam" )
    l6 = ax.scatter(high_peng_flops, high_peng_acc, s=size, marker= "^" , color=colors[2])
    l7 = ax.scatter([0], [0], s=size, marker='o', color= "black" , label= "Without PENGUIN" )
    l8 = ax.scatter([0], [0], s=size, marker='^', color= "black" , label= "With PENGUIN" )

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    ax.legend(handles=[l1, l3, l5, l7, l8], loc= "lower right" , prop=tick_font)

    ax.set_xticklabels(ax.get_xticks(), color= "black" , font=tick_font)
    ax.set_yticklabels(ax.get_yticks(), color= "black" , font=tick_font)
    plt.savefig('figures/supplemental_figure2.png')

if __name__=="__main__":
    files = load_data()

    # Figure 2 - Predictive Model Predictions vs. Observed Accuracy
    make_figure2()

    # Figure 8 - Single GPU Top 3 Pareto Optimal Architectures
    make_figure8()

    # Figure 9 - 4 GPUs Top 3 Pareto Optimal Architectures
    make_figure9()

    # Figure 10
    no_stop, stop = make_figure10(files)

    # Figure 11 - Violin Plot of Convergence
    make_figure11()

    # Figure 12
    print_wall_times(no_stop, stop)
    save_and_plot_times()

    # Supplemental Figure 1
    df = all_paretos(files)
    make_sup1(df)
    
    # Supplemental Figure 2
    make_sup2(df)