"""
This file runs experiment for two instructions with a variable number of instructions
Ugly scipt that modifies the .S file to change instructions / number of lines
It then plots the results

The instructions are hardcoded in the main. You can change them here and it will
change them in the .S file then build.
"""


import plotly
import subprocess
import json
import array
import plotly.graph_objects as go
import statistics
import os
import csv


#X-Axis for the plot, ie the number of instruction for each experiment
DATA_POINTS = [1,2,5,10,20,50,100,200,500,1000, 2000,5000,10000,20000,50000, 100000, 200000, 500000]
# DATA_POINTS = [1000]
# DATA_POINTS = [k for k in range(0,100,2)]


#Instructions executed
INSTRUCTION_1 = "aesdec %xmm0, %xmm1" # First instruction
INSTRUCTION_2 = "popcnt %r8, %r8" # Second instruction
#                              Setting Parameters                              #

def set_loop(loop_iter, path = "./src/seq_pc.S"):
    print("Changing loop iterations...")
    with open(path,'r') as file:
        data = file.readlines()
    for i in range(len(data)):
        if "#flag" in data[i]:
            data[i] = ".rept {} #flag\n".format(loop_iter)
    with open(path,'w') as file:
        file.writelines(data)
    print("Done.")



def set_instruction(instruction_1, instruction_2, path = "./src/seq_pc.S"):
    print("Changing instructions...")
    with open(path,'r') as file:
        data = file.readlines()
    for i in range(len(data)):
        if "#instruction 1" in data[i]:
            data[i] = "{} #instruction 1\n".format(instruction_1)
        if "#instruction 2" in data[i]:
            data[i] = "{} #instruction 2\n".format(instruction_2)
    with open(path,'w') as file:
        file.writelines(data)
    print("Done")



def make():
    print("Compiling...")
    make_process = subprocess.Popen("make", stderr=subprocess.STDOUT)
    make_process.wait()
    if make_process.poll():
        print("\nCompilation failed")
        exit()
    print("Done.")


#                             Running  Experiments                             #

def run_experiment():
    exp = subprocess.Popen("./build/seq_pc", stderr=subprocess.STDOUT)
    exp.wait()
    if exp.poll():
        print("Failed run")
        exit()



def parse_file(path):
    with open(path, 'rb') as input:
        data = input.read()
    timings = array.array('I')
    timings.frombytes(data)
    diff = []
    for i in range(0,len(timings),2):
        diff.append(timings[i+1]-timings[i])
    return(diff)



def init_json(json_path):
    results = {"pc": {}, "nopc": {}}
    with open(json_path, 'w') as file:
        json.dump(results, file)



def append_json(json_path, pc, nopc, repetitions):
    with open(json_path,'r') as file:
        data = json.load(file)
    data['pc'][repetitions] = pc
    data['nopc'][repetitions] = nopc
    with open(json_path,'w') as file:
        json.dump(data,file)



def parse_results(json_path, repetitions):
    pc = parse_file("./timings_pc.bin")
    nopc = parse_file("./timings_nopc.bin")
    append_json(json_path, pc, nopc, repetitions)



def experiment(json_path, repetitions):
    print("Testing for {} repetitions".format(repetitions))
    set_loop(repetitions)
    make()
    run_experiment()
    parse_results(json_path,repetitions)



def experiments(json_path, data_points = DATA_POINTS):
    init_json(json_path)
    for repetitions in data_points:
        experiment(json_path,repetitions)


#                                     Plot                                     #
def read_json(json_path):
    with open(json_path) as file:
        data = json.load(file)
    return data



def plot_linechart(json_path, data_points = DATA_POINTS):
    data = read_json(json_path)
    fig = go.Figure()
    fig.update_layout(title="Execution time in function of repetition")
    pc_values = [statistics.median(l) for l in list(data['pc'].values())]
    nopc_values = [statistics.median(l) for l in list(data['nopc'].values())]
    fig.add_trace(go.Scatter(y = pc_values, x = list(data['pc'].keys()), name = "Grouped"))
    fig.add_trace(go.Scatter(y = nopc_values, x = list(data['nopc'].keys()), name = "Interleaved"))
    # fig.update_yaxes(type="log")
    fig.show()

    fig = go.Figure()
    fig.update_layout(title="\u03C1(grouped/interleaved)")
    ratio = [pc_values[i]/nopc_values[i] for i in range(len(list(data['pc'].keys())))]
    fig.add_trace(go.Scatter(x = list(data['pc'].keys()), y = ratio, name = "Grouped"))
    fig.show()

def dump_csv(json_path):
    data = read_json(json_path)
    x_axis = list(data['pc'].keys())
    pc_values = [statistics.median(l) for l in list(data['pc'].values())]
    nopc_values = [statistics.median(l) for l in list(data['nopc'].values())]

    #Time in function of repetitions
    output = './time_rep.csv'
    with open(output,'w') as output_file:
        csv_writer = csv.writer(output_file, delimiter=' ')
        csv_writer.writerow(['repetitions', 'pc', 'nopc'])
        for i in range(len(x_axis)):
            csv_writer.writerow([x_axis[i], pc_values[i], nopc_values[i]])


    #Ratio in function of repetitions
    output = './ratio_rep.csv'
    with open(output,'w') as output_file:
        csv_writer = csv.writer(output_file, delimiter=' ')
        csv_writer.writerow(['repetitions', 'ratio'])
        for i in range(len(x_axis)):
            csv_writer.writerow([x_axis[i], pc_values[i]/nopc_values[i]])


if __name__ == '__main__':
    # if not os.path.isdir("./build"):
    #     os.mkdir("./build")
    # if not os.path.isdir("./data"):
    #     os.mkdir("./data")
    instruction_1 = INSTRUCTION_1
    instruction_2 = INSTRUCTION_2
    instruction_1_name = instruction_1.split(" ")[0]
    set_instruction(instruction_1,instruction_2)
    json_path="data/{}_{}".format(instruction_1.split(" ")[0], instruction_2.split(" ")[0])
    experiments(json_path)
    plot_linechart(json_path)
    dump_csv(json_path)
    # print(json_path)
