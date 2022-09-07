#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creates wat files for our experiments
Create a wat file with the two experiments (grouped and interleaved) that can be built with
wat2wasm (wabt)
"""

import argparse
import re
import os
import instructions

SRCDIR = './wat/'
OBJDIR = './build/'
DATA_POINTS = [1,2,5,10,20,50,100,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000]




VIUNOP = instructions.cross_product(["i16x8"], instructions.VIUNOP)
VIBINOP = instructions.cross_product(["i16x8"], instructions.VIBINOP) \
        + instructions.cross_product(["i16x8"], instructions.VIMINMAXOP) \
        + instructions.cross_product(["i16x8"], instructions.VISATBINOP) \
        + instructions.cross_product(["i16x8"],["mul"]) \
        + instructions.cross_product(["i16x8"], ["avgr_u"]) \
        + ["i16x8.q15mulr_sat_s"]

VIOP = VIUNOP + VIBINOP

VFUNOP = instructions.cross_product(["f64x2"], instructions.VFUNOP)
VFBINOP = instructions.cross_product(["f64x2"], instructions.VFBINOP)

VFOP = VFUNOP + VFBINOP
################################ SPAM FUNCTIONS ################################
def parse_vshape(vshape):
    vshape_re = re.search("([if][0-9]{1,2})x([0-9]{1,2})", vshape)
    num_type = vshape_re.group(1)
    param_count = vshape_re.group(2)
    return (num_type, param_count)

def get_lines(instruction_1, instruction_2):
    if instruction_1 in instructions.PUNOP:
        type_i = instruction_1[1][:3]
    else:
        type_i = instruction_1[:3]
    if instruction_2 in instructions.PUNOP:
        type_o = instruction_2[0][:3]
    else:
        type_o = instruction_2[:3]

    line_1 = []
    if instruction_1 in instructions.PUNOP:
        line_1.append('\t\t({})\n'.format(instruction_1[0]))
        line_1.append('\t\t({})\n'.format(instruction_1[1]))
    elif instruction_1 in instructions.BINOP:
        line_1.append('\t\t(local.get $p)\n')
        line_1.append("\t\t({})\n".format(instruction_1))
    else:
        line_1.append("\t\t({})\n".format(instruction_1))
    line_2 = []
    if instruction_2 in instructions.PUNOP:
        line_2.append('\t\t({})\n'.format(instruction_2[0]))
        line_2.append('\t\t({})\n'.format(instruction_2[1]))
    elif instruction_2 in instructions.BINOP:
        line_2.append('\t\t(local.get $p)\n')
        line_2.append("\t\t({})\n".format(instruction_2))
    else:
        line_2.append("\t\t({})\n".format(instruction_2))
    return(line_1, line_2, type_i, type_o)

def get_lines_v(instruction_1, instruction_2):
    type_i = parse_vshape(instruction_1)
    type_o = parse_vshape(instruction_2)
    line_1 = []
    line_2 = []

    if instruction_1 in VIUNOP+VFUNOP:
        line_1.append("\t\t({})\n".format(instruction_1))
    else:
        line_1.append('\t\t(local.get $v)\n')
        line_1.append("\t\t({})\n".format(instruction_1))
    if instruction_2 in VIUNOP+VFUNOP:
        line_2.append("\t\t({})\n".format(instruction_2))
    else:
        line_2.append('\t\t(local.get $v)\n')
        line_2.append("\t\t({})\n".format(instruction_2))
    return(line_1, line_2, type_i, type_o)


def create_seq_pc(lines, instruction_1, instruction_2, output = SRCDIR + 'seq-pc.wat'):
    (line_1, line_2, type_i, type_o) = get_lines(instruction_1, instruction_2)
    wat_lines = []
    wat_lines.append('(module\n')
    wat_lines.append('\t(func $interleaved (param $p {})(result {})\n'.format(type_i,type_o))
    wat_lines.append('\t\t(local.get $p)\n')
    for _ in range(lines+1):
        wat_lines += line_1
        wat_lines += line_2
    wat_lines.append('\t)\n')
    wat_lines.append('\t(export "interleaved" (func $interleaved))\n')
    wat_lines.append('\n\n\n')
    wat_lines.append('\t(func $grouped (param $p {})(result {})\n'.format(type_i,type_o))
    wat_lines.append('\t\t(local.get $p)\n')
    for _ in range(lines+1):
        wat_lines += line_1
    for _ in range(lines+1):
        wat_lines += line_2
    wat_lines.append('\t)\n')
    wat_lines.append('\t(export "grouped" (func $grouped))\n')
    wat_lines.append(')\n')
    with open(output,'w') as file:
        file.writelines(wat_lines)


def create_seq_pc_vi(lines, instruction_1, instruction_2, output = SRCDIR + 'seq-pc.wat'):
    (line_1, line_2, (type, param_count), _) = get_lines_v(instruction_1,instruction_2)
    param = ' '.join(["(param $p{} {})".format(k, "i32") for k in range(int(param_count))])

    wat_lines = []
    wat_lines.append('(module\n')
    wat_lines.append('\t(func $interleaved {} (result {})\n'.format(param,"i32"))
    wat_lines.append("\t\t(local $v v128)\n")
    wat_lines.append("\t\t(v128.const i16x8 {})\n".format( ' '.join(["0" for _ in range(int(param_count))])))
    for lane in range(int(param_count)):
        wat_lines.append("\t\t(local.get $p{})\n".format(lane))
        wat_lines.append("\t\t(i16x8.replace_lane {})\n".format(lane))
    wat_lines.append("\t\t(local.set $v)\n")
    wat_lines.append("\t\t(local.get $v)\n")
    for _ in range(lines+1):
        wat_lines += line_1
        wat_lines += line_2
    wat_lines.append("\t\t(i16x8.extract_lane_s 0)")
    wat_lines.append('\t)\n')
    wat_lines.append('\t(export "interleaved" (func $interleaved))\n')
    wat_lines.append('\n\n\n')
    wat_lines.append('\t(func $grouped {} (result {})\n'.format(param,"i32"))
    wat_lines.append("\t\t(local $v v128)\n")
    wat_lines.append("\t\t(v128.const i16x8 {})\n".format( ' '.join(["0" for _ in range(int(param_count))])))
    for lane in range(int(param_count)):
        wat_lines.append("\t\t(local.get $p{})\n".format(lane))
        wat_lines.append("\t\t(i16x8.replace_lane {})\n".format(lane))
    wat_lines.append("\t\t(local.set $v)\n")
    wat_lines.append("\t\t(local.get $v)\n")
    for _ in range(lines+1):
        wat_lines += line_1
    for _ in range(lines+1):
        wat_lines += line_2
    wat_lines.append("\t\t(i16x8.extract_lane_s 0)")
    wat_lines.append('\t)\n')
    wat_lines.append('\t(export "grouped" (func $grouped))\n')
    wat_lines.append(')\n')
    with open(output,'w') as file:
        file.writelines(wat_lines)

def create_seq_pc_vf(lines, instruction_1, instruction_2, output = SRCDIR + 'seq-pc.wat'):
    (line_1, line_2, (type, param_count), _) = get_lines_v(instruction_1,instruction_2)
    param = ' '.join(["(param $p{} {})".format(k, "f64") for k in range(int(param_count))])

    wat_lines = []
    wat_lines.append('(module\n')
    wat_lines.append('\t(func $interleaved {} (result {})\n'.format(param,"f64"))
    wat_lines.append("\t\t(local $v v128)\n")
    wat_lines.append("\t\t(v128.const f64x2 {})\n".format( ' '.join(["0" for _ in range(int(param_count))])))
    for lane in range(int(param_count)):
        wat_lines.append("\t\t(local.get $p{})\n".format(lane))
        wat_lines.append("\t\t(f64x2.replace_lane {})\n".format(lane))
    wat_lines.append("\t\t(local.set $v)\n")
    wat_lines.append("\t\t(local.get $v)\n")
    for _ in range(lines+1):
        wat_lines += line_1
        wat_lines += line_2
    wat_lines.append("\t\t(f64x2.extract_lane 0)")
    wat_lines.append('\t)\n')
    wat_lines.append('\t(export "interleaved" (func $interleaved))\n')
    wat_lines.append('\n\n\n')
    wat_lines.append('\t(func $grouped {} (result {})\n'.format(param,"f64"))
    wat_lines.append("\t\t(local $v v128)\n")
    wat_lines.append("\t\t(v128.const f64x2 {})\n".format( ' '.join(["0" for _ in range(int(param_count))])))
    for lane in range(int(param_count)):
        wat_lines.append("\t\t(local.get $p{})\n".format(lane))
        wat_lines.append("\t\t(f64x2.replace_lane {})\n".format(lane))
    wat_lines.append("\t\t(local.set $v)\n")
    wat_lines.append("\t\t(local.get $v)\n")
    for _ in range(lines+1):
        wat_lines += line_1
    for _ in range(lines+1):
        wat_lines += line_2
    wat_lines.append("\t\t(f64x2.extract_lane 0)")
    wat_lines.append('\t)\n')
    wat_lines.append('\t(export "grouped" (func $grouped))\n')
    wat_lines.append(')\n')
    with open(output,'w') as file:
        file.writelines(wat_lines)
##################################### MAIN #####################################

def wasm_generator(lines = 1000):
    if not os.path.isdir(SRCDIR):
        os.mkdir(SRCDIR)
    type = ["i64"]
    instruction_list = instructions.cross_product(type, instructions.IUNOP) + instructions.cross_product(type,instructions.IBINOP)
    for instruction_1 in instruction_list:
        for instruction_2 in instruction_list:
            create_seq_pc(lines,instruction_1,instruction_2, SRCDIR + '{}_{}.wat'.format(instruction_1,instruction_2))
    type = ["f64"]
    instruction_list = instructions.cross_product(type, instructions.FUNOP) + instructions.cross_product(type,instructions.FBINOP)
    for instruction_1 in instruction_list:
        for instruction_2 in instruction_list:
            create_seq_pc(lines,instruction_1,instruction_2, SRCDIR + '{}_{}.wat'.format(instruction_1,instruction_2))


    for instruction_1 in VIOP:
        for instruction_2 in VIOP:
            create_seq_pc_vi(lines,instruction_1,instruction_2, SRCDIR + '{}_{}.wat'.format(instruction_1,instruction_2))
    for instruction_1 in VFOP:
        for instruction_2 in VFOP:
            create_seq_pc_vf(lines,instruction_1,instruction_2, SRCDIR + '{}_{}.wat'.format(instruction_1,instruction_2))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lines', help='number of lines in the wat file', type=int, default = 1000000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    wasm_generator(args.lines)
