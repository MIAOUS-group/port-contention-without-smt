"""
This file contains a simple parser for results of the pc tests.
It parses the two file created by seq-pc (one should have contention the other not)
And output stats about them.
"""

import array
import statistics


def parse_file(path):
    with open(path, 'rb') as input:
        data = input.read()

    timings = array.array('I')
    timings.frombytes(data)

    diff = []
    for i in range(0,len(timings),2):
        diff.append(timings[i+1]-timings[i])
    print("Filename: ", path, " Average: ", statistics.mean(diff)," Median: ", statistics.median(diff))
    return (statistics.median(diff))


if __name__ == '__main__':
    pc = parse_file("./timings_pc.bin")
    nopc = parse_file("./timings_nopc.bin")
    print("Ratio: ", pc/nopc)
