#!/usr/bin/python3

import json
import matplotlib.pyplot as plt
import pdb
import sys

def main(report_file_name):
    with open(report_file_name) as report_file:
        report = json.loads(report_file.read())
        with open(report["stats_file"]) as stats_file:
            x      = []
            best_y = []
            avg_y  = []
            
            for line in stats_file:
                i, best, avg = line.strip().split(',')
                x.append(int(i))
                best_y.append(float(best))
                avg_y.append(float(avg))

            plt.gca().add_line(plt.Line2D(x, best_y ,lw=2.0, color='blue'))
            plt.gca().add_line(plt.Line2D(x, avg_y ,lw=2.0, color='red'))

#            plt.axis('scaled')
            plt.axis([0, max(x), min(avg_y), max(best_y)])
#            plt.show()
            output_filename = 'images/' + report_file_name.split('/')[-1].replace('json', 'png')
            plt.savefig(output_filename)

if __name__=='__main__':
    main(sys.argv[1])
    
