import csv
import os
import numpy as np
DELIM = ","
QUOTECHAR = '|'




def get_real_size(imageName):
    return read_real_sizes(imageName)
    

def read_real_sizes(imageName):
    real_sizes = {}
    real_sizes_file = "../real_sizes.csv"
    size = -1.0
    with open(real_sizes_file, 'rU') as csvfile:

        csvreader = csv.reader(csvfile, delimiter=DELIM, quotechar=QUOTECHAR)
        try:
            for row in csvreader:
                name = row[0]
                currSize = row[1]
                name = name.replace(":", "_")
                
                imageName = imageName.replace(".jpg","")
                imageName = imageName.replace(".JPG","")
                imageName = imageName.replace("white/","")
                imageName = imageName.replace("blue/","")
                if name == imageName:
                    return float(currSize)

        except StandardError, e:
            print "can't real real files: {}".format(e)

    return size

def read_write_csv(out_file, imageName, bestAbaloneKey, bestRulerKey, abaloneLength, rulerLength, ellipseAbaloneLength):

    all_rows = {}
    all_diffs = {}
    all_ellipse_diffs = {}
    last_total_diff = 0.0
    total_diffs = 0.0
    if os.path.exists(out_file):
        with open(out_file, 'rU') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=DELIM, quotechar=QUOTECHAR)
            try:
                for i, row in enumerate(csvreader):
                    if i > 0:
                        name = row[0]
                        size = row[1]
                        real_size = row[2]
                        diff = row[3]

                        best_ab_key = row[4]
                        best_ruler_key = row[5]
                        ellipse_length = row[6]
                        ellipse_diff = row[7]

                        if name != "Total":
                            #print "for {}, best ab key: {}, best ruler key: {}".format(name, best_ab_key, best_ruler_key)
                            all_rows[name] = [size, real_size, best_ab_key, best_ruler_key, ellipse_length]
                            all_diffs[name] = float(diff)
                            all_ellipse_diffs[name] = float(ellipse_diff)
                        else:
                            last_total_diff = float(diff)
                            last_total_ellipse_diff = float(ellipse_diff)

            except StandardError, e:
                print("problem here: {}".format(e))

    try:
        
        real_size = get_real_size(imageName)
        if real_size > 0.0:
            diff = ((abaloneLength - real_size)/real_size)*100.0
            ellipse_diff = ((ellipseAbaloneLength - real_size)/real_size)*100.0
            all_rows[imageName] = [abaloneLength, real_size, bestAbaloneKey, bestRulerKey, ellipseAbaloneLength]
            all_diffs[imageName] = diff
            all_ellipse_diffs[imageName] = ellipse_diff
            #total_diffs = np.sum(all_diffs.values())
            total_diffs = sum((abs(d) for d in all_diffs.values()))
            total_ellipse_diffs = sum((abs(d) for d in all_ellipse_diffs.values()))
            with open(out_file, 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=DELIM, quotechar=QUOTECHAR, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Name", "Estimated", "Real", "Difference %","Best Abalone Match","Best Ruler Match", "Ellipse Length", "Ellipse Diff %"])
                for name, sizes in all_rows.items():
                    diff = all_diffs.get(name)
                    ellipse_diff = all_ellipse_diffs.get(name)
                    est_size = sizes[0]
                    real_size = sizes[1]
                    ab_key = sizes[2]
                    ruler_key = sizes[3]
                    ellipse_size = sizes[4]

                    writer.writerow([name, est_size, real_size, diff,ab_key,ruler_key, ellipse_size, ellipse_diff])

                writer.writerow(["Total", 0,0,total_diffs,"-","-","-", total_ellipse_diffs])
        else:
            print "Couldn't find real size for {}".format(imageName)
        print "last total: {}; this total: {}".format(last_total_diff, total_diffs)
    except StandardError, e:
        print "error trying to write the real size and diff: {}".format(e)
