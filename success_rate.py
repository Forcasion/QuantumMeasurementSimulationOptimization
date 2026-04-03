import csv
import sys


for case in range(9):
    print(f"Case {case}")
    file_name_partial = f"storage/case{case+1}/"
    filename_merged = file_name_partial + "output/multiple_states_ent1_merged.csv"
    filename_time = file_name_partial + "time"

    total = 0
    inf_count = 0

    with open(filename_merged) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            value = row[1].strip()
            if value in ("f", "ideal"):  # skip header/ideal lines
                continue
            total += 1
            if value == "inf":
                inf_count += 1

    # print(f"Total states:  {total}")
    # print(f"Inf count:     {inf_count}")
    print(f"Success rate:  {100 * (1 - inf_count / total):.1f}%")

    with open(filename_time) as f:
        print(f.read())