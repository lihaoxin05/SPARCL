import os
import argparse
import csv


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_csv",
                        default="",
                        type=str)
    parser.add_argument("--target_csv",
                        default="",
                        type=str)
    parser.add_argument("--target_ind",
                        default=-1,
                        type=int)
    parser.add_argument("--target_check_ind",
                        default=-1,
                        type=int)
    parser.add_argument("--source_csv",
                        default="",
                        type=str)
    parser.add_argument("--source_ind",
                        default=-1,
                        type=int)
    parser.add_argument("--source_check_ind",
                        default=-1,
                        type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = config()
    source_reader = csv.reader(open(args.source_csv), delimiter ='\t')
    source_head = next(source_reader)
    select_head = source_head[args.source_ind]

    target_reader = csv.reader(open(args.target_csv), delimiter ='\t')
    target_head = next(target_reader)
    target_head.insert(args.target_ind, select_head)
    count = len(target_head)

    writer = csv.writer(open(args.write_csv, 'w'), delimiter ='\t', lineterminator='\n')
    writer.writerow(target_head)

    data_dict = {}
    for item in target_reader:
        data_dict[item[0]] = item

    for item in source_reader:
        assert item[0] in data_dict.keys()
        if args.target_check_ind != -1:
            assert data_dict[item[0]][args.target_check_ind] == item[args.source_check_ind]
        data_dict[item[0]].insert(args.target_ind, item[args.source_ind])
    
    for key in data_dict.keys():
        assert len(data_dict[key]) == count
        writer.writerow(data_dict[key])
