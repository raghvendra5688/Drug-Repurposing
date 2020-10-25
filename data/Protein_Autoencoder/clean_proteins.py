import argparse
import csv
import os


def seq_column_index(header):
    index = -1
    for token in header:
        index += 1
        if token == 'Sequence':
            return index
    print("Column 'Sequence' is not found in the file")
    os._exit(os.EX_DATAERR)


def main(input_file, output_file):
    assert os.path.exists(input_file)
    print("Loading protein sequences")
    with open(input_file, "r") as input:
        reader = csv.reader(input, delimiter=",")
        with open(output_file, "w") as output:
            seq_index = -1
            first = True
            for line in reader:
                if seq_index == -1:
                    seq_index = seq_column_index(line)
                else:
                    line = line[seq_index]
                    if len(line) <= 2000:
                        if first:
                            first = False
                        else:
                            output.write('\n')
                        output.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts proteins in the given csv file column \'Sequence\' with length <= 2000')
    parser.add_argument('input', help='input file with protein sequences in the column labelled Sequence')
    parser.add_argument('output', help='output file')
    args = parser.parse_args()
    main(args.input, args.output)
    print('Successfully cleaned protein sequences')
