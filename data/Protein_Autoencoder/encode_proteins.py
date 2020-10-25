import argparse
import csv
import os

protein_encoding = {
    "A": "0",
    "B": "19",
    "C": "1",
    "D": "2",
    "E": "3",
    "F": "4",
    "G": "5",
    "H": "6",
    "I": "7",
    "J": "19",
    "K": "8",
    "L": "9",
    "M": "10",
    "N": "11",
    "O": "19",
    "P": "12",
    "Q": "13",
    "R": "14",
    "S": "15",
    "T": "16",
    "U": "19",
    "V": "17",
    "W": "18",
    "X": "19",
    "Y": "20",
    "Z": "21"
}


def encode(line):
    length = len(line)
    encoded = protein_encoding[line[0]]
    for i in range(1, length):
        encoded += "," + protein_encoding[line[i]]
    while length < 2000:
        encoded += ",22"
        length += 1
    encoded += '\n'
    return encoded


def main(input_file, output_file):
    assert os.path.exists(input_file)

    with open(input_file, "r") as input:
        with open(output_file, "w") as output:
            reader = csv.reader(input, delimiter=",")
            for line in reader:
                output.write(encode(line[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encodes proteins in the given file to one-hot coded values')
    parser.add_argument('input', help='input file obtained from clean_proteins.py')
    parser.add_argument('output', help='output file')
    args = parser.parse_args()
    main(args.input, args.output)
    print('Successfully encoded proteins')
