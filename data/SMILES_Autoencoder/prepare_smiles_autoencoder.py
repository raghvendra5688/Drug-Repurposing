#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trimming SMILES within the limit range')
    parser.add_argument('input1', help='input file')
    parser.add_argument('output', help='output file')
    args = parser.parse_args()

    org_df = pd.read_csv(args.input1,header=None)
    src = org_df.iloc[:,0].tolist()
    out_list = [ [x,x] for x in src if len(x)>=10 and len(x)<=128 ]

    out_df = pd.DataFrame(out_list)
    out_df.columns = ['src','tgt']

    out_df.to_csv(args.output,index=False)
    print("No of SMILES generated: ",out_df.shape[0])
    print("SMILES preparation done")


