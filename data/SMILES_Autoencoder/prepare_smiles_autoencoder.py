#!/usr/bin/env python
import numpy as np
import pandas as pd

org_df = pd.read_csv("all_smiles_revised.csv",header=None)
src = org_df.iloc[:,0].tolist()
out_list = [ [x,x] for x in src if len(x)>=10 and len(x)<=128 ]

out_df = pd.DataFrame(out_list)
out_df.columns = ['src','tgt']

out_df.to_csv("all_smiles_revised_final.csv",index=False)


