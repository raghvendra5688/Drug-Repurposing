import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse

#Define the src and target for torchtext to process
def run_smiles_generator(smiles):
    fp = []
    fpz = ""
    for i in range(256):
        fpz += '0'

    for i in range(len(smiles)):
        m = Chem.MolFromSmiles(smiles[i])
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=256)
            fp1 = fp1.ToBitString()
            fp.append(fp1)
        except:
            fp.append(fpz)

    data = []
    for i in range(len(smiles)):
        t = []
        for j in range(len(fp[i])):
            t.append(fp[i][j])
        data.append(t)

    f1 = pd.DataFrame(data)
    st = 'MFP_'
    for i in range(256):
        col = st + str(i)
        f1=f1.rename(columns = {i:col})

    return(f1)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "input filename")
    parser.add_argument("--output", help = "output filename")
    args = parser.parse_args()

    print('Inputfile:',args.input)
    print('Outputfile:',args.output)

    #Read input file containing uniprot_id, protein sequence, inchikey, smiles, pchembl value
    input_df = pd.read_csv("../data/"+args.input,header='infer')
    all_sequences = input_df['canonical_smiles'].values.tolist()

    #Pass the file to Autoencoder and get latent space dataframe
    ls_df = run_smiles_generator(all_sequences)

    #Write output 
    ls_df.to_csv("../data/"+args.output,index=False)
    print("Morgan fingerprints generation done")

