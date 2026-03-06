import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import sys


class CONFIG:

    SMILES_FILE = 'drug_smi.csv'


    OUTPUT_CSV_PATH = 'drug_morgan_fingerprints_1024.csv'

    FP_RADIUS = 2
    FP_BITS = 1024


def generate_fingerprint(smiles, radius, nbits):
    """
    为单个SMILES字符串生成Morgan指纹。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"警告: 无法解析 SMILES: {smiles}", file=sys.stderr)
        return [0] * nbits

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

    return list(fp.ToBitString())


def main():
    config = CONFIG()


    try:
        smiles_df = pd.read_csv(config.SMILES_FILE, header=0, sep='\t')
        if 'DrugID' not in smiles_df.columns or 'smi' not in smiles_df.columns:
            return
    except Exception as e:
        return


    all_fingerprints = []
    all_drug_ids = []

    for index, row in tqdm(smiles_df.iterrows(), total=len(smiles_df), desc="生成指纹"):
        drug_id = row['DrugID']
        smiles = row['smi']
        fp_list = generate_fingerprint(smiles, config.FP_RADIUS, config.FP_BITS)
        all_fingerprints.append(fp_list)
        all_drug_ids.append(drug_id)

    print("指纹生成完毕。")

    print("\n--- 步骤 3: 创建并保存新的 CSV 文件 ---")

    fp_columns = [f'MFP_{i}' for i in range(config.FP_BITS)]


    morgan_df = pd.DataFrame(all_fingerprints, columns=fp_columns, index=all_drug_ids)

    morgan_df.index.name = 'drug_id'

    try:
        morgan_df.to_csv(config.OUTPUT_CSV_PATH, index=True)
    except Exception as e:
        print(f"\n错误。 {e}")


if __name__ == '__main__':
    main()