import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class CONFIG:

    GNN_FEATURE_PATH = 'features/20251112/1112moact.npz'
    MORGAN_FEATURE_PATH = 'drug_morgan_fingerprints_1024.csv'
    COMBINED_OUTPUT_PATH = 'features/2025115/115com.npz'


def main():
    config = CONFIG()

    try:
        gnn_data = np.load(config.GNN_FEATURE_PATH)
        gnn_embeddings = gnn_data['embeddings']
        gnn_drug_ids = gnn_data['drug_ids']

    except Exception as e:
        return


    gnn_df = pd.DataFrame(gnn_embeddings, index=gnn_drug_ids)

    try:

        morgan_df = pd.read_csv(config.MORGAN_FEATURE_PATH, index_col=0)



    except Exception as e:

        return
    # (*** 修改结束 ***)



    common_drug_ids = gnn_df.index.intersection(morgan_df.index)

    if len(common_drug_ids) == 0:
        return



    gnn_aligned = gnn_df.loc[common_drug_ids]
    morgan_aligned = morgan_df.loc[common_drug_ids]


    combined_embeddings = np.concatenate([gnn_aligned.values, morgan_aligned.values], axis=1)

    aligned_drug_ids = gnn_aligned.index.values

    os.makedirs(os.path.dirname(config.COMBINED_OUTPUT_PATH), exist_ok=True)

    np.savez(
        config.COMBINED_OUTPUT_PATH,
        embeddings=combined_embeddings,
        drug_ids=aligned_drug_ids
    )




if __name__ == '__main__':
    main()