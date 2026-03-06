import torch
import pandas as pd
import numpy as np
from transformers import EsmModel, AutoTokenizer
import re
from tqdm import tqdm


def extract_protein_embeddings(csv_path, output_embeddings_path, output_ids_path):
    """
    使用ESM-2模型从CSV文件中提取蛋白质序列的表征。

    参数:
    - csv_path: 输入的CSV文件路径。
    - output_embeddings_path: 保存.npy格式嵌入向量的路径。
    - output_ids_path: 保存蛋白质ID列表的路径。
    """


    local_model_path = r"../../esm2_local_model"

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = EsmModel.from_pretrained(local_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


    sequences = []
    protein_ids = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'([^ \t]+)[\t ]+(.*)', line)
            if match:
                protein_id, sequence = match.groups()
                protein_ids.append(protein_id)
                sequences.append(sequence)


    all_embeddings = []

    with torch.no_grad():
        for seq in tqdm(sequences, desc="处理蛋白质"):

            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1022).to(device)

            outputs = model(**inputs)

            hidden_states = outputs.last_hidden_state

            embedding = hidden_states[:, 1:-1, :].mean(dim=1).squeeze()

            all_embeddings.append(embedding.cpu().numpy())



    embeddings_array = np.array(all_embeddings)

    np.save(output_embeddings_path, embeddings_array)


    pd.DataFrame(protein_ids, columns=['id']).to_csv(output_ids_path, index=None)



if __name__ == '__main__':
    INPUT_CSV = 'tar_seq.csv'
    OUTPUT_EMBEDDINGS = 'features/2025115/protein_embeddings.npy'
    OUTPUT_IDS = 'features/2025115/protein_ids.csv'

    extract_protein_embeddings(INPUT_CSV, OUTPUT_EMBEDDINGS, OUTPUT_IDS)