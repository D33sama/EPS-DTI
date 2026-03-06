from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os
from tqdm import tqdm
import sys


def create_autogluon_df(dti_df, drug_map, protein_map, drug_embeds, protein_embeds):
    """将DTI交互数据和嵌入拼接成一个AutoGluon可以使用的DataFrame"""
    feature_list, label_list = [], []
    for _, row in dti_df.iterrows():
        drug_id, protein_id = str(row['DrugID']), str(row['TargetID'])
        drug_idx, protein_idx = drug_map.get(drug_id), protein_map.get(protein_id)

        if drug_idx is None or protein_idx is None:
            continue

        combined_features = np.concatenate((drug_embeds[drug_idx], protein_embeds[protein_idx]))
        feature_list.append(combined_features)
        label_list.append(row['label'])

    feature_df = pd.DataFrame(feature_list)
    feature_df.columns = [f'f_{i}' for i in range(feature_df.shape[1])]
    feature_df['label'] = label_list
    return feature_df


def run_kfold_autogluon(setting='warm_start'):
    """执行完整的十折交叉验证流程，并手动计算所有评估指标"""

    DRUG_EMBEDDINGS_NPZ_PATH = 'features/2025115/115com.npz'
    PROTEIN_EMBEDDINGS_PATH = 'features/2025115/protein_embeddings.npy'
    PROTEIN_IDS_PATH = 'features/2025115/protein_ids.csv'
    DATA_FOLD_DIR = f'data_folds/{setting}/'
    print("--- Loading Main Feature Files (Once) ---")
    drug_data = np.load(DRUG_EMBEDDINGS_NPZ_PATH)
    drug_embeddings = drug_data['embeddings']
    drug_ids_array = drug_data['drug_ids']
    drug_ids_df = pd.DataFrame(drug_ids_array, columns=['drug_id'])

    protein_embeddings = np.load(PROTEIN_EMBEDDINGS_PATH)
    protein_ids_df = pd.read_csv(PROTEIN_IDS_PATH)

    drug_id_to_idx = {str(row['drug_id']): i for i, row in drug_ids_df.iterrows()}
    protein_id_to_idx = {str(row['id']): i for i, row in protein_ids_df.iterrows()}
    print(f"Loaded {len(drug_id_to_idx)} drugs and {len(protein_id_to_idx)} proteins.")

    all_fold_metrics = []
    print(f"\n--- Starting 10-Fold Cross-Validation for '{setting}' setting ---")

    for fold_idx in tqdm(range(10), desc="Cross-Validation Folds", file=sys.stdout):
        tqdm.write(f"\n===== Processing Fold {fold_idx + 1}/10 =====")
        train_dti_path = os.path.join(DATA_FOLD_DIR, f'train_fold_{fold_idx}.csv')
        test_dti_path = os.path.join(DATA_FOLD_DIR, f'test_fold_{fold_idx}.csv')
        if not os.path.exists(train_dti_path) or not os.path.exists(test_dti_path):
            tqdm.write(f"Warning: Data for fold {fold_idx} not found. Skipping.");
            continue

        train_dti_df = pd.read_csv(train_dti_path)
        test_dti_df = pd.read_csv(test_dti_path)
        train_data_ag = create_autogluon_df(train_dti_df, drug_id_to_idx, protein_id_to_idx, drug_embeddings,
                                            protein_embeddings)
        test_data_ag = create_autogluon_df(test_dti_df, drug_id_to_idx, protein_id_to_idx, drug_embeddings,
                                           protein_embeddings)

        predictor = TabularPredictor(
            label='label',
            problem_type='binary',
            eval_metric='roc_auc',
            path=f'Autogluon/yam2025_1112_fold_{fold_idx}'
        )

        predictor.fit(
            train_data=train_data_ag,
            presets='best_quality'
        )

        y_test_true = test_data_ag['label']
        X_test = test_data_ag.drop(columns=['label'])

        pred_probs = predictor.predict_proba(X_test)[1]
        pred_labels = predictor.predict(X_test)

        auroc = roc_auc_score(y_test_true, pred_probs)
        aupr = average_precision_score(y_test_true, pred_probs)
        f1 = f1_score(y_test_true, pred_labels)
        accuracy = accuracy_score(y_test_true, pred_labels)

        metrics_to_save = {
            'AUROC': auroc,
            'AUPR': aupr,
            'F1': f1,
            'Accuracy': accuracy
        }

        all_fold_metrics.append(metrics_to_save)
        tqdm.write(f"Fold {fold_idx + 1} Results: {metrics_to_save}")


    results_df = pd.DataFrame(all_fold_metrics)
    mean_results = results_df.mean()
    std_results = results_df.std()


    output_dir = 'reults'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f'1112gatcat{setting}.csv')

    print("\n\n========== 10-Fold Cross-Validation Final Summary ==========")
    print("\n--- Results for Each Fold ---\n" + results_df.to_string())
    print("\n--- Mean of Results ---\n", mean_results)
    print("\n--- Standard Deviation of Results ---\n", std_results)

    results_df.to_csv(output_path)
    print(f"\n✅ Final results saved to '{output_path}'")


if __name__ == '__main__':
    run_kfold_autogluon(setting='warm_start')