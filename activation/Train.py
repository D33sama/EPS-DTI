import os
import sys
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from rdkit import Chem, rdBase
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.loader import DataLoader
from collections import defaultdict

rdBase.DisableLog('rdApp.warning')


class CONFIG:
    # ==========================================================================
    SMILES_FILE = 'drug_smi.csv'
    MODEL_SAVE_PATH = 'features/20251112/1112moact.pth'
    OUTPUT_NPZ_PATH = 'features/20251112/1112moact.npz'

    INPUT_DIM = 74
    EDGE_DIM = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AUG_PROB_NODE_DROP = 0.2
    AUG_PROB_ATTR_MASK = 0.2
    BATCH_SIZE = 512
    EPOCHS = 500
    LEARNING_RATE = 1e-4
    GNN_EMBEDDING_DIM = 256
    GNN_LAYERS = 5
    PROJECTION_DIM = 256
    GAT_HEADS = 4
    GRAPH_LOSS_TEMP = 0.1
    NODE_LOSS_TEMP = 0.1
    LOSS_ALPHA = 0.5
    MOMENTUM_START = 0.98
    MOMENTUM_END = 0.999
    VALIDATION_SPLIT = 0.1
    NUM_WORKERS = 0
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0




class MoleculeFeaturizer:
    def __init__(self, config):
        self.config = config
        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'other']
        self.hybridizations = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
        self.structural_smarts = {
            "Specified chiral carbon": "[$([#6X4@](*)(*)(*)*),$([#6X4@H](*)(*)*)]",
            "Quaternary Nitrogen": "[$([NX4+]),$([NX4]=*)]",
            "S double-bonded to Carbon": "[$([SX1]=[#6])]", "Triply bonded N": "[$([NX1]#*)]",
            "Divalent Oxygen": "[$([OX2])]",
            "Long_chain groups": "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]",
            "Carbon_isolating": "[$([#6+0]);!$(C(F)(F)F);!$(c(:[!c]):[!c])!$([#6]=,#[!#6])]",
            "Rotatable bond": "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]",
            "Bicyclic": "[$([*R2]([*R])([*R])([*R]))].[$([*R2]([*R])([*R])([*R]))]",
            "Ortho": "*-!:aa-!:*", "Meta": "*-!:aaa-!:*", "Para": "*-!:aaaa-!:*", "Non-ring atom": "[!R]",
            "Ring atom": "[R]",
            "Macrocycle groups": "[r;!r3;!r4;!r5;!r6;!r7]",
            "Spiro-ring center": "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",
            "Unfused benzene ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1", "Fused benzene rings": "c12ccccc1cccc2",
        }
        self.functional_group_smarts = {
            "Carbonyl group": "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]", "Aldehyde": "[CX3H1](=O)[#6]",
            "Amide": "[NX3][CX3](=[OX1])[#6]",
            "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]", "Carboxylate Ion": "[CX3](=O)[O-]",
            "Carboxylic acid": "[CX3](=O)[OX1H0-,OX2H1]",
            "Ester": "[#6][CX3](=O)[OX2H0][#6]", "Ketone": "[#6][CX3](=O)[#6]", "Ether": "[OD2]([#6])[#6]",
            "Primary or secondary amine, not amide": "[NX3;H2,H1;!$(NC=O)]", "Enamine": "[NX3][CX3]=[CX3]",
            "Azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]", "Hydrazine": "[NX3][NX3]",
            "Hydrazone": "[NX3][NX2]=[*]",
            "Imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]", "Nitrile": "[NX1]#[CX2]",
            "Nitro group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]", "Hydroxyl": "[OX2H]", "Phenol": "[OX2H][cX3]:[c]",
            "Thiol": "[SX2H]", "Sulfide": "[#16X2H0]",
            "Sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
            "Sulfonamide": "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
            "Sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]", "Halogen": "[F,Cl,Br,I]",
        }
        self._compiled_structural = {k: Chem.MolFromSmarts(p) for k, p in self.structural_smarts.items()}
        self._compiled_functional = {k: Chem.MolFromSmarts(p) for k, p in self.functional_group_smarts.items()}

    def _atom_props(self, atom):
        props = []
        symbol = atom.GetSymbol();
        symbol_feature = [0] * len(self.atom_types);
        symbol_feature[self.atom_types.index(symbol) if symbol in self.atom_types else -1] = 1;
        props.extend(symbol_feature)
        degree = atom.GetDegree();
        degree_feature = [0] * 6;
        degree_feature[min(degree, 6) - 1 if degree > 0 else 0] = 1;
        props.extend(degree_feature)
        charge = atom.GetFormalCharge();
        charge_feature = [0] * 5;
        charge_feature[charge + 2 if -2 <= charge <= 2 else (4 if charge > 2 else 0)] = 1;
        props.extend(charge_feature)
        hybrid = str(atom.GetHybridization());
        hybrid_feature = [1 if h == hybrid else 0 for h in self.hybridizations];
        props.extend(hybrid_feature)
        props.extend([atom.GetIsAromatic(), atom.IsInRing()])
        return props

    def _match_patterns(self, mol, compiled_patterns):
        x = torch.zeros(len(compiled_patterns), mol.GetNumAtoms())
        for i, pattern in enumerate(compiled_patterns.values()):
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                if matches: x[i, sorted(list(set(sum(matches, ()))))] = 1
        return x.T

    def get_features(self, mol):
        return torch.cat([torch.tensor([self._atom_props(atom) for atom in mol.GetAtoms()], dtype=torch.float),
                          self._match_patterns(mol, self._compiled_structural),
                          self._match_patterns(mol, self._compiled_functional)], dim=1)

    def _get_bond_features(self, bond):
        bond_type = bond.GetBondType()
        features = [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.IsInRing()
        ]
        return [int(f) for f in features]

    def smiles_to_graph(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if not mol: return None
        x = self.get_features(mol)
        edge_list = []
        edge_attr_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_features = self._get_bond_features(bond)
            edge_list.append((i, j))
            edge_attr_list.append(bond_features)
            edge_list.append((j, i))
            edge_attr_list.append(bond_features)
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.config.EDGE_DIM), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MolMultiTaskDataset(Dataset):
    def __init__(self, smiles_df, config, featurizer):
        super(MolMultiTaskDataset, self).__init__()
        self.config = config
        self.smiles_map = smiles_df.set_index('DrugID')['smi'].to_dict()
        print("Step 1: Converting all SMILES to graphs with new features...")
        self.graphs = {
            drug_id: featurizer.smiles_to_graph(smi)
            for drug_id, smi in tqdm(self.smiles_map.items(), desc="Converting SMILES")
        }
        self.graphs = {k: v for k, v in self.graphs.items() if v is not None and v.num_nodes > 0}
        self.anchors = list(self.graphs.keys())
        print(f"Dataset initialized. Found {len(self.anchors)} valid molecules for training.")

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor_id = self.anchors[index]
        anchor_original = self.graphs[anchor_id].clone()
        anchor_augmented, kept_node_indices = augment_graph(anchor_original.clone(), self.config)
        anchor_original.kept_node_indices = kept_node_indices
        return anchor_original, anchor_augmented


def augment_graph(data, config):
    num_nodes = data.num_nodes
    if num_nodes == 0: return data.clone(), torch.tensor([], dtype=torch.long)
    num_nodes_to_keep = int(num_nodes * (1 - config.AUG_PROB_NODE_DROP))
    if num_nodes_to_keep < 1: num_nodes_to_keep = 1
    nodes_to_keep_mask = torch.randperm(num_nodes)[:num_nodes_to_keep]
    aug_data = data.subgraph(nodes_to_keep_mask)
    if aug_data.num_nodes > 0:
        mask = torch.rand(aug_data.x.size()) < config.AUG_PROB_ATTR_MASK
        aug_data.x[mask] = 0.0
    return aug_data, nodes_to_keep_mask.sort().values


class GNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_dim = config.GNN_EMBEDDING_DIM
        num_heads = config.GAT_HEADS
        num_layers = config.GNN_LAYERS

        self.atom_embedding = nn.Linear(config.INPUT_DIM, emb_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATConv(
                emb_dim,
                emb_dim,
                heads=num_heads,
                concat=False,
                edge_dim=config.EDGE_DIM
            )
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(emb_dim))

        self.gate_nn = nn.Sequential(
            nn.Linear(emb_dim * num_layers, num_layers),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.atom_embedding(x))

        layer_outputs = []

        for i, conv in enumerate(self.convs):
            h = x;
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = h + x
            layer_outputs.append(x)



        pooled_outputs = [global_add_pool(h, batch) for h in layer_outputs]


        z_stacked = torch.cat(pooled_outputs, dim=1)


        gate_weights = self.gate_nn(z_stacked)


        z_tensor = torch.stack(pooled_outputs, dim=1)

        z_aggregated = torch.sum(z_tensor * gate_weights.unsqueeze(-1), dim=1)

        h_last_layer = layer_outputs[-1]

        return z_aggregated, h_last_layer


class MolMultiTaskGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.student_gnn = GNNEncoder(config)
        self.teacher_gnn = copy.deepcopy(self.student_gnn)
        for param in self.teacher_gnn.parameters(): param.requires_grad = False

        projector_input_dim = config.GNN_EMBEDDING_DIM

        self.projector = nn.Sequential(
            nn.Linear(projector_input_dim, config.GNN_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(config.GNN_EMBEDDING_DIM, config.PROJECTION_DIM)
        )



    def forward(self, anchor_orig, anchor_aug):
        with torch.no_grad():

            z_teacher, h_teacher = self.teacher_gnn(anchor_orig)

        z_student_aug, h_student_aug = self.student_gnn(anchor_aug)
        z_student_aug = self.projector(z_student_aug)

        return {
            "z_student_aug": z_student_aug, "z_teacher": z_teacher,
            "h_student_aug": h_student_aug, "h_teacher": h_teacher
        }


class InfoNCELoss(nn.Module):
    def __init__(self,
                 temperature=0.1): super().__init__(); self.temperature = temperature; self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_student, z_teacher):
        features = F.normalize(torch.cat([z_student, z_teacher], dim=0), dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=features.device)
        sim_matrix.masked_fill_(mask, -9e15)
        labels = (torch.arange(z_student.shape[0], device=features.device) + z_student.shape[0])
        return self.criterion(sim_matrix[:z_student.shape[0]], labels)


class NodeLevelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__();
        self.temperature = temperature;
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, student_nodes, teacher_nodes, original_batch, augmented_batch):
        total_loss, num_valid = 0.0, 0
        for i in range(original_batch.num_graphs):
            s_nodes = student_nodes[slice(augmented_batch.ptr[i], augmented_batch.ptr[i + 1])]
            if s_nodes.numel() == 0: continue
            t_nodes_all = teacher_nodes[slice(original_batch.ptr[i], original_batch.ptr[i + 1])]
            kept_indices = original_batch.kept_node_indices[slice(augmented_batch.ptr[i], augmented_batch.ptr[i + 1])]
            t_nodes_aligned = t_nodes_all[kept_indices]
            scores = torch.matmul(F.normalize(s_nodes), F.normalize(t_nodes_aligned).T) / self.temperature
            total_loss += self.loss_fct(scores, torch.arange(s_nodes.size(0), device=scores.device))
            num_valid += 1
        return total_loss / num_valid if num_valid > 0 else torch.tensor(0.0, device=student_nodes.device)




def compute_losses(outputs, batch_data, loss_fns, device, config):
    anchor_orig, anchor_aug = batch_data
    loss_vae_align = torch.tensor(0.0, device=device)
    loss_g = loss_fns['graph'](outputs["z_student_aug"], outputs["z_teacher"])
    loss_n_val = loss_fns['node'](outputs["h_student_aug"], outputs["h_teacher"], anchor_orig, anchor_aug)
    loss_n = loss_n_val if isinstance(loss_n_val, torch.Tensor) else torch.tensor(loss_n_val, device=device)
    loss_s = torch.tensor(0.0, device=device)
    total_loss = (1 - config.LOSS_ALPHA) * loss_g + config.LOSS_ALPHA * loss_n
    return {'total': total_loss, 'graph': loss_g, 'node': loss_n, 'sim': loss_s, 'vae_align': loss_vae_align}


def train_model(config, featurizer, device):
    print("--- Stage 1: Training Model (using GATConv + Adaptive Aggregation) ---")
    smiles_df = pd.read_csv(config.SMILES_FILE, header=0, sep='\t')
    dataset = MolMultiTaskDataset(smiles_df, config, featurizer)
    if not dataset.anchors:
        print("Error: No valid molecules found in the dataset.")
        return None
    val_size = int(len(dataset) * config.VALIDATION_SPLIT);
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Data split: {len(train_dataset)} for training, {len(val_dataset)} for validation.")

    def collate_fn(batch):
        anchor_orig, anchor_aug = zip(*batch)
        return Batch.from_data_list(list(anchor_orig)), Batch.from_data_list(list(anchor_aug))

    train_loader = TorchDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True,
                                   collate_fn=collate_fn, num_workers=config.NUM_WORKERS)
    val_loader = TorchDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                 num_workers=config.NUM_WORKERS)


    model = MolMultiTaskGNN(config).to(device)
    loss_fns = {'graph': InfoNCELoss(config.GRAPH_LOSS_TEMP), 'node': NodeLevelContrastiveLoss(config.NODE_LOSS_TEMP)}
    optimizer = optim.AdamW(model.student_gnn.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_val_loss = float('inf')
    print("Starting training loop with validation checkpointing...")
    for epoch in range(config.EPOCHS):
        model.train()
        train_losses = defaultdict(float)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Train]", leave=False)
        for batch in pbar:
            anchor_orig, anchor_aug = [item.to(device) for item in batch]

            outputs = model(anchor_orig, anchor_aug)

            losses = compute_losses(outputs, batch, loss_fns, device, config)
            if not torch.isfinite(losses['total']): continue
            optimizer.zero_grad();
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.student_gnn.parameters(), max_norm=config.GRAD_CLIP_NORM)
            optimizer.step()
            with torch.no_grad():
                m = 1.0 - (1.0 - config.MOMENTUM_START) * (1.0 + np.cos(np.pi * epoch / config.EPOCHS)) / 2.0
                for param_s, param_t in zip(model.student_gnn.parameters(), model.teacher_gnn.parameters()):
                    param_t.data = param_t.data * m + param_s.data * (1.0 - m)
            for k, v in losses.items():
                if torch.isfinite(v): train_losses[k] += v.item()
            pbar.set_postfix({k: f"{v / (pbar.n + 1):.4f}" for k, v in train_losses.items()})
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                anchor_orig, anchor_aug = [item.to(device) for item in batch]
                outputs = model(anchor_orig, anchor_aug)
                val_loss += compute_losses(outputs, batch, loss_fns, device, config)['total'].item()
        avg_val_loss = val_loss / len(val_loader)
        tqdm.write(
            f"Epoch {epoch + 1} | Avg Train Loss: {train_losses['total'] / len(train_loader):.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.student_gnn.state_dict(), config.MODEL_SAVE_PATH)
            tqdm.write(f"✅ New best model saved with validation loss: {best_val_loss:.4f}")

    print("\nTraining finished.")

    best_model = GNNEncoder(config)
    best_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))
    return best_model


def generate_embeddings(trained_gnn, config, featurizer, device):
    print("\n--- Stage 2: Generating Embeddings with Trained GAT Model (Adaptive Aggregation) ---")
    trained_gnn.to(device).eval()
    smiles_df = pd.read_csv(config.SMILES_FILE, header=0, sep='\t')

    graphs, valid_ids = [], []
    for _, row in tqdm(smiles_df.iterrows(), total=len(smiles_df), desc="Converting SMILES"):
        graph = featurizer.smiles_to_graph(row['smi'])
        if graph and graph.num_nodes > 0:
            graphs.append(graph)
            valid_ids.append(row['DrugID'])

    loader = DataLoader(graphs, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=config.NUM_WORKERS)
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating embeddings"):
            batch = batch.to(device)


            z_aggregated, _ = trained_gnn(batch)

            embeddings.append(z_aggregated.cpu().numpy())

    embedding_matrix = np.vstack(embeddings)
    os.makedirs(os.path.dirname(config.OUTPUT_NPZ_PATH), exist_ok=True)
    np.savez(config.OUTPUT_NPZ_PATH, embeddings=embedding_matrix, drug_ids=np.array(valid_ids))

    print(f"\n✅ Embeddings saved to: {config.OUTPUT_NPZ_PATH}")
    print(f"   Shape: {embedding_matrix.shape}")


def main():
    config = CONFIG()
    device = torch.device(config.DEVICE)
    featurizer = MoleculeFeaturizer(config)


    mol = Chem.MolFromSmiles('CCO')
    test_graph = featurizer.smiles_to_graph('CCO')
    config.INPUT_DIM = test_graph.x.shape[1]
    config.EDGE_DIM = test_graph.edge_attr.shape[1]
    if config.EDGE_DIM == 0: config.EDGE_DIM = 5

    print(f"Atom feature dimension automatically set to: {config.INPUT_DIM}")
    print(f"Edge feature dimension automatically set to: {config.EDGE_DIM}")


    print(f"\n--- Stage 1: Skipping Training ---")
    print(f"Loading best model from: {config.MODEL_SAVE_PATH}")


    trained_model = GNNEncoder(config)


    try:

        trained_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))
    except:

        print("Warning: Could not load with weights_only=True. Falling back to unsafe load.")
        trained_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=False))

    trained_model.to(device)


    if trained_model:
        generate_embeddings(trained_model, config, featurizer, device)


if __name__ == '__main__':
    main()
