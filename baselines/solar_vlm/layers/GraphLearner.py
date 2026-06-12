# File: layers/GraphLearner.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    GAT layer with:
      - hard kNN structural mask (bool)
      - additive logits bias (distance bias + dynamic edge bias)
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W, gain=1.414)

        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

        # 关键：bias 系数建议小初始化（配置1去掉dense self-attn时更稳）
        self.dist_coef = nn.Parameter(torch.tensor(0.1))
        self.edge_coef = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, knn_mask, dist_bias=None, edge_bias=None):
        """
        h:        [B, N, Fin]
        knn_mask: [N, N] bool, True=允许注意力 (包含 self-loop)
        dist_bias:[N, N] float, logits bias (例如 log(sim+eps))
        edge_bias:[B, N, N] float, logits bias (例如 log(w_ij+eps))
        """
        B, N, _ = h.shape

        Wh = torch.matmul(h, self.W)  # [B, N, Fout]

        a_input = self._prepare_attentional_mechanism_input(Wh)          # [B, N, N, 2Fout]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))     # [B, N, N]

        logits = e

        if dist_bias is not None:
            logits = logits + self.dist_coef * dist_bias.to(dtype=logits.dtype).unsqueeze(0)  # [B,N,N]
        if edge_bias is not None:
            logits = logits + self.edge_coef * edge_bias.to(dtype=logits.dtype)               # [B,N,N]

        # hard mask：非kNN边置为极小
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~knn_mask.unsqueeze(0), neg_inf)

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout_layer(attn)

        out = torch.matmul(attn, Wh)  # [B, N, Fout]
        return F.elu(out)

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, Fout = Wh.shape
        Wh_i = Wh.repeat_interleave(N, dim=1)  # [B, N*N, Fout]
        Wh_j = Wh.repeat(1, N, 1)              # [B, N*N, Fout]
        all_comb = torch.cat([Wh_i, Wh_j], dim=-1)
        return all_comb.view(B, N, N, 2 * Fout)


class GraphLearner(nn.Module):
    """
    配置1专用：只做结构化图传播（Route A）
    - kNN mask 决定结构
    - 距离相似度作为 logits bias（不参与mask）
    - 动态边权作为 logits bias（不参与mask）
    - 不做 global dense attention / fusion
    """
    def __init__(self, coords, in_dim, out_dim=None, num_layers=2, k=5, dropout=0.1, eps=1e-6):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.k = int(k)
        self.dropout_p = float(dropout)
        self.eps = float(eps)

        # coords: dict 或 list
        if isinstance(coords, dict):
            coord_list = [coords[sid] for sid in sorted(coords.keys())]
        else:
            coord_list = coords
        self.num_nodes = len(coord_list)

        # ===== precompute distance(sim) + knn mask =====
        dist_km = self._haversine_dist_matrix(coord_list)                 # [N,N]
        dist_sim = self._dist_to_sim(dist_km)                             # (0,1]
        knn_mask = self._build_knn_mask(dist_km, k=self.k)                # bool [N,N], True=allowed

        # 距离 bias 用 log(sim+eps)：越近越接近0，越远越负（在mask内起到soft偏置）
        dist_bias = torch.log(dist_sim.clamp_min(self.eps))               # [N,N]

        self.register_buffer("knn_mask_buffer", knn_mask, persistent=False)
        self.register_buffer("dist_bias_buffer", dist_bias, persistent=False)
        self.register_buffer("dist_sim_buffer", dist_sim, persistent=False)

        # ===== dynamic edge MLP: (feat_diff_norm, dist_sim) -> w_ij in (0,1) =====
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # ===== GAT stack =====
        self.gat_layers = nn.ModuleList()
        cur = in_dim
        for i in range(num_layers):
            nxt = out_dim if i == num_layers - 1 else in_dim
            self.gat_layers.append(GraphAttentionLayer(cur, nxt, dropout=dropout))
            cur = nxt

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(in_dim if i < num_layers - 1 else out_dim)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self.residual_projection = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def _haversine_dist_matrix(self, coord_list):
        N = len(coord_list)
        dist = torch.zeros(N, N)
        R = 6371.0
        for i in range(N):
            lon_i, lat_i = coord_list[i]
            lat_i = math.radians(lat_i); lon_i = math.radians(lon_i)
            for j in range(i + 1, N):
                lon_j, lat_j = coord_list[j]
                lat_j = math.radians(lat_j); lon_j = math.radians(lon_j)
                dlat = lat_j - lat_i
                dlon = lon_j - lon_i
                a = math.sin(dlat / 2) ** 2 + math.cos(lat_i) * math.cos(lat_j) * math.sin(dlon / 2) ** 2
                c = 2 * math.asin(math.sqrt(a))
                d = R * c
                dist[i, j] = d
                dist[j, i] = d
        dist.fill_diagonal_(0.0)
        return dist

    def _dist_to_sim(self, dist_km):
        # 自适应 sigma（中位数距离）
        mask = dist_km > 0
        sigma = dist_km[mask].median() if mask.any() else torch.tensor(100.0)
        sim = torch.exp(-dist_km / (sigma + 1e-6))
        sim.fill_diagonal_(1.0)
        return sim

    def _build_knn_mask(self, dist_km, k):
        N = dist_km.size(0)
        # 排除自身：对角设 inf
        d = dist_km.clone()
        d.fill_diagonal_(float("inf"))
        # 每行选 k 个最近
        _, idx = torch.topk(d, k=min(k, N - 1), largest=False)
        mask = torch.zeros(N, N, dtype=torch.bool)
        for i in range(N):
            mask[i, idx[i]] = True
        # 对称 + self-loop
        mask = mask | mask.t()
        mask.fill_diagonal_(True)
        return mask

    def _learn_edge_bias(self, node_features):
        """
        node_features: [B, N, F]
        return edge_bias: [B, N, N] (log(w_ij+eps))
        """
        B, N, _ = node_features.shape
        diff = node_features.unsqueeze(2) - node_features.unsqueeze(1)  # [B,N,N,F]
        feat_diff = diff.norm(dim=-1, keepdim=True)                     # [B,N,N,1]
        dist_sim = self.dist_sim_buffer.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)  # [B,N,N,1]

        edge_in = torch.cat([feat_diff, dist_sim], dim=-1)              # [B,N,N,2]
        w = self.edge_mlp(edge_in).squeeze(-1)                          # [B,N,N] in (0,1)

        # self-loop w=1 -> log(1)=0，不额外偏置
        eye = torch.eye(N, device=w.device, dtype=w.dtype).unsqueeze(0)
        w = w * (1 - eye) + 1.0 * eye

        return torch.log(w.clamp_min(self.eps))

    def forward(self, X):
        """
        X: [B, N, D]
        """
        B, N, _ = X.shape
        assert N == self.num_nodes, f"GraphLearner expects N={self.num_nodes}, got {N}"

        knn_mask = self.knn_mask_buffer.to(device=X.device)
        # 防止出现某行全False（会导致softmax异常）
        if not torch.all(knn_mask.any(dim=-1)):
            raise RuntimeError("kNN mask has an all-false row; check k/self-loop/symmetry.")

        dist_bias = self.dist_bias_buffer.to(device=X.device, dtype=X.dtype)
        edge_bias = self._learn_edge_bias(X)  # [B,N,N]

        residual = self.residual_projection(X)

        H = X
        for i, gat in enumerate(self.gat_layers):
            H = gat(H, knn_mask, dist_bias=dist_bias, edge_bias=edge_bias)
            H = self.layer_norms[i](H)
            if i < len(self.gat_layers) - 1:
                H = self.dropout(H)

        return H + residual
