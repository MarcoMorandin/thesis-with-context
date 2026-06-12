import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import CLIPModel, CLIPProcessor
from datetime import datetime, timedelta

sys.path.append("../")
from layers.Embed import PatchEmbedding
from layers.GraphLearner import GraphLearner


def _get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.dirname(os.path.dirname(current_dir)),
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),
    ]
    for base in candidates:
        if os.path.isdir(os.path.join(base, 'dataset')) or os.path.isfile(os.path.join(base, 'README.md')):
            return base
    return current_dir


PROJECT_ROOT = _get_project_root()


class PatchMemoryBank(nn.Module):
    """Patch记忆库（修复：n >> max_size 时不再整批覆盖）"""
    def __init__(self, max_size, patch_size, feature_dim, max_write=None, use_cosine=True):
        super().__init__()
        self.max_size = int(max_size)
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.max_write = int(max_write) if max_write is not None else min(32, self.max_size)  # 每步最多写入多少条
        self.use_cosine = use_cosine

        self.register_buffer("patches", torch.zeros(self.max_size, self.feature_dim), persistent=False)
        self.register_buffer("valid_size", torch.zeros((), dtype=torch.long), persistent=False)
        self.ptr = 0

    @torch.no_grad()
    def update(self, new_patches):
        """
        new_patches: [N, P, D] 或 [N, ..., D]，这里假设 dim=1 是 patch/token 维
        """
        # In DataParallel, only update on primary GPU to avoid cross-replica buffer conflicts
        dev = new_patches.device
        if dev.index is not None and dev.index != 0:
            return
        if new_patches.device != self.patches.device:
            new_patches = new_patches.to(self.patches.device, non_blocking=True)

        # 压缩成 [N, D]
        new_flat = new_patches.mean(dim=1)
        n = new_flat.size(0)

        # ===== 关键修复：限制每步写入条数 =====
        if n > self.max_write:
            # 随机抽样 max_write 条（也可以改为均匀采样/基于多样性采样）
            idx = torch.randperm(n, device=new_flat.device)[:self.max_write]
            new_flat = new_flat.index_select(0, idx)
            n = self.max_write

        # 可选：归一化，便于 cosine 检索更稳定
        if self.use_cosine:
            new_flat = F.normalize(new_flat, dim=-1)

        # ===== 环形写入 =====
        end = self.ptr + n
        if end <= self.max_size:
            self.patches[self.ptr:end] = new_flat
        else:
            first = self.max_size - self.ptr
            self.patches[self.ptr:] = new_flat[:first]
            self.patches[:end % self.max_size] = new_flat[first:]
        self.ptr = (self.ptr + n) % self.max_size

        # 更新有效长度（避免从全 0 槽位检索）
        self.valid_size.fill_(min(self.max_size, int(self.valid_size.item()) + n))

    @torch.no_grad()
    def retrieve(self, query_patches, top_k=5):
        dev = self.patches.device
        if query_patches.device != dev:
            query_patches = query_patches.to(dev, non_blocking=True)

        query_flat = query_patches.mean(dim=1)  # [Q, D]
        if self.use_cosine:
            query_flat = F.normalize(query_flat, dim=-1)

        vs = int(self.valid_size.item())
        if vs == 0:
            # 还没写入任何内容：返回零
            Q = query_flat.size(0)
            gathered = torch.zeros(Q, top_k, self.feature_dim, device=dev)
            idx = torch.zeros(Q, top_k, dtype=torch.long, device=dev)
            return gathered, idx

        memory_flat = self.patches[:vs]  # 只在有效区检索

        sim = torch.matmul(query_flat, memory_flat.T)  # [Q, vs]
        k = min(int(top_k), vs)
        _, idx = sim.topk(k, dim=-1)

        gathered = memory_flat.index_select(0, idx.reshape(-1)).view(idx.shape + (memory_flat.size(1),))
        return gathered, idx


class ModalityGate(nn.Module):
    def __init__(self, temporal_dim, multimodal_dim, hidden_dim=128):
        super().__init__()
        
        # 综合门控（简化设计，直接学习融合权重）
        self.gate = nn.Sequential(
            nn.Linear(temporal_dim + multimodal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, temporal_feat, multimodal_feat):
        combined = torch.cat([temporal_feat, multimodal_feat], dim=-1)
        gate_weight = self.gate(combined)
        return gate_weight.squeeze(-1)



class VisualAdapter(nn.Module):
    """视觉特征适配层"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.adapter(x)


class VisualTemporalEncoder(nn.Module):
    def __init__(self, in_dim, n_frames=8, n_layers=2, n_heads=4,
                 ff_dim=None, dropout=0.1, pooling: str = "attn",
                 attn_hidden: int = None):
        super().__init__()
        if ff_dim is None:
            ff_dim = max(1024, in_dim * 4)
        self.n_frames = n_frames
        self.in_dim = in_dim
        self.pooling = pooling.lower()

        # 可训练的位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, n_frames, in_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ✅ learnable attention pooling scorer
        if attn_hidden is None:
            attn_hidden = max(128, in_dim // 4)
        self.attn_scorer = nn.Sequential(
            nn.Linear(in_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1, bias=False)  # [N,T,D] -> [N,T,1]
        )

        # 输出投影（保持你原来的设计）
        self.pool_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim)
        )

    def forward(self, x):
        """
        x: [B, S, T, D]
        return: [B, S, D]
        """
        B, S, T, D = x.shape

        # 截断/补齐到 n_frames
        if T > self.n_frames:
            x = x[:, :, -self.n_frames:, :]
            T = self.n_frames
        elif T < self.n_frames:
            pad_len = self.n_frames - T
            pad = x[:, :, -1:, :].expand(B, S, pad_len, D)
            x = torch.cat([x, pad], dim=2)
            T = self.n_frames

        # 位置编码
        pos = self.pos_emb[:, :T, :]              # [1,T,D]
        x = x + pos.unsqueeze(1)                  # [B,S,T,D]

        # 合并 batch 和站点维
        x = x.view(B * S, T, D)                   # [BS,T,D]
        h = self.encoder(x)                       # [BS,T,D]

        # ===== pooling 分支 =====
        if self.pooling == "last":
            pooled = h[:, -1, :]                  # [BS,D]

        elif self.pooling == "mean":
            pooled = h.mean(dim=1)                # [BS,D]

        elif self.pooling == "mean_last":
            pooled = 0.7 * h.mean(dim=1) + 0.3 * h[:, -1, :]

        elif self.pooling == "attn":
            # learnable attention weights over frames
            logits = self.attn_scorer(h).squeeze(-1)       # [BS,T]
            alpha = torch.softmax(logits, dim=1)           # [BS,T]
            pooled = (h * alpha.unsqueeze(-1)).sum(dim=1)  # [BS,D]

        else:
            raise ValueError(f"Unknown pooling={self.pooling}. Use last/attn/mean/mean_last")

        pooled = self.pool_proj(pooled)            # [BS,D]
        return pooled.view(B, S, D)

class CrossStationAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        
        # 特征归一化层（确保输入 scale 一致）
        self.temporal_norm = nn.LayerNorm(d_model)
        self.mm_norm = nn.LayerNorm(d_model)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # 残差缩放因子（可学习）
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, temporal_feat, mm_feat):
        """
        temporal_feat: [B, S, D] - 时序特征
        mm_feat: [B, S, D] - 多模态特征
        返回: [B, S, D] - 融合后的特征
        """
        # 先归一化确保 scale 一致
        temporal_feat = self.temporal_norm(temporal_feat)
        mm_feat = self.mm_norm(mm_feat)
        
        # 站点间自注意力
        h, _ = self.self_attn(temporal_feat, temporal_feat, temporal_feat)
        temporal_feat = self.norm1(temporal_feat + h)
        
        # 跨注意力（temporal 作为 Q，multimodal 作为 KV）
        h, _ = self.cross_attn(temporal_feat, mm_feat, mm_feat)
        # 使用可学习的残差缩放
        fused = self.norm2(temporal_feat + self.residual_scale * h)
        
        # FFN
        h = self.ffn(fused)
        fused = self.norm3(fused + h)
        
        return fused


class Model(nn.Module):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        
        self.use_mem_gate = getattr(config, 'use_mem_gate', True)
        # 【修复】统一命名，使用 disable_xxx 更清晰
        self.disable_visual = bool(getattr(config, "disable_visual", False))
        self.disable_text   = bool(getattr(config, "disable_text",   False))
        self.disable_gnn    = bool(getattr(config, "disable_gnn",    False))
        self.disable_csa    = bool(
            getattr(config, "disable_csa",
                    getattr(config, "disable_cross_site_attn", False))
        )
        
        self.register_buffer("_dev_buf", torch.empty(0), persistent=False)
        
        # 站点配置 — configurable via config; defaults to the Hebei 8-station setup
        _default_station_list = [
            'station00', 'station01', 'station02', 'station04',
            'station06', 'station07', 'station08', 'station09',
        ]
        self.station_list = getattr(config, 'station_list', _default_station_list)
        self.num_stations = len(self.station_list)

        _n_features = getattr(config, 'enc_in', 14)
        _default_feature_schema = [f'feat_{i}' for i in range(_n_features)]
        self.feature_schema = getattr(
            config, "station_feature_order",
            getattr(config, "feature_schema", _default_feature_schema),
        )
        self.feature_name_to_idx = {name: idx for idx, name in enumerate(self.feature_schema)}

        _default_station_positions = {
            'station00': (0.5243, 0.4855), 'station01': (0.7489, 0.4734),
            'station02': (0.4569, 0.4846), 'station04': (0.5168, 0.3545),
            'station06': (0.4879, 0.5880), 'station07': (0.4070, 0.6107),
            'station08': (0.4301, 0.6050), 'station09': (0.5340, 0.4245),
        }
        self.station_positions = getattr(
            config, 'station_positions',
            {s: _default_station_positions.get(s, (0.5, 0.5)) for s in self.station_list},
        )

        _default_station_coords = {
            'station00': (114.95139, 38.04778), 'station01': (117.45722, 38.18306),
            'station02': (114.19887, 38.05728), 'station04': (114.86767, 39.51550),
            'station06': (114.54841, 36.89891), 'station07': (113.64187, 36.64403),
            'station08': (113.89999, 36.70761), 'station09': (115.059855, 38.731417),
        }
        self.station_coords = getattr(
            config, 'station_coords',
            {s: _default_station_coords.get(s, (0.0, 0.0)) for s in self.station_list},
        )
        
        # 时间配置
        self.start_dt = datetime.strptime(config.start_time, '%Y-%m-%d %H:%M')
        self.freq = getattr(config, 'freq', 't')
        self.freq_minutes = {'s': 1, 't': 15, '15min': 15, 'h': 60}.get(self.freq, 15)
        self.image_tolerance_minutes = getattr(config, 'image_tolerance_minutes', 5)
        self.image_freq_minutes = int(getattr(config, "image_freq_minutes", 10)) 
        
        # 【修复】统一默认值
        self.modal_dropout_rate = getattr(config, 'modal_dropout_rate', 0.1)
        
        # Patch记忆库
        self.patch_memory_bank = PatchMemoryBank(
            max_size=config.patch_memory_size,
            patch_size=config.patch_len,
            feature_dim=config.d_model,
        )
        
# ===================== VLM backend =====================
        self.vlm_type = getattr(config, "vlm_type", "clip").lower()

# ===== 维度策略：qwen3vl 默认 2048 =====
        default_dim = 2048 if self.vlm_type in ("qwen3vl", "qwen3_vl", "qwen3-vl") else 512

        self.image_feature_dim = int(getattr(config, "vlm_image_dim",
                                getattr(config, "vlm_embed_dim", default_dim)))
        self.text_feature_dim  = int(getattr(config, "vlm_text_dim",
                                getattr(config, "vlm_embed_dim", default_dim)))

        # 离线视觉特征
        self.use_offline_vision = getattr(config, "use_offline_vision", True)
        self.vision_feat_dir = getattr(config, "vision_feat_dir",
                                    os.path.join(PROJECT_ROOT, 'vision_feats_qwen3vl'))

        # ✅ 关键：仅当视觉未禁用时才初始化 store
        self.vision_store = None
        if (not self.disable_visual) and self.use_offline_vision:
            from src.SolarVLM.vision_store import VisionFeatureStore
            self.vision_store = VisionFeatureStore(self.vision_feat_dir, cache_size=8192, feat_dim=self.image_feature_dim)


        self.clip_model = None
        self.clip_processor = None
        self.qwen_embedder = None

        if self.vlm_type == "clip":
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model_path = getattr(
                config, "clip_model_path",
                os.path.join(PROJECT_ROOT, 'clip-vit-base-patch32')
            )
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_path)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_path)
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False

        elif self.vlm_type in ("qwen3vl", "qwen3_vl", "qwen3-vl"):
            if not self.disable_text:
                from src.SolarVLM.qwen3_vl_embedding import Qwen3VLEmbedder
                self.qwen3_vl_model_path = getattr(
                    config, "qwen3_vl_model_path",
                    os.path.join(PROJECT_ROOT, 'QwenQwen3-VL-Embedding-2B')
                )
                # 参考官方用法：Qwen3VLEmbedder(model_name_or_path=...):contentReference[oaicite:5]{index=5}
                self.qwen_embedder = Qwen3VLEmbedder(model_name_or_path=self.qwen3_vl_model_path)
                # 保险起见：如果 embedder 内部暴露了 model，可 eval
                if hasattr(self.qwen_embedder, "model"):
                    self.qwen_embedder.model.eval()

        else:
            raise ValueError(f"Unknown vlm_type={self.vlm_type}")


        
        self._init_modules(config)
        
        self.vision_hits = 0
        self.vision_requests = 0
        self.nonnegative = getattr(config, "nonnegative", False)
        self.n_vision_frames = getattr(config, "num_frames", 8)

        # 视觉时序编码器
        vision_pool = getattr(config, "vision_pool", "attn")  
        self.visual_temporal = VisualTemporalEncoder(
            in_dim=self.image_feature_dim,
            n_frames=self.n_vision_frames,
            n_layers=getattr(config, "vision_temporal_layers", 2),
            n_heads=4,
            ff_dim=getattr(config, "vision_ff_dim", None),
            dropout=config.dropout,
            pooling=vision_pool, 
        )

        
    def _pick_head_count(self, dim):
        for h in [8, 6, 4, 3, 2, 1]:
            if dim % h == 0:
                return h
        return 1
    
    def _module_device(self, m=None):
        if isinstance(m, nn.Module):
            p = next(m.parameters(), None)
            if p is not None:
                return p.device
        return self._dev_buf.device
    
    def _init_modules(self, config):
        # ========== 时序骨干 ==========
        self.patch_embedding = PatchEmbedding(
            config.d_model, config.patch_len, config.stride, config.padding, config.dropout
        )
        self.head_nf = config.d_model * int(
            (config.seq_len - config.patch_len) / config.stride + 2
        )
        self.flatten = nn.Flatten(start_dim=-2)
        
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout),
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        # 记忆模块
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model, num_heads=4,
            dropout=config.dropout, batch_first=True
        )
        
        # 站点特征聚合
        self.station_feature_layers = getattr(config, 'station_feature_layers', 2)
        temporal_heads = self._pick_head_count(config.d_model)
        
        self.temporal_feature_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.d_model, num_heads=temporal_heads,
                dropout=config.dropout, batch_first=True
            ) for _ in range(self.station_feature_layers)
        ])
        self.temporal_feature_norms = nn.ModuleList([
            nn.LayerNorm(config.d_model) for _ in range(self.station_feature_layers)
        ])
        self.temporal_feature_query = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        self.memory_pred_in = nn.Linear(config.pred_len, config.d_model)
        self.memory_pred_out = nn.Linear(config.d_model, config.pred_len)
        
        self.memory_feature_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.d_model, num_heads=temporal_heads,
                dropout=config.dropout, batch_first=True
            ) for _ in range(self.station_feature_layers)
        ])
        self.memory_feature_norms = nn.ModuleList([
            nn.LayerNorm(config.d_model) for _ in range(self.station_feature_layers)
        ])
        self.memory_feature_query = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        # 站点注意力
        self.station_attention = nn.MultiheadAttention(
            embed_dim=config.d_model, num_heads=temporal_heads,
            dropout=config.dropout, batch_first=True
        )
        self.station_attn_norm = nn.LayerNorm(config.d_model)
        self.station_weights = nn.Parameter(torch.zeros(self.num_stations))
        
        # Memory fusion gate
        if self.use_mem_gate:
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
                nn.Softmax(dim=-1),
            )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # ========== 多模态模块 ==========
        # ===== Text encoder: only build when text branch is enabled =====
        if not self.disable_text:
            from src.SolarVLM.text_encoders import HybridTextEncoder
            self.text_encoder = HybridTextEncoder(
                backend=self.vlm_type,
                clip_model=self.clip_model,
                clip_processor=self.clip_processor,
                qwen_embedder=self.qwen_embedder,
                num_stations=self.num_stations,
                num_features=len(self.feature_schema),
                d_model=config.d_model,
                dropout=config.dropout,
                out_dim=self.text_feature_dim,
                freq_minutes=self.freq_minutes,
                pred_len=config.pred_len,
                ctx_short=12,
                ctx_long=48,
                cache_size=20000,
                normalize_text_emb=True,
                station_list=self.station_list,
                station_coords=self.station_coords,
                year_hint=self.start_dt.year,
            )
        else:
            self.text_encoder = None


        self.visual_adapter = VisualAdapter(self.image_feature_dim, self.image_feature_dim)
        
        self.graph_learner = GraphLearner(
            coords=self.station_coords,
            in_dim=config.d_model,
            out_dim=config.d_model,
            num_layers=getattr(config, "gnn_layers", 2),
            k=getattr(config, "gnn_k", 5),
            dropout=config.dropout,
        )
        
        self.visual_proj = nn.Sequential(
            nn.Linear(self.image_feature_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        if not self.disable_text:
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        else:
            # not used when disable_text=True, keep a harmless placeholder
            self.text_proj = nn.Identity()
        
        # 模态门控
        self.modality_gate = ModalityGate(
            temporal_dim=config.d_model,
            multimodal_dim=config.d_model,
            hidden_dim=128
        )
        
        # 跨站点注意力模块
        self.cross_station_attn = CrossStationAttention(
            d_model=config.d_model,
            n_heads=4,
            dropout=config.dropout
        )
        
        # 多模态融合
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.LayerNorm(config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model)
        )

        # 多模态预测头
        self.multimodal_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.pred_len),
        )
        
        self.multimodal_scale = nn.Parameter(torch.tensor(0.0))
        
    def _ts_from_marks(self, x_mark_enc, x_mark_dec):
        """
        ts_key 取“预测起点”（forecast start），而不是 dec 的最后一个点
        规则：
        - 有 x_mark_dec: 用 x_mark_dec 在 index=label_len 的时间点作为预测起点
            * 若 x_mark_dec 只有 pred_len（没有 label_len 部分），则取 index=0
        - 否则：用 x_mark_enc 的最后一个时间点
        返回: List[str]，每个样本一个 ts_key（'YYYYmmddHHMM'）
        """
        def step_to_dt(step, year_hint, prev_month):
            month = int(step[0]) if len(step) > 0 else self.start_dt.month
            if month <= 0:
                month = self.start_dt.month
            day = max(1, int(step[1])) if len(step) > 1 else self.start_dt.day
            hour = int(step[3]) if len(step) > 3 else self.start_dt.hour

            # minute_bucket: 你的 mark 通常是 15min bucket（跟时序 freq 走）
            minute_bucket = int(step[4]) if len(step) > 4 and self.freq in ('t', '15min') else 0
            minute = minute_bucket * self.freq_minutes if self.freq in ('t', '15min') else minute_bucket

            # 年份滚动（month 变小就认为跨年）
            year = year_hint
            if prev_month is not None and month < prev_month:
                year += 1

            try:
                dt = datetime(year, month, day, hour, minute)
            except ValueError:
                safe_day = max(1, min(day, 28))
                dt = datetime(year, month, safe_day, hour, minute)
            return dt, year, month

        # 选 mark 来源
        use_dec = (x_mark_dec is not None and x_mark_dec.numel() > 0)
        use_enc = (x_mark_enc is not None and x_mark_enc.numel() > 0)

        if not use_dec and not use_enc:
            return []

        mark = (x_mark_dec if use_dec else x_mark_enc).detach().cpu().numpy()
        ts_keys = []

        label_len = int(getattr(self.config, "label_len", 0))
        pred_len  = int(getattr(self.config, "pred_len", 0))

        for seq in mark:
            year = self.start_dt.year
            prev_month = None

            # ===== 预测起点 index =====
            if use_dec:
                # 常见情况：len = label_len + pred_len
                if label_len > 0 and len(seq) > label_len:
                    target_idx = label_len
                else:
                    # 若 dec 里只有 pred_len（或 label_len 不存在），预测起点取 0
                    target_idx = 0
            else:
                # 没有 dec：退化用 enc 末尾
                target_idx = len(seq) - 1

            # 为了正确处理跨年：从头跑到 target_idx
            target_dt = None
            for j in range(target_idx + 1):
                dt, year, prev_month = step_to_dt(seq[j], year, prev_month)
                if j == target_idx:
                    target_dt = dt

            if target_dt is None:
                target_dt = self.start_dt

            ts_keys.append(target_dt.strftime('%Y%m%d%H%M'))

        return ts_keys
    def get_vision_features(self, ts_keys):
        """
        【修复】获取视觉特征
        正确实现：加载连续 n_vision_frames 帧，用 VisualTemporalEncoder 处理
        """
        dev = self._module_device(self)
        B = len(ts_keys)
        
        if self.disable_visual or (self.vision_store is None):
            return torch.zeros(B, self.num_stations, self.image_feature_dim, device=dev)

        all_feats = []
        
        for idx, ts_key in enumerate(ts_keys):
            self.vision_requests += 1
            try:
                feat_seq = self.vision_store.get_sequence(
                    ts_key,
                    n_frames=self.n_vision_frames,
                    freq_minutes=self.image_freq_minutes,     
                    tolerance_minutes=self.image_tolerance_minutes
                ).to(dev)
                
                self.vision_hits += 1
                all_feats.append(feat_seq)
                
            except Exception:
                feat_seq = torch.zeros(self.num_stations, self.n_vision_frames, self.image_feature_dim, device=dev)
                all_feats.append(feat_seq)
        
        # 堆叠: [B, 8, n_frames, 512]
        vision_seq = torch.stack(all_feats, dim=0)
        
        # 用 VisualTemporalEncoder 处理时序
        vision_agg = self.visual_temporal(vision_seq)  # [B, 8, 512]
        
        # 通过 adapter
        vision_features = self.visual_adapter(vision_agg)  # [B, 8, 512]

        return vision_features

    def get_text_features(self, x_struct, time_marks=None, ts_keys=None):
        if self.disable_text:
            B, T, S, F = x_struct.shape
            return torch.zeros(B, S, self.text_feature_dim, device=x_struct.device)
        out = self.text_encoder(x_struct, time_marks, ts_keys=ts_keys)
        return out

    
    def get_vision_hit_rate(self):
        if self.vision_requests == 0:
            return 0.0
        return self.vision_hits / self.vision_requests
    
    def _compute_local_memory(self, patches):
        retrieved_patches, _ = self.patch_memory_bank.retrieve(
            patches, top_k=self.config.top_k
        )
        local_memory = self.local_memory_mlp(retrieved_patches)
        local_memory = local_memory.mean(dim=1, keepdim=True)
        local_memory = local_memory + patches
        return local_memory
    
    def _compute_global_memory(self, patches):
        attn_output, _ = self.memory_attention(
            query=patches, key=patches, value=patches
        )
        self.patch_memory_bank.update(patches.detach())
        
        if self.use_mem_gate:
            return attn_output
        else:
            return attn_output.mean(dim=1, keepdim=True)
    
    def _aggregate_station_tokens(self, tokens, attn_layers, norm_layers, query_param):
        B, S, F, D = tokens.shape
        seq = tokens.view(B * S, F, D)
        query = query_param.expand(B * S, -1, -1)
        for attn, norm in zip(attn_layers, norm_layers):
            attn_out, _ = attn(query, seq, seq)
            query = norm(attn_out + query)
        return query.squeeze(1).view(B, S, D)
    
    def _temporal_forward(self, x_enc):
        """纯时序前向"""
        B, L, n_vars = x_enc.shape

        patches, _ = self.patch_embedding(x_enc.transpose(1, 2).contiguous())
        
        local_memory = self._compute_local_memory(patches)
        global_memory = self._compute_global_memory(patches)
        
        if self.use_mem_gate:
            combined_features = torch.cat([local_memory, global_memory], dim=-1)
            gate_weights = self.memory_fusion_gate(combined_features)
            memory_features = (
                gate_weights[:, :, 0:1] * local_memory +
                gate_weights[:, :, 1:2] * global_memory
            )
        else:
            memory_features = local_memory + global_memory
        
        memory_features = self.flatten(memory_features)
        temporal_features = self.temporal_head(memory_features)
        memory_predictions = self.memory_head(memory_features)
        
        temporal_features = einops.rearrange(
            temporal_features, '(b n) d -> b n d', b=B, n=n_vars
        )
        memory_predictions = einops.rearrange(
            memory_predictions, '(b n) d -> b n d', b=B, n=n_vars
        )
        
        if n_vars > self.num_stations:
            features_per_station = max(1, n_vars // self.num_stations)
            
            temporal_tokens = temporal_features.view(
                B, self.num_stations, features_per_station, -1
            )
            memory_tokens = memory_predictions.view(
                B, self.num_stations, features_per_station, -1
            )
            
            temporal_features = self._aggregate_station_tokens(
                temporal_tokens, self.temporal_feature_attn,
                self.temporal_feature_norms, self.temporal_feature_query
            )
            
            memory_tokens = self.memory_pred_in(memory_tokens)
            memory_tokens = self._aggregate_station_tokens(
                memory_tokens, self.memory_feature_attn,
                self.memory_feature_norms, self.memory_feature_query
            )
            memory_predictions = self.memory_pred_out(memory_tokens)
        
        return temporal_features, memory_predictions
    
    def forward_prediction(self, x_enc, vision_features, text_features, x_struct=None):
        """
        【修复版】前向预测
        """
        B = x_enc.shape[0]
        dev = x_enc.device

        # 1. 时序骨干前向
        temporal_features, memory_predictions = self._temporal_forward(x_enc)

        # ===== 情况一：完全关闭多模态 =====
        if self.disable_visual and self.disable_text:
            final_pred = memory_predictions.permute(0, 2, 1)
            dummy_mm_pred = torch.zeros_like(final_pred)
            return final_pred, final_pred, dummy_mm_pred

        # ===== 情况二：启用多模态增强 =====
        
        # 2. 时序 + 空间图
        if self.disable_gnn:
            # 保留 layer_norm，避免仅因“少一个 LN”导致尺度变化
            h_spatial = self.layer_norm(temporal_features)
        else:
            h_spatial = self.layer_norm(self.graph_learner(temporal_features))

        # 3. 投影视觉 & 文本特征
        if self.disable_visual:
            v_proj = torch.zeros(B, self.num_stations, self.config.d_model, device=dev)
        else:
            v_proj = self.visual_proj(vision_features)   # [B,S,d_model]

        if self.disable_text:
            t_proj = torch.zeros(B, self.num_stations, self.config.d_model, device=dev)
        else:
            t_proj = self.text_proj(text_features)       # [B,S,d_model]


        # 模态 dropout（训练时）
        if self.training and self.modal_dropout_rate > 0:
            if (not self.disable_visual) and torch.rand(1).item() < self.modal_dropout_rate:
                v_proj = v_proj * 0
            if (not self.disable_text) and torch.rand(1).item() < self.modal_dropout_rate:
                t_proj = t_proj * 0

        # 4. 多模态特征融合
        mm_input = torch.cat([h_spatial, v_proj, t_proj], dim=-1)  # [B, S, 3*d_model]
        mm_features = self.multimodal_fusion(mm_input)             # [B, S, d_model]
        
        # 5. 跨站点注意力
        if self.disable_csa:
            fused_features = mm_features
        else:
            fused_features = self.cross_station_attn(h_spatial, mm_features)

        # 6. 模态门控
        mm_weight = self.modality_gate(h_spatial, fused_features)  # [B, S]
        mm_weight = mm_weight.unsqueeze(-1)  # [B, S, 1]

        # 7. 使用 multimodal_scale 控制多模态贡献上限
        # effective_scale 在 [0.1, 0.9] 范围内
        effective_scale = torch.sigmoid(self.multimodal_scale) * 0.8 + 0.1
        mm_weight = mm_weight * effective_scale  # 缩放门控权重

        # 8. 多模态预测
        mm_pred = self.multimodal_head(fused_features)  # [B, S, pred_len]
        
        # 9. 融合预测
        final_predictions = memory_predictions + mm_weight * (mm_pred - memory_predictions)

        # 10. 形状对齐 & 非负约束
        mm_pred_seq = mm_pred.permute(0, 2, 1)

        if self.nonnegative:
            final_predictions = F.softplus(final_predictions)
            memory_predictions = F.softplus(memory_predictions)
            mm_pred_seq = F.softplus(mm_pred_seq)

        final_pred = final_predictions.permute(0, 2, 1)
        memory_pred = memory_predictions.permute(0, 2, 1)

        return final_pred, memory_pred, mm_pred_seq

    def _normalize_input(self, x):
        if x.dim() == 4:
            means = x.mean(1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(
                torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            stdev = stdev / self.config.norm_const
            x_norm = x_norm / stdev
            return x_norm, means, stdev

        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(
            torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        stdev = stdev / self.config.norm_const
        x_norm = x_norm / stdev
        return x_norm, means, stdev
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, ts_keys=None):
        """主前向传播"""
        dev = self._module_device(self)
        x_enc = x_enc.to(dev).contiguous()
        
        if x_enc.dim() == 4:
            B, L, S, F = x_enc.shape
            x_struct = x_enc
        else:
            B, L, D = x_enc.shape
            S = self.num_stations
            F = max(1, D // S)
            x_struct = x_enc.view(B, L, S, F)
        
        x_norm_struct, _, _ = self._normalize_input(x_struct)
        x_norm = x_norm_struct.view(B, L, -1)
        
        if ts_keys is None:
            ts_keys = self._ts_from_marks(x_mark_enc, x_mark_dec)

        if self.disable_visual:
            vision_features = torch.zeros(B, self.num_stations, self.image_feature_dim, device=dev)
        else:
            vision_features = self.get_vision_features(ts_keys)

        if self.disable_text:
            text_features = torch.zeros(B, self.num_stations, self.text_feature_dim, device=dev)
        else:
            text_features = self.get_text_features(x_struct, x_mark_enc, ts_keys=ts_keys)
        
        predictions, memory_predictions, multimodal_predictions = self.forward_prediction(
            x_norm, vision_features, text_features, x_struct
        )
        
        return predictions, memory_predictions, multimodal_predictions
