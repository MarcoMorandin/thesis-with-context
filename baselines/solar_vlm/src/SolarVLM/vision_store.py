# src/SolarVLM/vision_store.py
import os
from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np
import torch


class VisionFeatureStore:
    """
    按 ts_key 读取预计算的特征：
    - 支持单帧: [S, D]
    - 支持多帧: [S, T, D]
    - 支持 get_sequence(): 返回 [S, n_frames, D]
    """
    def __init__(self, feat_dir: str, cache_size: int = 8192, dtype=np.float32, feat_dim: int = None):
        self.feat_dir = feat_dir
        self.cache_size = cache_size
        self.dtype = dtype
        self.feat_dim = feat_dim  # 新增：期望的最后一维 D（比如 2048）
        self._cache = OrderedDict()

        self._available_keys = {}
        self._scan_available_keys()

    def _scan_available_keys(self):
        if not os.path.exists(self.feat_dir):
            return
        for root, _, files in os.walk(self.feat_dir):
            for fname in files:
                if fname.endswith('.npy'):
                    ts_key = fname[:-4]
                    self._available_keys[ts_key] = os.path.join(root, fname)

    def _path(self, ts_key: str) -> str:
        return self._available_keys.get(ts_key, os.path.join(self.feat_dir, f"{ts_key}.npy"))

    def exists(self, ts_key: str) -> bool:
        return ts_key in self._available_keys

    def _match_dim_np(self, arr: np.ndarray) -> np.ndarray:
        """把 arr 的最后一维对齐 to self.feat_dim（pad 或 truncate）"""
        if self.feat_dim is None:
            return arr
        if arr.shape[-1] == self.feat_dim:
            return arr

        D = arr.shape[-1]
        if D > self.feat_dim:
            return arr[..., :self.feat_dim]

        # D < feat_dim: pad zeros
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, self.feat_dim - D)
        return np.pad(arr, pad_width, mode='constant', constant_values=0)

    def get(self, ts_key: str) -> torch.Tensor:
        """
        获取单个 ts_key 对应文件内容：
        - 可能是 [S, D] 或 [S, T, D]
        """
        if ts_key in self._cache:
            arr = self._cache.pop(ts_key)
            self._cache[ts_key] = arr
        else:
            path = self._path(ts_key)
            if not os.path.exists(path):
                raise FileNotFoundError(f"vision feature not found: {path}")
            arr = np.load(path).astype(self.dtype)
            arr = self._match_dim_np(arr)  
            self._cache[ts_key] = arr

            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return torch.from_numpy(self._cache[ts_key].copy())

    def get_sequence(self, ts_key: str, n_frames: int = 8,
                     freq_minutes: int = 60, tolerance_minutes: int = 5,
                     num_stations: int = 1) -> torch.Tensor:
        """
        获取以 ts_key 为终点的连续 n_frames 帧特征序列
        返回: [S, n_frames, D]
        """
        fmt = '%Y%m%d%H%M'
        S = num_stations
        D = self.feat_dim if self.feat_dim is not None else 2048

        try:
            target_dt = datetime.strptime(ts_key, fmt)
        except ValueError:
            return torch.zeros(S, n_frames, D)

        frames = []
        last_valid_feat = None

        for i in range(n_frames - 1, -1, -1):
            frame_dt = target_dt - timedelta(minutes=i * freq_minutes)
            frame_key = frame_dt.strftime(fmt)

            feat = self._find_feature_with_tolerance(frame_key, tolerance_minutes)

            if feat is not None:
                last_valid_feat = feat
                frames.append(feat)
            elif last_valid_feat is not None:
                frames.append(last_valid_feat.clone())
            else:
                frames.append(torch.zeros(S, D))

        #  现在 frames 里每个都是 [S, D]，一定能 stack
        stacked = torch.stack(frames, dim=0)      # [n_frames, S, D]
        return stacked.permute(1, 0, 2)           # [S, n_frames, D]

    def _find_feature_with_tolerance(self, ts_key: str, tolerance_minutes: int):
        fmt = '%Y%m%d%H%M'

        # 1) 精确匹配
        if self.exists(ts_key):
            try:
                feat = self.get(ts_key)
                if feat.dim() == 3:
                    feat = feat[:, -1, :]
                if self.feat_dim is not None and feat.shape[-1] != self.feat_dim:
                    D = feat.shape[-1]
                    if D > self.feat_dim:
                        feat = feat[..., :self.feat_dim]
                    else:
                        pad = torch.zeros(feat.shape[0], self.feat_dim - D, dtype=feat.dtype)
                        feat = torch.cat([feat, pad], dim=-1)
                return feat
            except Exception:
                pass

        # 2) 容差搜索
        try:
            target_dt = datetime.strptime(ts_key, fmt)
        except ValueError:
            return None

        for delta in range(1, tolerance_minutes + 1):
            for sign in (-1, 1):
                cand_dt = target_dt + timedelta(minutes=sign * delta)
                cand_key = cand_dt.strftime(fmt)
                if self.exists(cand_key):
                    try:
                        feat = self.get(cand_key)
                        if feat.dim() == 3:
                            feat = feat[:, -1, :]
                        if self.feat_dim is not None and feat.shape[-1] != self.feat_dim:
                            D = feat.shape[-1]
                            if D > self.feat_dim:
                                feat = feat[..., :self.feat_dim]
                            else:
                                pad = torch.zeros(feat.shape[0], self.feat_dim - D, dtype=feat.dtype)
                                feat = torch.cat([feat, pad], dim=-1)
                        return feat
                    except Exception:
                        continue

        return None
