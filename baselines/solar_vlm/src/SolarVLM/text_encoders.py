import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LRUCache:
    """CPU LRU cache: prompt(str) -> embedding(torch.float16 on cpu)"""
    def __init__(self, max_items=20000):
        self.max_items = int(max_items)
        self.od = OrderedDict()

    def get(self, key):
        if key not in self.od:
            return None
        v = self.od.pop(key)
        self.od[key] = v
        return v

    def put(self, key, value):
        if self.max_items <= 0:
            return
        if key in self.od:
            self.od.pop(key)
        self.od[key] = value
        while len(self.od) > self.max_items:
            self.od.popitem(last=False)


class HybridTextEncoder(nn.Module):
    """

    - backend=clip 或 qwen3vl（你现在是 qwen3vl）
    - 输入 x_struct: [B, T, S, F]（通常是 StandardScaler 后的 z-score）
    - time_marks: [B, T, 5] (timeenc=0) -> month/day/weekday/hour/min_bucket
    - 输出: [B, S, out_dim]，qwen3vl 建议 out_dim=2048
    """
    def __init__(self,
                 backend: str = "qwen3vl",
                 clip_model=None,
                 clip_processor=None,
                 qwen_embedder=None,
                 num_stations: int = 8,
                 num_features: int = 14,
                 feature_names=None,   # actual ordered list of feature names
                 d_model: int = 256,
                 temporal_dim: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 out_dim: int = 2048,
                 # ===== PV prompt config =====
                 freq_minutes: int = 15,
                 pred_len: int = None,
                 ctx_short: int = 12,    # 3小时(15min*12)
                 ctx_long: int = 48,     # 12小时(15min*48)
                 # ===== speed =====
                 normalize_text_emb: bool = True,
                 cache_size: int = 20000,
                 # ===== station meta =====
                 station_list=None,
                 station_coords=None,    # dict: name -> (lon, lat)
                 year_hint: int = 2019,
                 ):
        super().__init__()
        self.backend = (backend or "qwen3vl").lower()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.qwen_embedder = qwen_embedder

        self.num_stations = int(num_stations)
        self.output_dim = int(out_dim)
        self.temporal_dim = int(temporal_dim)
        self.max_seq_len = int(max_seq_len)

        self.freq_minutes = int(freq_minutes)
        self.pred_len = pred_len
        self.ctx_short = int(ctx_short)
        self.ctx_long = int(ctx_long)
        self.normalize_text_emb = bool(normalize_text_emb)

        # 缓存（强烈推荐：Qwen3 embedding 计算贵）
        self.cache = _LRUCache(max_items=cache_size)

        _default_names = [
            'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
            'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection',
            'nwp_pressure', 'lmd_totalirrad', 'lmd_diffuseirrad',
            'lmd_temperature', 'lmd_pressure', 'lmd_winddirection',
            'lmd_windspeed', 'power',
        ]
        _names = feature_names if feature_names is not None else _default_names
        self.feature_to_idx = {name: idx for idx, name in enumerate(_names)}
        self.num_actual_features = len(_names)

        # station meta
        self.station_coords = station_coords if station_coords is not None else {}

        # 关键：station_list 的顺序必须与数据 x_struct 的站点维度 S 顺序一致
        if station_list is None:
            if len(self.station_coords) > 0:
                # 默认按 key 排序，保证 deterministic
                self.station_list = sorted(list(self.station_coords.keys()))
            else:
                self.station_list = [f"station{i:02d}" for i in range(self.num_stations)]
        else:
            self.station_list = list(station_list)

        # 如果 station_list 长度与 num_stations 不一致，以 station_list 为准（更安全）
        self.num_stations = len(self.station_list)

        self.year_hint = int(year_hint)

        # ===== 站点经纬度“可辩护”粗粒度分箱（避免精确经纬度变相站点ID）=====
        # 这批站点：lat 36.64~39.52, lon 113.64~117.46
        # - 纬度粗分两档：36-38 / 38-40（solar geometry 与气候带粗先验）
        # - 经度粗分两档：113-115 / 115-118（local solar time 粗修正）
        self.site_lat_band = []
        self.site_lon_band = []
        for name in self.station_list:
            lon_lat = self.station_coords.get(name, None)
            if lon_lat is None:
                # 缺失则不在 prompt 里输出（避免 unk）
                self.site_lat_band.append(None)
                self.site_lon_band.append(None)
                continue
            lon, lat = float(lon_lat[0]), float(lon_lat[1])

            lat_band = "latitude_36_to_38" if (lat < 38.0) else "latitude_38_to_40"
            lon_band = "longitude_113_to_115" if (lon < 115.0) else "longitude_115_to_118"

            self.site_lat_band.append(lat_band)
            self.site_lon_band.append(lon_band)

        # ===== Temporal branch（保留你原结构）=====
        self.temporal_proj = nn.Linear(num_features, temporal_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=4,
            dim_feedforward=temporal_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.temporal_out = nn.Sequential(
            nn.Linear(temporal_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, temporal_dim) * 0.02)

        # ===== Text branch adapter =====
        self.text_adapter = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ===== Explicit fusion (NO cross-attn) =====
        self.t_norm = nn.LayerNorm(out_dim)
        self.c_norm = nn.LayerNorm(out_dim)

        self.c_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

        self.output = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.station_emb = nn.Parameter(torch.randn(self.num_stations, out_dim) * 0.02)

        # backend checks
        if self.backend == "clip":
            if self.clip_model is None or self.clip_processor is None:
                raise ValueError("backend=clip 但 clip_model/clip_processor 为空")
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_model.eval()

        elif self.backend in ("qwen3vl", "qwen3_vl", "qwen3-vl"):
            if self.qwen_embedder is None:
                raise ValueError("backend=qwen3vl 但 qwen_embedder 为空")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # -------------------- helpers --------------------
    def _safe(self, x, default=0.0):
        try:
            if torch.is_tensor(x):
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).item()
            x = float(x)
            if math.isnan(x) or math.isinf(x):
                return float(default)
            return x
        except Exception:
            return float(default)

    def _zbin(self, z):
        """z-score 分箱（稳定离散 token，适配 StandardScaler 后的现实）"""
        z = self._safe(z)
        if z < -1.5: return "very_low"
        if z < -0.5: return "low"
        if z <  0.5: return "medium"
        if z <  1.5: return "high"
        return "very_high"

    def _trend_bin(self, dz):
        dz = self._safe(dz)
        if dz >  1.0: return "strong_increase"
        if dz >  0.3: return "increase"
        if dz < -1.0: return "strong_decrease"
        if dz < -0.3: return "decrease"
        return "stable"

    def _var_bin(self, std):
        std = abs(self._safe(std))
        if std < 0.25: return "very_stable"
        if std < 0.50: return "stable"
        if std < 1.00: return "variable"
        return "highly_variable"

    def _corr_bin(self, c):
        c = self._safe(c)
        if c > 0.70: return "strong_positive"
        if c > 0.30: return "positive"
        if c < -0.70: return "strong_negative"
        if c < -0.30: return "negative"
        return "weak"

    def _solar_elev_bin(self, lat_deg, month, day, hour, minute):
        """
        粗略太阳高度角分箱：只要能把“夜/低/中/高”区分出来即可
        """
        try:
            lat = float(lat_deg)
            m = int(month); d = int(day)
            h = int(hour); mi = int(minute)

            import datetime as _dt
            doy = _dt.date(self.year_hint, m, d).timetuple().tm_yday

            # declination (deg)
            dec = 23.44 * math.sin(2 * math.pi * (284 + doy) / 365.0)

            # hour angle (deg), local solar time 简化用 clock time
            t = h + mi / 60.0
            ha = 15.0 * (t - 12.0)

            lat_r = math.radians(lat)
            dec_r = math.radians(dec)
            ha_r  = math.radians(ha)

            sin_e = math.sin(lat_r)*math.sin(dec_r) + math.cos(lat_r)*math.cos(dec_r)*math.cos(ha_r)
            sin_e = max(-1.0, min(1.0, sin_e))
            elev = math.degrees(math.asin(sin_e))

            if elev < 0:  return "night"
            if elev < 15: return "low"
            if elev < 35: return "medium"
            return "high"
        except Exception:
            # 不输出 unknown，直接让上层决定是否输出此字段
            return None

    def _get_mark(self, time_marks, b, t_idx=-1):
        """
        time_marks: [B,T,5] (timeenc=0) -> month/day/weekday/hour/min_bucket
        """
        if time_marks is None:
            return None
        try:
            mk = time_marks[b, t_idx, :].detach().cpu().tolist()
            month = int(mk[0]); day = int(mk[1])
            hour = int(mk[3]); min_bucket = int(mk[4])
            minute = int(min_bucket) * 15
            return month, day, hour, minute
        except Exception:
            return None

    def _corrcoef(self, a, b):
        """Pearson corr (float). a,b: 1D torch."""
        a = a.float()
        b = b.float()
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.pow(2).mean().sqrt() * b.pow(2).mean().sqrt() + 1e-6)
        return self._safe((a*b).mean() / denom)

    def _delay_select(self, irr, pwr):
        """
        判断 delay_0_step vs delay_1_step 哪个相关更强（一步=15min）
        用更“可解释”的名字替代 LagSelection / lag0/lag1
        """
        if irr.numel() < 4:
            return "delay_0_step"
        c0 = self._corrcoef(irr, pwr)
        c1 = self._corrcoef(irr[:-1], pwr[1:])
        return "delay_1_step" if (c1 > c0 + 0.05) else "delay_0_step"

    # -------------------- prompt core --------------------
    def _build_prompt(self, x_struct, b, s, time_marks=None, ts_keys=None):
        """
        更可辩护版 prompt：
        - 不包含 station00/stationxx
        - 不包含精确经纬度，只包含 coarse band（且不输出 unknown/NA）
        - 字段命名更直观
        """
        idx = self.feature_to_idx
        xs = x_struct[b, :, s, :].detach()

        T = xs.size(0)
        k1 = min(self.ctx_short, T)
        k2 = min(self.ctx_long, T)

        w1 = xs[-k1:, :]
        w2 = xs[-k2:, :]

        def _col(w, key, aliases=()):
            """Return column by key or alias; zeros if absent or out of bounds."""
            F = w.size(1)
            for k in (key,) + tuple(aliases):
                if k in idx and idx[k] < F:
                    return w[:, idx[k]]
            return torch.zeros(w.size(0), device=w.device, dtype=w.dtype)

        def _scalar(w, key, aliases=()):
            return _col(w, key, aliases)[-1]

        # 主要信号（z-score）— 'pv' is SKIPPD alias for 'power'
        power_s = _col(w1, 'power', ('pv',))
        power_l = _col(w2, 'power', ('pv',))

        irr_s = _col(w1, 'lmd_totalirrad', ('GHI', 'ghi'))
        irr_l = _col(w2, 'lmd_totalirrad', ('GHI', 'ghi'))

        nwp_irr_s = _col(w1, 'nwp_globalirrad', ('GHI',))

        # last bins
        power_last_bin = self._zbin(power_s[-1])
        irr_last_bin   = self._zbin(irr_s[-1])

        temp_last_bin  = self._zbin(_scalar(w1, 'nwp_temperature', ('Tamb', 'temperature')))
        humid_last_bin = self._zbin(_scalar(w1, 'nwp_humidity', ('RH', 'humidity')))
        wind_last_bin  = self._zbin(_scalar(w1, 'nwp_windspeed', ('Wspd', 'wind_speed')))
        pres_last_bin  = self._zbin(_scalar(w1, 'nwp_pressure', ('Patm', 'pressure')))

        # trends (short / long)
        power_trend_short = self._trend_bin(power_s[-1] - power_s[0])
        irr_trend_short   = self._trend_bin(irr_s[-1] - irr_s[0])

        power_trend_long  = self._trend_bin(power_l[-1] - power_l[0])
        irr_trend_long    = self._trend_bin(irr_l[-1] - irr_l[0])

        # variability (short)
        power_variability_short = self._var_bin(power_s.std(unbiased=False))
        irr_variability_short   = self._var_bin(irr_s.std(unbiased=False))

        # coherence / delay
        coherence_short = self._corr_bin(self._corrcoef(irr_s, power_s))
        response_delay  = self._delay_select(irr_l, power_l)

        # forecast vs observation irradiance gap (z-space)
        gap_mean = abs(self._safe((nwp_irr_s - irr_s).mean()))
        if gap_mean < 0.5:
            irr_gap = "small"
        elif gap_mean < 1.0:
            irr_gap = "medium"
        elif gap_mean < 2.0:
            irr_gap = "large"
        else:
            irr_gap = "very_large"

        # time / solar
        got_time = False
        if ts_keys is not None and b < len(ts_keys):
            try:
                ts_str = ts_keys[b]
                month = int(ts_str[4:6])
                day = int(ts_str[6:8])
                hour = int(ts_str[8:10])
                minute = int(ts_str[10:12])
                got_time = True
            except (ValueError, TypeError, IndexError):
                pass

        if not got_time:
            mk = self._get_mark(time_marks, b, t_idx=-1)
            if mk is not None:
                month, day, hour, minute = mk
                got_time = True

        season = None
        time_of_day = None
        solar_elevation = None

        if got_time:
            if month in (12, 1, 2):   season = "winter"
            elif month in (3, 4, 5):  season = "spring"
            elif month in (6, 7, 8):  season = "summer"
            else:                     season = "autumn"

            if 5 <= hour < 10:        time_of_day = "morning"
            elif 10 <= hour < 15:     time_of_day = "noon"
            elif 15 <= hour < 19:     time_of_day = "afternoon"
            elif 19 <= hour < 23:     time_of_day = "night"
            else:                     time_of_day = "late_night"

            # solar elevation needs lat
            lon_lat = self.station_coords.get(self.station_list[s], None)
            if lon_lat is not None:
                lat = float(lon_lat[1])
                solar_elevation = self._solar_elev_bin(lat, month, day, hour, minute)

        # defendable site bands (only if available)
        lat_band = self.site_lat_band[s] if (s < len(self.site_lat_band)) else None
        lon_band = self.site_lon_band[s] if (s < len(self.site_lon_band)) else None

        # “cloud / curtailment” indicator：仅在逻辑触发时输出；否则不输出该字段（避免 NA）
        cloud_indicator = None
        curtailment_indicator = None
        if solar_elevation in ("medium", "high"):
            if irr_last_bin in ("very_low", "low"):
                cloud_indicator = "likely_cloudy"
            if irr_last_bin in ("high", "very_high") and power_last_bin in ("very_low", "low"):
                curtailment_indicator = "likely_curtailment_or_clipping"

        # 任务描述（更清晰：多步长序列、步间隔、每站点）
        parts = []

        # ---- task / horizon ----
        parts.append("task=pv_power_forecasting")
        parts.append("target=power")
        parts.append("output=per_station_multi_step_sequence")
        if self.pred_len is not None:
            parts.append(f"forecast_steps={int(self.pred_len)}")
        parts.append(f"step_minutes={int(self.freq_minutes)}")

        # ---- context length ----
        parts.append(f"context_short_steps={int(k1)}")
        parts.append(f"context_long_steps={int(k2)}")

        # ---- site coarse priors (no station id, no exact coords) ----
        if lat_band is not None:
            parts.append(f"site_latitude_band={lat_band}")
        if lon_band is not None:
            parts.append(f"site_longitude_band={lon_band}")

        # ---- time priors ----
        if season is not None:
            parts.append(f"season={season}")
        if time_of_day is not None:
            parts.append(f"time_of_day={time_of_day}")
        if solar_elevation is not None:
            parts.append(f"solar_elevation={solar_elevation}")

        # ---- power summary ----
        parts.append(f"power_last_bin={power_last_bin}")
        parts.append(f"power_trend_short={power_trend_short}")
        parts.append(f"power_trend_long={power_trend_long}")
        parts.append(f"power_variability_short={power_variability_short}")

        # ---- irradiance summary ----
        parts.append(f"irradiance_last_bin={irr_last_bin}")
        parts.append(f"irradiance_trend_short={irr_trend_short}")
        parts.append(f"irradiance_trend_long={irr_trend_long}")
        parts.append(f"irradiance_variability_short={irr_variability_short}")

        # ---- coupling / uncertainty ----
        parts.append(f"irradiance_power_coherence_short={coherence_short}")
        parts.append(f"irradiance_to_power_response_delay={response_delay}")
        parts.append(f"forecast_observation_irradiance_gap={irr_gap}")

        # ---- meteo bins ----
        parts.append(f"temperature_last_bin={temp_last_bin}")
        parts.append(f"humidity_last_bin={humid_last_bin}")
        parts.append(f"wind_speed_last_bin={wind_last_bin}")
        parts.append(f"pressure_last_bin={pres_last_bin}")

        # ---- conditional indicators (no NA) ----
        if cloud_indicator is not None:
            parts.append(f"cloud_indicator={cloud_indicator}")
        if curtailment_indicator is not None:
            parts.append(f"curtailment_indicator={curtailment_indicator}")

        return " | ".join(parts)

    @torch.no_grad()
    def _encode_text(self, prompts, device):
        """
        返回 [N, out_dim] float32 on device
        带缓存：prompt -> cpu fp16 embedding
        """
        embs = [None] * len(prompts)
        miss_prompts = []
        miss_ids = []
        for i, p in enumerate(prompts):
            v = self.cache.get(p)
            if v is None:
                miss_prompts.append(p)
                miss_ids.append(i)
            else:
                embs[i] = v  # cpu fp16

        if len(miss_prompts) > 0:
            if self.backend == "clip":
                clip_device = next(self.clip_model.parameters()).device
                inputs = self.clip_processor(
                    text=miss_prompts, return_tensors="pt",
                    padding=True, truncation=True, max_length=77
                )
                inputs = {k: v.to(clip_device) for k, v in inputs.items()}
                feat = self.clip_model.get_text_features(**inputs).to(dtype=torch.float32)
            else:
                items = [{"text": t} for t in miss_prompts]
                try:
                    feat = self.qwen_embedder.process(items, normalize=self.normalize_text_emb)
                except TypeError:
                    feat = self.qwen_embedder.process(items)
                if not torch.is_tensor(feat):
                    feat = torch.tensor(feat)
                feat = feat.to(dtype=torch.float32)

            # 维度对齐
            if feat.size(-1) != self.output_dim:
                if feat.size(-1) > self.output_dim:
                    feat = feat[:, :self.output_dim]
                else:
                    pad = torch.zeros(feat.size(0), self.output_dim - feat.size(-1), dtype=feat.dtype)
                    feat = torch.cat([feat, pad], dim=-1)

            feat_cpu = feat.detach().to("cpu", dtype=torch.float16)
            for j, idx in enumerate(miss_ids):
                self.cache.put(miss_prompts[j], feat_cpu[j])
                embs[idx] = feat_cpu[j]

        emb = torch.stack([e for e in embs], dim=0).to(device=device, dtype=torch.float32)
        return emb

    def forward(self, x_struct, time_marks=None, ts_keys=None):
        """
        x_struct: [B, T, S, F]
        return:  [B, S, out_dim]
        """
        B, T, S, F = x_struct.shape
        device = x_struct.device

        # 1) Temporal branch
        st = x_struct.permute(0, 2, 1, 3).contiguous().view(B * S, T, F)
        t_in = self.temporal_proj(st)

        if T > self.max_seq_len:
            t_in = t_in[:, -self.max_seq_len:, :]
            T_eff = self.max_seq_len
        else:
            T_eff = T

        pos = self.pos_emb[:, :T_eff, :]
        t_in = t_in[:, -T_eff:, :] + pos

        t_enc = self.temporal_encoder(t_in)
        t_feat_raw = 0.6 * t_enc[:, -1, :] + 0.4 * t_enc.mean(dim=1)
        t_feat = self.temporal_out(t_feat_raw)  # [B*S, out_dim]

        # 2) PV prompt -> text embedding
        prompts = [self._build_prompt(x_struct, b, s_idx, time_marks=time_marks, ts_keys=ts_keys)
                   for b in range(B) for s_idx in range(S)]
        c_feat = self._encode_text(prompts, device=device)  # [B*S, out_dim]
        c_feat = self.text_adapter(c_feat)

        # 3) Gated fusion
        t = self.t_norm(t_feat)
        c = self.c_norm(c_feat)
        c2 = self.c_proj(c)
        g = self.gate(torch.cat([t, c], dim=-1))
        fused = t + g * (c2 - t)

        out = self.output(fused).view(B, S, self.output_dim)
        out = out + self.station_emb.unsqueeze(0)
        return out
