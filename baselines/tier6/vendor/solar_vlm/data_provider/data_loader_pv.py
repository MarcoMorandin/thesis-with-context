# File: data_provider/data_loader_pv.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_PV(Dataset):
    """
    Custom dataset for PV power forecasting with multi-station data
    """
    def __init__(self, root_path, flag='train', size=None, 
                 features='MS', 
                 target='power', scale=True, timeenc=0, freq='t',start_time='2018-12-01 00:00', end_time='2019-06-01 00:00'):
        
        # Size parameters
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Dataset configuration
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        # Time range parameters (start_time and end_time)
        self.start_time = start_time
        self.end_time = end_time

        # Station configuration
        self.station_list = ['station00', 'station01', 'station02', 'station04',
                            'station06', 'station07', 'station08', 'station09']
        self.num_stations = len(self.station_list)
        
        # Data paths
        self.root_path = root_path
        
        
        # Dataset split ratios
        self.flag = flag
        
        self.__read_data__()

    def __read_data__(self):
        """Read and preprocess PV data from multiple stations; robust time parsing, 15-min alignment, train-only scaling."""
        # ---- 1) 读取每站数据，解析时间列 -> 对齐到15分钟 -> 去重聚合 ----
        station_data_list = []
        step_freq = '15T'  # 15分钟粒度

        for station in self.station_list:
            station_file = os.path.join(self.root_path, f'{station}.csv')
            if not os.path.exists(station_file):
                print(f"Warning: Station file {station_file} not found")
                station_data_list.append(pd.DataFrame())  # 占位
                continue

            df_raw = pd.read_csv(station_file)
            df_raw.columns = [str(c).strip() for c in df_raw.columns]

            # 时间列优先用 date_time，没有则回退到 date，再不行用首列
            if 'date_time' in df_raw.columns:
                tcol = 'date_time'
            elif 'date' in df_raw.columns:
                tcol = 'date'
            else:
                tcol = df_raw.columns[0]

            dt = pd.to_datetime(df_raw[tcol], errors='coerce', infer_datetime_format=True)
            # 统一去 tz，避免 tz/naive 混用
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                pass

            # 对齐到 15 分钟
            dt = dt.dt.floor(step_freq)
            df_raw = df_raw.drop(columns=[tcol])
            df_raw.index = dt
            df_raw = df_raw.sort_index()
            from datetime import datetime
            start_time = datetime.strptime(self.start_time, '%Y-%m-%d %H:%M')
            end_time = datetime.strptime(self.end_time, '%Y-%m-%d %H:%M')
            # 时间范围筛选：只保留 start_time 和 end_time 范围内的数据
            df_raw = df_raw[(df_raw.index >= start_time) & (df_raw.index < end_time)]

            # 若同一 15 分钟格存在多条记录：数值取均值，非数值取首个
            if df_raw.index.has_duplicates:
                num_cols = df_raw.select_dtypes(include='number').columns.tolist()
                agg_map = {c: 'mean' for c in num_cols}
                for c in df_raw.columns:
                    if c not in num_cols:
                        agg_map[c] = 'first'
                df_raw = df_raw.groupby(level=0).agg(agg_map).sort_index()

            # 选特征
            if self.features in ('M', 'MS'):
                feature_cols = [
                    'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
                    'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection',
                    'nwp_pressure', 'lmd_totalirrad', 'lmd_diffuseirrad',
                    'lmd_temperature', 'lmd_pressure', 'lmd_winddirection',
                    'lmd_windspeed', 'power'
                ]
                cols_to_use = [c for c in feature_cols if c in df_raw.columns]
                if not cols_to_use:
                    raise ValueError(f"{station} 没有可用特征列，请检查数据。")
                df_raw = df_raw[cols_to_use]
            elif self.features == 'S':
                if self.target not in df_raw.columns:
                    raise ValueError(f"{station} 缺少目标列 {self.target}")
                df_raw = df_raw[[self.target]]
            else:
                raise ValueError(f"Unsupported features mode: {self.features}")

            # 先粗填一下（完整插值在对齐 common_index 后再做）
            df_raw = df_raw.ffill().bfill()

            station_data_list.append(df_raw)

        # ---- 2) 计算公共时间索引（交集）并对齐+插值 ----
        datetime_indices = [df.index for df in station_data_list if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0]
        if not datetime_indices:
            raise ValueError("没有有效的时间索引，请检查原始数据。")

        common_index = datetime_indices[0]
        for idx in datetime_indices[1:]:
            common_index = common_index.intersection(idx)
        if len(common_index) == 0:
            raise ValueError("各站点时间索引交集为空，请检查数据时间覆盖范围是否一致。")

        aligned_data = []
        for df in station_data_list:
            if not isinstance(df.index, pd.DatetimeIndex) or len(df) == 0:
                tmp = pd.DataFrame(index=common_index)  # 空站点占位
            else:
                # 再保险去重
                if df.index.has_duplicates:
                    num_cols = df.select_dtypes(include='number').columns.tolist()
                    agg_map = {c: 'mean' for c in num_cols}
                    for c in df.columns:
                        if c not in num_cols:
                            agg_map[c] = 'first'
                    df = df.groupby(level=0).agg(agg_map).sort_index()
                tmp = df.reindex(common_index)

            num_cols = tmp.select_dtypes(include='number').columns.tolist()
            if len(num_cols) > 0:
                tmp[num_cols] = tmp[num_cols].interpolate(method='time', limit_direction='both')
                tmp[num_cols] = tmp[num_cols].ffill().bfill()
            else:
                tmp = tmp.ffill().bfill()

            aligned_data.append(tmp)

        # ---- 3) 组装原始数据（暂不缩放）----
        raw_x_list, raw_y_list = [], []
        for df in aligned_data:
            data = df.values
            raw_x_list.append(data)
            if self.features == 'MS':
                raw_y_list.append(data[:, -1:])  # power 为最后一列
            else:
                raw_y_list.append(data)

        raw_x = np.stack(raw_x_list, axis=1)  # (T, num_stations, num_features)
        raw_y = np.stack(raw_y_list, axis=1)

        # ---- 4) 依据总长度计算切分边界 ----
        num_samples = raw_x.shape[0]
        num_train = int(num_samples * 0.8)
        num_val = int(num_samples * 0.1)
        num_test = num_samples - num_train - num_val

        border1s = [0, num_train - self.seq_len, num_train + num_val - self.seq_len]
        border2s = [num_train, num_train + num_val, num_samples]
        border1 = border1s[['train', 'val', 'test'].index(self.flag)]
        border2 = border2s[['train', 'val', 'test'].index(self.flag)]

        # ---- 5) 逐站点归一化 ----
        self.scaler_x = []
        self.scaler_y = []

        if self.scale:
            scaled_x = raw_x.copy()
            scaled_y = raw_y.copy()
            for station_idx in range(self.num_stations):
                scaler_x = StandardScaler()
                scaler_x.fit(raw_x[:num_train, station_idx, :])
                scaled_x[:, station_idx, :] = scaler_x.transform(raw_x[:, station_idx, :])
                self.scaler_x.append(scaler_x)

                scaler_y = StandardScaler()
                scaler_y.fit(raw_y[:num_train, station_idx, :])
                scaled_y[:, station_idx, :] = scaler_y.transform(raw_y[:, station_idx, :])
                self.scaler_y.append(scaler_y)
        else:
            scaled_x = raw_x
            scaled_y = raw_y

        self.scaler = self.scaler_y

        # ---- 6) 生成时间特征 ----
        df_stamp = pd.DataFrame({'date': common_index})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")

        # ---- 7) 最后按 split 切片 ----
        self.data_x = scaled_x[border1:border2]
        self.data_y = scaled_y[border1:border2]
        self.data_stamp = data_stamp[border1:border2]


    
    def __getitem__(self, index):
        """Get a single sample"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Input sequence
        seq_x = self.data_x[s_begin:s_end]  # [seq_len, num_stations, features]
        seq_y = self.data_y[r_begin:r_end]  # [label_len+pred_len, num_stations, 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.squeeze(-1).astype(np.float32)
        seq_x_mark = seq_x_mark.astype(np.float32)
        seq_y_mark = seq_y_mark.astype(np.float32)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """Inverse transform for y (power)."""
        if not hasattr(self, 'scaler_y') or not self.scaler_y:
            return data
        import torch
        was_tensor = isinstance(data, torch.Tensor)
        if was_tensor:
            device = data.device
            np_data = data.detach().cpu().numpy()
        else:
            np_data = np.array(data)
        orig_shape = np_data.shape
        if orig_shape[-1] != self.num_stations:
            raise ValueError(f"Expected last dim == num_stations ({self.num_stations}), got {orig_shape[-1]}")
        flat = np_data.reshape(-1, self.num_stations)
        restored = np.zeros_like(flat)
        for idx, scaler in enumerate(self.scaler_y):
            restored[:, idx] = scaler.inverse_transform(flat[:, idx:idx+1]).ravel()
        restored = restored.reshape(orig_shape)
        if was_tensor:
            return torch.from_numpy(restored).to(device)
        return restored
