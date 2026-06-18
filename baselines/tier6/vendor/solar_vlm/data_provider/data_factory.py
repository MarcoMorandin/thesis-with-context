from torch.utils.data import DataLoader
import torch
import inspect

from data_provider.data_loader_skippd     import Dataset_SKIPPD
from data_provider.data_loader_wollongong import Dataset_WOLLONGONG
from data_provider.data_loader_ukpv       import Dataset_UKPV

data_dict = {
    'SKIPPD':     Dataset_SKIPPD,
    'WOLLONGONG': Dataset_WOLLONGONG,
    'UKPV':       Dataset_UKPV,
}

def _make_dataset(Data, args, flag):
    timeenc = 0 if getattr(args, 'embed', 'timeF') != 'timeF' else 1
    freq    = getattr(args, 'freq', 't')

    candidates = {
        'args': args,
        'configs': args,
        'root_path': args.root_path,
        'flag': flag,
        'size': [args.seq_len, args.label_len, args.pred_len],
        'features': args.features,
        'data_path': args.data_path,
        'target': args.target,
        'scale': getattr(args, 'scale', True),
        'timeenc': timeenc,
        'freq': freq,
        'seasonal_patterns': getattr(args, 'seasonal_patterns', None),
        'periodicity': getattr(args, 'periodicity', 96),
        'start_time': args.start_time,
        'end_time': args.end_time,
        'use_era5': getattr(args, 'use_era5', True),
        'station':  getattr(args, 'wollongong_station', 'loc1'),
        'num_stations': getattr(args, 'num_stations', 8),
        'dataset':  getattr(args, 'ukpv_dataset', 'uk_pv'),
    }

    sig = inspect.signature(Data.__init__)
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
    return Data(**kwargs)

def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    data_set = _make_dataset(Data, args, flag)
    if args.percent < 1. and flag == 'train':
        num_samples = int(len(data_set) * args.percent)
        indices = torch.randperm(len(data_set))[:num_samples]
        data_set = torch.utils.data.Subset(data_set, indices)
        print(f"Few-shot sampling: {args.percent*100}% of data, {len(data_set)} samples")
    n_windows = len(data_set)
    print(flag, n_windows)
    if n_windows == 0:
        n_rows = len(data_set.dataset.data_x) if hasattr(data_set, 'dataset') else len(getattr(data_set, 'data_x', []))
        raise ValueError(
            f"[{flag}] 0 windows: split has {n_rows} rows but "
            f"seq_len={args.seq_len}+pred_len={args.pred_len}={args.seq_len+args.pred_len} required. "
            f"Reduce seq_len or pred_len."
        )
    num_workers = args.num_workers
    import os
    hostname = os.uname().nodename if hasattr(os, "uname") else ""
    if not torch.cuda.is_available() or "login" in hostname:
        print(f"Running on login node or CPU: forcing num_workers = 0 (was {num_workers})")
        num_workers = 0

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return data_set, data_loader
