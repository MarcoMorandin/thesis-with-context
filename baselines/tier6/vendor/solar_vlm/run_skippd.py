"""
Entry point for training/testing SolarVLM on the SKIPPD dataset
(solarbench/SKIPPD, HuggingFace).

Usage:
    python run_skippd.py                          # train + test
    python run_skippd.py --is_training 0          # test only (load checkpoint)
    python run_skippd.py --disable_visual True \\
                         --disable_text True       # ablation: temporal-only

SKIPPD is a single-station dataset, so GNN and cross-station attention are
disabled by default.  Vision features are loaded from --vision_feat_dir if
available; the dataset provides sky images that can be pre-embedded with
tools/precompute_vision_feats.py.
"""

import argparse
import os
import random

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed: int = 2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for base in [current_dir,
                 os.path.dirname(current_dir),
                 os.path.dirname(os.path.dirname(current_dir))]:
        if os.path.isdir(os.path.join(base, 'dataset')) or \
           os.path.isfile(os.path.join(base, 'README.md')):
            return base
    return current_dir


PROJECT_ROOT = _get_project_root()

# On HPC: set SOLARVLM_SCRATCH=/leonardo_scratch/fast/IscrC_MTSFM/SolarVLM
# All large files (datasets, weights, checkpoints) go there; code stays in home.
_HPC_SCRATCH = os.environ.get('SOLARVLM_SCRATCH', '')
SCRATCH_ROOT = _HPC_SCRATCH if _HPC_SCRATCH else PROJECT_ROOT


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    random.seed(2024)
    torch.manual_seed(2024)
    np.random.seed(2024)

    parser = argparse.ArgumentParser(
        description='SolarVLM — SKIPPD single-site solar forecasting')

    # ── Basic config ──────────────────────────────────────────────────────
    parser.add_argument('--task_name',   type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1,
                        help='1: train+test, 0: test only')
    parser.add_argument('--model_id',    type=str, default='skippd_v1')
    parser.add_argument('--model',       type=str, default='SolarVLM')
    parser.add_argument('--seed',        type=int, default=2024)

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument('--data',      type=str, default='SKIPPD')
    parser.add_argument('--root_path', type=str,
                        default=os.path.join(SCRATCH_ROOT, 'dataset', 'skippd'),
                        help='Directory for local SKIPPD cache')
    parser.add_argument('--data_path', type=str, default='skippd.csv',
                        help='Unused for SKIPPD; kept for API compatibility')
    parser.add_argument('--features',  type=str, default='MS')
    parser.add_argument('--target',    type=str, default='pv',
                        help='Target column name (auto-detected if not found)')
    parser.add_argument('--freq',      type=str, default='h',
                        help='h: hourly, t: minutely')
    parser.add_argument('--use_era5',  type=str2bool, default=True,
                        help='True: ERA5 covariates hourly; False: 1-min pv only')
    parser.add_argument('--checkpoints', type=str, default=os.path.join(SCRATCH_ROOT, 'checkpoints'))
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                        help='Explicit checkpoint .pth path. If set, overrides setting-based path in test mode.')
    parser.add_argument('--wollongong_station', type=str, default='loc1',
                        choices=['loc1', 'loc3'],
                        help='Which Wollongong PV system to use (only relevant for --data WOLLONGONG)')
    parser.add_argument('--results_dir', type=str, default=os.path.join(SCRATCH_ROOT, 'results'))
    parser.add_argument('--test_results_dir', type=str, default=os.path.join(SCRATCH_ROOT, 'test_results'))

    # ── Sequence lengths ──────────────────────────────────────────────────
    parser.add_argument('--seq_len',   type=int, default=72,
                        help='3 days of hourly input')
    parser.add_argument('--label_len', type=int, default=24,
                        help='1 day decoder overlap')
    parser.add_argument('--pred_len',  type=int, default=24,
                        help='1-day-ahead forecast')
    parser.add_argument('--inverse',   action='store_true', default=True)

    # ── Model architecture ────────────────────────────────────────────────
    # SKIPPD: 1 station, ~7-8 feature columns
    parser.add_argument('--enc_in',   type=int, default=7,
                        help='Total input features = num_features (auto-set)')
    parser.add_argument('--dec_in',   type=int, default=7)
    parser.add_argument('--c_out',    type=int, default=1,
                        help='Output size = num_stations = 1')
    parser.add_argument('--d_model',  type=int, default=128)
    parser.add_argument('--n_heads',  type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff',     type=int, default=512)
    parser.add_argument('--dropout',  type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu')

    # ── SolarVLM specific ─────────────────────────────────────────────────
    parser.add_argument('--vlm_type',          type=str,   default='qwen3vl')
    parser.add_argument('--image_size',        type=int,   default=224)
    parser.add_argument('--memory_bank_size',  type=int,   default=20)
    parser.add_argument('--patch_memory_size', type=int,   default=100)
    parser.add_argument('--periodicity',       type=int,   default=24,
                        help='24h for hourly SKIPPD')
    parser.add_argument('--norm_const',        type=float, default=0.4)
    parser.add_argument('--top_k',             type=int,   default=5)

    parser.add_argument('--qwen3_vl_model_path', type=str,
                        default=os.path.join(SCRATCH_ROOT,
                                             'QwenQwen3-VL-Embedding-2B'))
    parser.add_argument('--vlm_embed_dim', type=int, default=2048)

    # ── SKIPPD single-station config ──────────────────────────────────────
    parser.add_argument('--num_stations', type=int, default=1)
    parser.add_argument('--roi_size',     type=int, default=64)
    parser.add_argument('--num_frames',   type=int, default=8)

    # For single-site, GNN and cross-station attention are off by default
    parser.add_argument('--use_gnn',    type=str2bool, default=False)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--gnn_k',      type=int, default=0)

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--itr',            type=int,   default=1)
    parser.add_argument('--train_epochs',   type=int,   default=50)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--patience',       type=int,   default=7)
    parser.add_argument('--learning_rate',  type=float, default=0.0005)
    parser.add_argument('--memory_loss_weight',    type=float, default=0.05)
    parser.add_argument('--loss_type',      type=str,   default='mse',
                        choices=['mse', 'huber'])
    parser.add_argument('--huber_beta',     type=float, default=1.0)

    # ── Phased training ───────────────────────────────────────────────────
    parser.add_argument('--warmup_epochs',      type=int,   default=5)
    parser.add_argument('--multimodal_epochs',  type=int,   default=10)
    parser.add_argument('--multimodal_lr_ratio',type=float, default=0.2)
    parser.add_argument('--modal_dropout_rate', type=float, default=0.0)

    # ── Training stability ────────────────────────────────────────────────
    parser.add_argument('--grad_clip_norm',  type=float, default=1.0)
    parser.add_argument('--lr_warmup_steps', type=int,   default=300)

    # ── Loss weights ──────────────────────────────────────────────────────
    parser.add_argument('--multimodal_loss_weight', type=float, default=0.1)
    parser.add_argument('--modal_temp',       type=float, default=0.7)
    parser.add_argument('--min_modal_weight', type=float, default=0.0)
    parser.add_argument('--nonnegative',      type=str2bool, default=False,
                        help='Softplus output (power is non-negative)')

    parser.add_argument('--lradj',   type=str,  default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # ── Feature control / ablations ───────────────────────────────────────
    parser.add_argument('--learnable_image', type=str2bool, default=False)
    parser.add_argument('--save_images',     type=str2bool, default=False)
    parser.add_argument('--use_mem_gate',    type=str2bool, default=True)

    parser.add_argument('--disable_visual',           type=str2bool, default=False)
    parser.add_argument('--disable_text',             type=str2bool, default=False)
    parser.add_argument('--disable_gnn',              type=str2bool, default=True,
                        help='Single-site: GNN disabled by default')
    parser.add_argument('--disable_cross_site_attn',  type=str2bool, default=True,
                        help='Single-site: cross-station attention disabled')

    # ── GPU ───────────────────────────────────────────────────────────────
    parser.add_argument('--gpu',           type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices',       type=str, default='0,1,2,3')

    # ── Patch embedding ───────────────────────────────────────────────────
    parser.add_argument('--stride',    type=int, default=8)
    parser.add_argument('--padding',   type=int, default=8)
    parser.add_argument('--patch_len', type=int, default=12)
    parser.add_argument('--embed',     type=str, default='timeF')
    parser.add_argument('--seasonal_patterns', type=str, default='')
    parser.add_argument('--percent',   type=float, default=1.0)
    parser.add_argument('--use_dtw',   type=bool,  default=False)

    # ── Vision ────────────────────────────────────────────────────────────
    parser.add_argument('--use_offline_vision',    action='store_true', default=True)
    parser.add_argument('--vision_feat_dir',       type=str,
                        default=os.path.join(SCRATCH_ROOT,
                                             'vision_feats_skippd_qwen3vl'))
    parser.add_argument('--clip_model_path',       type=str,
                        default=os.path.join(SCRATCH_ROOT,
                                             'clip-vit-base-patch32'))
    parser.add_argument('--vision_temporal_layers',type=int, default=2)

    # ── Time range (informational; SKIPPD splits handled by HF) ──────────
    parser.add_argument('--base_year',   type=int, default=2010)
    parser.add_argument('--base_month',  type=int, default=1)
    parser.add_argument('--base_hour',   type=int, default=0)
    parser.add_argument('--start_time',  type=str, default='2010-01-01 00:00')
    parser.add_argument('--end_time',    type=str, default='2023-12-31 23:00')

    args = parser.parse_args()
    set_seed(args.seed)

    # ── Auto-configure ────────────────────────────────────────────────────
    if not hasattr(args, 'use_dtw'):
        args.use_dtw = False

    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu and args.use_multi_gpu:
        args.devices    = args.devices.replace(' ', '')
        device_ids      = args.devices.split(',')
        args.device_ids = [int(x) for x in device_ids]
        args.gpu        = args.device_ids[0]

    # Single-site station config injected into args so model.py can read it
    args.station_list      = ['skippd_site']
    args.station_coords    = {'skippd_site': (-105.1786, 39.7392)}
    args.station_positions = {'skippd_site': (0.5, 0.5)}
    from data_provider.data_loader_skippd import ERA5_NN1_COLS
    if args.data == 'SKIPPD' and args.use_era5:
        _feature_schema = ERA5_NN1_COLS + ['pv']   # 27 features
    else:
        _feature_schema = ['pv']                    # 1 feature (WOLLONGONG or 1-min SKIPPD)
    args.feature_schema        = _feature_schema
    args.station_feature_order = _feature_schema
    args.enc_in = len(_feature_schema)
    args.dec_in = args.enc_in

    args.content = ("Single-site solar power forecasting using the SKIPPD dataset "
                    "(solarbench/SKIPPD, HuggingFace).")

    # ── Print config ──────────────────────────────────────────────────────
    print('=' * 70)
    print('SolarVLM — SKIPPD Configuration')
    print('=' * 70)
    sections = {
        'Data':       ['data', 'root_path', 'features', 'target', 'freq'],
        'Sequence':   ['seq_len', 'label_len', 'pred_len'],
        'Model':      ['d_model', 'n_heads', 'e_layers', 'dropout'],
        'Training':   ['train_epochs', 'warmup_epochs', 'multimodal_epochs',
                       'batch_size', 'learning_rate'],
        'Multimodal': ['multimodal_lr_ratio', 'modal_dropout_rate',
                       'multimodal_loss_weight', 'disable_visual',
                       'disable_text', 'disable_gnn',
                       'disable_cross_site_attn'],
        'Stability':  ['grad_clip_norm', 'lr_warmup_steps', 'patience'],
        'Vision':     ['num_frames', 'vision_temporal_layers',
                       'use_offline_vision', 'vision_feat_dir'],
        'GPU':        ['use_gpu', 'gpu', 'use_multi_gpu'],
    }
    for sec, params in sections.items():
        print(f'\n[{sec}]')
        for p in params:
            if hasattr(args, p):
                print(f'  {p}: {getattr(args, p)}')
    print('\n' + '=' * 70)

    # ── Run experiments ───────────────────────────────────────────────────
    from exp.experiment import Experiment
    Exp = Experiment

    if args.is_training:
        for ii in range(args.itr):
            exp     = Exp(args)
            setting = (
                f'{args.task_name}_{args.model_id}_{args.model}_{args.data}'
                f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}'
                f'_dm{args.d_model}_wu{args.warmup_epochs}'
                f'_mm{args.multimodal_epochs}_gc{args.grad_clip_norm}_v1_{ii}'
            )
            abla = (f'_abV{int(args.disable_visual)}'
                    f'T{int(args.disable_text)}'
                    f'G{int(args.disable_gnn)}'
                    f'C{int(args.disable_cross_site_attn)}')
            setting += abla

            print('=' * 70)
            print(f'>>> Training: {setting}')
            print('=' * 70)
            exp.train(setting)

            print('=' * 70)
            print(f'>>> Testing: {setting}')
            print('=' * 70)
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii  = 0
        exp = Exp(args)
        setting = (
            f'{args.task_name}_{args.model_id}_{args.model}_{args.data}'
            f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}'
            f'_dm{args.d_model}_wu{args.warmup_epochs}'
            f'_mm{args.multimodal_epochs}_gc{args.grad_clip_norm}_v1_{ii}'
        )
        abla = (f'_abV{int(args.disable_visual)}'
                f'T{int(args.disable_text)}'
                f'G{int(args.disable_gnn)}'
                f'C{int(args.disable_cross_site_attn)}')
        setting += abla

        print('=' * 70)
        print(f'>>> Testing: {setting}')
        print('=' * 70)
        exp.test(setting, test=1)

        torch.cuda.empty_cache()
