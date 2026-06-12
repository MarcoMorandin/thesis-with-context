import argparse
import os
import torch
import numpy as np
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
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
    candidates = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.dirname(os.path.dirname(current_dir)),
    ]
    for base in candidates:
        if os.path.isdir(os.path.join(base, 'dataset')) or os.path.isfile(os.path.join(base, 'README.md')):
            return base
    return current_dir


PROJECT_ROOT = _get_project_root()

# On HPC: set SOLARVLM_SCRATCH=/leonardo_scratch/fast/IscrC_MTSFM/SolarVLM
# All large files (datasets, weights, checkpoints) go there; code stays in home.
_HPC_SCRATCH = os.environ.get('SOLARVLM_SCRATCH', '')
SCRATCH_ROOT = _HPC_SCRATCH if _HPC_SCRATCH else PROJECT_ROOT

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Fixed SolarVLM v3 for PV Power Forecasting')

    # ==================== Basic config ====================
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='1: training, 0: testing')
    parser.add_argument('--model_id', type=str, default='pv_forecast_fixed_v3')
    parser.add_argument('--model', type=str, default='SolarVLM')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')

    # ==================== Data loader ====================
    parser.add_argument('--data', type=str, default='PV')
    parser.add_argument('--root_path', type=str, default=os.path.join(SCRATCH_ROOT, 'dataset', '河北光伏发电'))
    parser.add_argument('--data_path', type=str, default='station_data.csv')
    parser.add_argument('--features', type=str, default='MS', 
                        help='M: multivariate, S: univariate, MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='power')
    parser.add_argument('--freq', type=str, default='t', help='t: 15min, h: hourly')
    parser.add_argument('--checkpoints', type=str, default=os.path.join(SCRATCH_ROOT, 'checkpoints'))
    parser.add_argument('--results_dir', type=str, default=os.path.join(SCRATCH_ROOT, 'results'))
    parser.add_argument('--test_results_dir', type=str, default=os.path.join(SCRATCH_ROOT, 'test_results'))

    # ==================== Forecasting task ====================
    parser.add_argument('--seq_len', type=int, default=288, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=144, help='decoder start token length (= seq_len // 2 for Full setting)')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction length')
    parser.add_argument('--inverse', action='store_true', default=True, help='inverse transform output')

    # ==================== Model architecture ====================
    parser.add_argument('--enc_in', type=int, default=112, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=112, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size (num stations)')
    parser.add_argument('--d_model', type=int, default=128, help='model dimension')
    parser.add_argument('--n_heads', type=int, default=16, help='num attention heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--activation', type=str, default='gelu')

    # ==================== SolarVLM specific ====================
    parser.add_argument('--vlm_type', type=str, default='qwen3vl')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--memory_bank_size', type=int, default=20)
    parser.add_argument('--patch_memory_size', type=int, default=100)
    parser.add_argument('--periodicity', type=int, default=96)
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--top_k', type=int, default=5)

    parser.add_argument('--qwen3_vl_model_path', type=str,
        default=os.path.join(SCRATCH_ROOT, 'QwenQwen3-VL-Embedding-2B'))
    parser.add_argument('--vlm_embed_dim', type=int, default=2048)  

    # ==================== PV specific ====================
    parser.add_argument('--num_stations', type=int, default=8)
    parser.add_argument('--roi_size', type=int, default=64)
    parser.add_argument('--num_frames', type=int, default=8, help='number of vision frames for temporal encoding')
    parser.add_argument('--use_gnn', type=str2bool, default=True)
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--gnn_k', type=int, default=5, help='kNN neighbors for GraphLearner (<= num_stations-1)')

    # ==================== Training parameters ====================
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1, help='experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=50, help='Phase 3 epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--memory_loss_weight', type=float, default=0.05)
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'huber'])
    parser.add_argument('--huber_beta', type=float, default=1.0)


    # ==================== 分阶段训练参数 ====================
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Phase 1: temporal backbone warmup epochs')
    parser.add_argument('--multimodal_epochs', type=int, default=10, help='Phase 2: multimodal branch training epochs')
    parser.add_argument('--multimodal_lr_ratio', type=float, default=0.2, 
                        help='Phase 3: multimodal learning rate ratio relative to base lr')
    # 【修复】统一默认值为 0.1
    parser.add_argument('--modal_dropout_rate', type=float, default=0.0, help='modality dropout rate during training')

    # ==================== 训练稳定性参数 ====================
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='gradient clipping threshold')
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help='learning rate warmup steps per phase')

    # ==================== Loss weights ====================
    parser.add_argument('--multimodal_loss_weight', type=float, default=0.1, help='weight for multimodal auxiliary loss')
    parser.add_argument('--modal_temp', type=float, default=0.7)
    parser.add_argument('--min_modal_weight', type=float, default=0.0)
    # 【修复】统一默认值为 False
    parser.add_argument('--nonnegative', type=str2bool, default=False, help='apply softplus to predictions')

    parser.add_argument('--lradj', type=str, default='type1', help='learning rate adjustment strategy')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision')

    # ==================== Feature control ====================
    parser.add_argument('--learnable_image', type=str2bool, default=False)
    parser.add_argument('--save_images', type=str2bool, default=False)
    parser.add_argument('--use_mem_gate', type=str2bool, default=True)

    parser.add_argument('--disable_visual', type=str2bool, default=False, help='disable visual modality')
    parser.add_argument('--disable_text', type=str2bool, default=False, help='disable text modality')
    parser.add_argument('--disable_gnn', type=str2bool, default=False, help='disable GraphLearner (spatial GNN)')
    parser.add_argument('--disable_cross_site_attn', type=str2bool, default=False, help='disable CrossStationAttention (cross-site attn)')

    # ==================== GPU ====================
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='GPU device ids for multi-GPU')

    # ==================== Patch embedding ====================
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding', type=int, default=8)
    parser.add_argument('--patch_len', type=int, default=10)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--seasonal_patterns', type=str, default='')
    parser.add_argument('--percent', type=float, default=1, help='data percentage for few-shot learning')
    parser.add_argument('--use_dtw', type=bool, default=False)

    # ==================== Vision ====================
    parser.add_argument('--use_offline_vision', action='store_true', default=True)
    parser.add_argument('--vision_feat_dir', type=str, default=os.path.join(SCRATCH_ROOT, 'vision_feats_qwen3vl'))
    parser.add_argument('--clip_model_path', type=str, default=os.path.join(SCRATCH_ROOT, 'clip-vit-base-patch32'))
    parser.add_argument('--vision_temporal_layers', type=int, default=2, help='num layers in vision temporal encoder')

    # ==================== Time ====================
    parser.add_argument('--base_year', type=int, default=2018)
    parser.add_argument('--base_month', type=int, default=12)
    parser.add_argument('--base_hour', type=int, default=0)
    parser.add_argument('--start_time', type=str, default='2018-12-01 00:00')
    parser.add_argument('--end_time', type=str, default='2019-06-01 00:00')

    args = parser.parse_args()
    set_seed(args.seed)

    # ==================== 自动配置 ====================
    if not hasattr(args, 'use_dtw'):
        args.use_dtw = False

    # 【修复】不再硬编码覆盖 use_gpu，而是根据 cuda 可用性自动判断
    args.use_gpu = torch.cuda.is_available()
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.content = "Photovoltaic power generation forecasting for multiple stations in Hebei Province"

    # ==================== 打印配置 ====================
    print('=' * 70)
    print('SolarVLM v3 Configuration')
    print('=' * 70)
    
    # 分类打印参数
    categories = {
        'Data': ['data', 'root_path', 'features', 'target', 'freq'],
        'Sequence': ['seq_len', 'label_len', 'pred_len'],
        'Model': ['d_model', 'n_heads', 'e_layers', 'dropout'],
        'Training': ['train_epochs', 'warmup_epochs', 'multimodal_epochs', 'batch_size', 'learning_rate'],
        'Multimodal': ['multimodal_lr_ratio', 'modal_dropout_rate', 'multimodal_loss_weight',
                       'disable_visual', 'disable_text', 'disable_gnn', 'disable_cross_site_attn'],
        'Stability': ['grad_clip_norm', 'lr_warmup_steps', 'patience'],
        'Vision': ['num_frames', 'vision_temporal_layers', 'use_offline_vision'],
        'GPU': ['use_gpu', 'gpu', 'use_multi_gpu'],
    }
    
    for cat_name, cat_params in categories.items():
        print(f"\n[{cat_name}]")
        for param in cat_params:
            if hasattr(args, param):
                print(f"  {param}: {getattr(args, param)}")
    
    print('\n' + '=' * 70)

    # ==================== 运行实验 ====================
    from exp.experiment import Experiment
    Exp = Experiment

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_st{}_wu{}_mm{}_mmlr{}_gc{}_v3_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.num_stations,
                args.warmup_epochs,
                args.multimodal_epochs,
                args.multimodal_lr_ratio,
                args.grad_clip_norm,
                ii
            )
            # 追加消融后缀，避免 checkpoint 覆盖
            abla = f"_abV{int(args.disable_visual)}T{int(args.disable_text)}G{int(args.disable_gnn)}C{int(args.disable_cross_site_attn)}"
            setting = setting + abla

            print('=' * 70)
            print(f'>>> Starting training: {setting}')
            print('=' * 70)
            exp.train(setting)

            print('=' * 70)
            print(f'>>> Starting testing: {setting}')
            print('=' * 70)
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        exp = Exp(args)
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_st{}_wu{}_mm{}_mmlr{}_gc{}_v3_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.num_stations,
            args.warmup_epochs,
            args.multimodal_epochs,
            args.multimodal_lr_ratio,
            args.grad_clip_norm,
            ii
        )

        print('=' * 70)
        print(f'>>> Starting testing: {setting}')
        print('=' * 70)
        exp.test(setting, test=1)

        torch.cuda.empty_cache()