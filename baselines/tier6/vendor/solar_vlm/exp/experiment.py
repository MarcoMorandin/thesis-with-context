"""
修复版训练流程 v3
关键修复：
1. 修复学习率预热（正确设置 initial_lr）
2. Phase 3 复用 _train_epoch 方法
3. 正确处理 multimodal_scale
4. 每个 Phase 独立的 global_step
5. 更清晰的日志输出
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import R2, metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Experiment(Exp_Basic):
    def __init__(self, args):
        super(Experiment, self).__init__(args)
        self.memory_loss_weight = getattr(self.args, 'memory_loss_weight', 0.0)
        self.multimodal_loss_weight = getattr(self.args, 'multimodal_loss_weight', 0.2)
        self.warmup_epochs = getattr(self.args, 'warmup_epochs', 10)
        self.multimodal_lr_ratio = getattr(self.args, 'multimodal_lr_ratio', 0.2)
        
        # 梯度裁剪参数
        self.grad_clip_norm = getattr(self.args, 'grad_clip_norm', 1.0)
        
        # 学习率预热步数
        self.lr_warmup_steps = getattr(self.args, 'lr_warmup_steps', 500)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, phase='full'):
        """
        分阶段优化器
        phase: 'warmup' - 只训练时序骨干
               'multimodal' - 固定时序骨干，只训练多模态
               'full' - 全部训练但多模态使用更低学习率
        """
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        disable_visual = bool(getattr(self.args, "disable_visual", False))
        disable_text   = bool(getattr(self.args, "disable_text", False))
        disable_gnn    = bool(getattr(self.args, "disable_gnn", False))
        disable_csa    = bool(getattr(self.args, "disable_csa",
                            getattr(self.args, "disable_cross_site_attn", False)))
        
        # 时序骨干相关参数名
        temporal_keywords = [
            'patch_embedding', 'memory_head', 'temporal_head',
            'patch_memory_bank', 'local_memory_mlp', 'memory_attention',
            'temporal_feature', 'memory_feature', 'station_attention',
            'station_attn_norm', 'station_weights', 'memory_fusion_gate',
            'memory_pred_in', 'memory_pred_out', 'flatten', 'alpha',
            'graph_learner'
        ]
        
        multimodal_keywords = [
            'visual_adapter', 'visual_temporal', 'visual_proj', 'vision_store',  # 视觉
            'text_encoder', 'text_proj',                                         # 文本
            'modality_gate', 'multimodal_fusion',
            'cross_attention', 'cross_station_attn', 'multimodal_head',
            'multimodal_scale'
        ]

        if disable_gnn:
            temporal_keywords = [k for k in temporal_keywords if "graph_learner" not in k]

        if disable_csa:
            multimodal_keywords = [k for k in multimodal_keywords
                                if ("cross_station_attn" not in k and "cross_attention" not in k)]

        # ✅ 最小侵入：按开关剔除关键词（让 optimizer 不碰被禁用分支）
        if disable_visual:
            multimodal_keywords = [k for k in multimodal_keywords if not k.startswith("visual") and "vision" not in k]

        if disable_text:
            multimodal_keywords = [k for k in multimodal_keywords if "text" not in k]
        
        if phase == 'warmup':
            temporal_params = []
            for name, param in model.named_parameters():

                # ===== 硬保险：禁用分支直接冻结 =====
                if disable_visual and ("visual" in name or "vision" in name):
                    param.requires_grad = False
                    continue
                if disable_text and ("text" in name):
                    param.requires_grad = False
                    continue
                if disable_gnn and ("graph_learner" in name):
                    param.requires_grad = False
                    continue
                if disable_csa and ("cross_station_attn" in name):
                    param.requires_grad = False
                    continue
                # ===================================

                is_temporal = any(key in name for key in temporal_keywords)
                if is_temporal:
                    param.requires_grad = True
                    temporal_params.append(param)
                else:
                    param.requires_grad = False
            
            print(f"[Optimizer] Warmup phase: {len(temporal_params)} temporal params, lr={self.args.learning_rate}")
            optimizer = optim.AdamW(temporal_params, lr=self.args.learning_rate, weight_decay=0.01)
            # 【修复】设置 initial_lr 用于预热
            for pg in optimizer.param_groups:
                pg['initial_lr'] = pg['lr']
            return optimizer
            
        elif phase == 'multimodal':
            multimodal_params = []
            for name, param in model.named_parameters():

                # ===== 硬保险：禁用分支直接冻结 =====
                if disable_visual and ("visual" in name or "vision" in name):
                    param.requires_grad = False
                    continue
                if disable_text and ("text" in name):
                    param.requires_grad = False
                    continue
                if disable_gnn and ("graph_learner" in name):
                    param.requires_grad = False
                    continue
                if disable_csa and ("cross_station_attn" in name):
                    param.requires_grad = False
                    continue
                # ===================================

                is_multimodal = any(key in name for key in multimodal_keywords)
                if is_multimodal:
                    param.requires_grad = True
                    multimodal_params.append(param)
                else:
                    param.requires_grad = False

            lr_mm = self.args.learning_rate
            print(f"[Optimizer] Multimodal phase: {len(multimodal_params)} multimodal params, lr={lr_mm}")
            optimizer = optim.AdamW(multimodal_params, lr=lr_mm, weight_decay=0.01)
            for pg in optimizer.param_groups:
                pg['initial_lr'] = pg['lr']
            return optimizer

        else:  # full
            temporal_params = []
            multimodal_params = []

            for name, param in model.named_parameters():

                # ===== 硬保险：禁用分支直接冻结 =====
                if disable_visual and ("visual" in name or "vision" in name):
                    param.requires_grad = False
                    continue
                if disable_text and ("text" in name):
                    param.requires_grad = False
                    continue
                if disable_gnn and ("graph_learner" in name):
                    param.requires_grad = False
                    continue
                if disable_csa and ("cross_station_attn" in name):
                    param.requires_grad = False
                    continue
                # ===================================

                param.requires_grad = True
                is_multimodal = any(key in name for key in multimodal_keywords)
                if is_multimodal:
                    multimodal_params.append(param)
                else:
                    temporal_params.append(param)

            
            lr_temporal = self.args.learning_rate
            lr_mm = self.args.learning_rate * self.multimodal_lr_ratio
            
            print(f"[Optimizer] Full phase: {len(temporal_params)} temporal params (lr={lr_temporal}), "
                  f"{len(multimodal_params)} multimodal params (lr={lr_mm})")
            
            optimizer = optim.AdamW([
                {'params': temporal_params, 'lr': lr_temporal, 'initial_lr': lr_temporal},
                {'params': multimodal_params, 'lr': lr_mm, 'initial_lr': lr_mm}
            ], weight_decay=0.01)
            return optimizer

    def _select_criterion(self):
        if getattr(self.args, 'loss_type', 'mse') == 'huber':
            return nn.HuberLoss(delta=getattr(self.args, 'huber_beta', 1.0), reduction='none')
        else:
            return nn.MSELoss(reduction='none')

    def _calc_loss(self, outputs, memory_outputs, multimodal_outputs, target, criterion, target_mark=None):
        """计算损失函数"""
        base = criterion(outputs, target)
        if base.dim() == 1:
            base = base.unsqueeze(0)

        loss_main = base.mean()
        loss = loss_main

        loss_aux_mem = 0.0
        if memory_outputs is not None and self.memory_loss_weight > 0:
            aux_base = criterion(memory_outputs, target)
            loss_aux_mem = aux_base.mean()
            loss = loss + self.memory_loss_weight * loss_aux_mem

        loss_aux_mm = 0.0
        if multimodal_outputs is not None and self.multimodal_loss_weight > 0:
            aux_mm = criterion(multimodal_outputs, target)
            loss_aux_mm = aux_mm.mean()
            loss = loss + self.multimodal_loss_weight * loss_aux_mm

        return loss, loss_main, loss_aux_mem, loss_aux_mm

    def _reset_vision_stats(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, 'vision_hits'):
            model.vision_hits = 0
            model.vision_requests = 0

    def _log_vision_stats(self, tag):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, 'vision_requests') and model.vision_requests > 0:
            rate = model.get_vision_hit_rate() if hasattr(model, 'get_vision_hit_rate') else model.vision_hits / model.vision_requests
            print(f"[Vision] {tag} hit rate: {rate:.2%} ({model.vision_hits}/{model.vision_requests})")
        self._reset_vision_stats()
    
    def _log_model_stats(self, tag):
        """打印模型关键参数状态"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        if hasattr(model, 'multimodal_scale'):
            raw_val = model.multimodal_scale.item()
            scale_val = torch.sigmoid(model.multimodal_scale).item() * 0.8 + 0.1
            print(f"[Model] {tag} multimodal_scale: raw={raw_val:.4f}, effective={scale_val:.4f}")
        
        if hasattr(model, 'station_weights'):
            weights = torch.softmax(model.station_weights, dim=0).detach().cpu().numpy()
            print(f"[Model] {tag} station_weights: {np.round(weights, 3)}")

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_ts_keys in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                label_token = batch_y[:, :self.args.label_len, :]
                zeros = torch.zeros(
                    batch_y.size(0),
                    self.args.pred_len,
                    batch_y.size(2),
                    device=self.device,
                )
                dec_inp = torch.cat([label_token, zeros], dim=1)

                outputs, memory_outputs, multimodal_outputs = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, ts_keys=batch_ts_keys
                )

                target = batch_y[:, -self.args.pred_len:, :]

                # 验证阶段：不加多模态辅助 loss
                loss, _, _, _ = self._calc_loss(
                    outputs, memory_outputs, None, target, criterion
                )

                if not torch.isfinite(loss).item():
                    print("[WARN] NaN/Inf loss in validation, skip this batch")
                    continue

                total_loss.append(loss.item())

        self.model.train()
        if len(total_loss) == 0:
            return float('inf')
        return float(np.average(total_loss))

    def _train_epoch(self, train_loader, optimizer, criterion, use_amp, scaler, 
                     epoch_desc="", global_step=0, use_lr_warmup=True, 
                     use_mm_loss=True, log_interval=100):
        """
        【修复版】单轮训练
        修复：
        1. 正确使用 initial_lr 进行预热
        2. 支持可选的多模态损失
        3. 更详细的日志
        """
        train_loss = []
        self.model.train()
        epoch_time = time.time()
        
        step = global_step
        time_now = time.time()
        iter_count = 0

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_ts_keys) in enumerate(train_loader):
            optimizer.zero_grad()
            
            step += 1
            iter_count += 1
            
            # 【修复】学习率预热（使用 initial_lr）
            if use_lr_warmup and step <= self.lr_warmup_steps:
                warmup_factor = step / self.lr_warmup_steps
                for param_group in optimizer.param_groups:
                    init_lr = param_group.get('initial_lr', self.args.learning_rate)
                    param_group['lr'] = init_lr * warmup_factor

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            label_token = batch_y[:, :self.args.label_len, :]
            zeros = torch.zeros(
                batch_y.size(0),
                self.args.pred_len,
                batch_y.size(2),
                device=self.device,
            )
            dec_inp = torch.cat([label_token, zeros], dim=1)
            target = batch_y[:, -self.args.pred_len:, :]

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs, memory_outputs, multimodal_outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, ts_keys=batch_ts_keys
                    )
                    mm_out = multimodal_outputs if use_mm_loss else None
                    loss, _, _, _ = self._calc_loss(
                        outputs, memory_outputs, mm_out, target, criterion
                    )
                if not torch.isfinite(loss).item():
                    print("[WARN] NaN/Inf loss in train (AMP), skip this batch")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, memory_outputs, multimodal_outputs = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, ts_keys=batch_ts_keys
                )
                mm_out = multimodal_outputs if use_mm_loss else None
                loss, _, _, _ = self._calc_loss(
                    outputs, memory_outputs, mm_out, target, criterion
                )
                if not torch.isfinite(loss).item():
                    print("[WARN] NaN/Inf loss in train, skip this batch")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()

            train_loss.append(loss.item())
            
            # 日志输出
            if (i + 1) % log_interval == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  {epoch_desc} | iter {i+1}/{len(train_loader)} | "
                      f"loss: {loss.item():.6f} | lr: {cur_lr:.2e}")

        avg_loss = np.average(train_loss) if len(train_loss) > 0 else float('inf')
        print(f"{epoch_desc} | time: {time.time() - epoch_time:.1f}s | avg loss: {avg_loss:.6f}")
        return avg_loss, step

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self._reset_vision_stats()

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        model_inner = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # ===== 阶段1：预热时序骨干（关闭多模态）=====
        if self.warmup_epochs > 0:
            print("=" * 70)
            print(f"Phase 1: Warming up temporal backbone for {self.warmup_epochs} epochs")
            print("(Multimodal branches disabled)")
            print("=" * 70)
            
            # 保存原始状态
            original_disable_visual = model_inner.disable_visual
            original_disable_text = model_inner.disable_text
            
            # 关闭多模态
            model_inner.disable_visual = True
            model_inner.disable_text = True
            
            warmup_optim = self._select_optimizer(phase='warmup')
            global_step_p1 = 0
            
            for epoch in range(self.warmup_epochs):
                _, global_step_p1 = self._train_epoch(
                    train_loader, warmup_optim, criterion, 
                    self.args.use_amp, scaler,
                    epoch_desc=f"[Phase1] Epoch {epoch + 1}/{self.warmup_epochs}",
                    global_step=global_step_p1,
                    use_lr_warmup=True,
                    use_mm_loss=False,
                    log_interval=200
                )
            
            # 恢复状态
            model_inner.disable_visual = original_disable_visual
            model_inner.disable_text = original_disable_text
            model_inner.disable_visual = original_disable_visual
            model_inner.disable_text = original_disable_text
            
            self._log_model_stats("Phase 1 End")
            print("=" * 70)

        # ===== 阶段2：固定时序骨干，训练多模态分支 =====
        multimodal_epochs = getattr(self.args, 'multimodal_epochs', 10)
        if multimodal_epochs > 0 and not (model_inner.disable_visual and model_inner.disable_text):
            print("=" * 70)
            print(f"Phase 2: Training multimodal branches for {multimodal_epochs} epochs")
            print("(Temporal backbone frozen)")
            print("=" * 70)

            multimodal_optim = self._select_optimizer(phase='multimodal')

            # 【修复】Phase 2：放大 multimodal_scale 让多模态能学到东西
            saved_scale = None
            saved_modal_dropout = None
            
            if hasattr(model_inner, "multimodal_scale"):
                saved_scale = model_inner.multimodal_scale.data.clone()
                # 设为较大值 (sigmoid(2.0)*0.8+0.1 ≈ 0.81)
                model_inner.multimodal_scale.data.fill_(2.0)
                print(f"[Phase 2] Temporarily set multimodal_scale to 2.0 (effective ~0.81)")
                
            if hasattr(model_inner, "modal_dropout_rate"):
                saved_modal_dropout = model_inner.modal_dropout_rate
                model_inner.modal_dropout_rate = 0.0  # 关闭 dropout 让多模态更稳定
            
            global_step_p2 = 0
            for epoch in range(multimodal_epochs):
                _, global_step_p2 = self._train_epoch(
                    train_loader, multimodal_optim, criterion,
                    self.args.use_amp, scaler,
                    epoch_desc=f"[Phase2] Epoch {epoch + 1}/{multimodal_epochs}",
                    global_step=global_step_p2,
                    use_lr_warmup=True,
                    use_mm_loss=True,
                    log_interval=200
                )

            # 【修复】Phase 2 后：恢复 modal_dropout，但保留学习到的 scale
            if saved_modal_dropout is not None:
                model_inner.modal_dropout_rate = saved_modal_dropout
            
            # 不再强制重置 scale，让模型保留学习结果
            # 但如果学习结果太极端，可以做一些约束
            if hasattr(model_inner, "multimodal_scale"):
                current_scale = model_inner.multimodal_scale.data.item()
                # 限制在合理范围 [-2, 3]，对应 effective [0.12, 0.85]
                clamped = max(-2.0, min(3.0, current_scale))
                if clamped != current_scale:
                    model_inner.multimodal_scale.data.fill_(clamped)
                    print(f"[Phase 2] Clamped multimodal_scale from {current_scale:.2f} to {clamped:.2f}")
            
            self._log_model_stats("Phase 2 End")
            print("=" * 70)

        # ===== 阶段3：联合微调 =====
        print("=" * 70)
        print(f"Phase 3: Joint fine-tuning for {self.args.train_epochs} epochs")
        print("=" * 70)
        
        model_optim = self._select_optimizer(phase='full')
        global_step_p3 = 0
        
        for epoch in range(self.args.train_epochs):
            train_loss, global_step_p3 = self._train_epoch(
                train_loader, model_optim, criterion,
                self.args.use_amp, scaler,
                epoch_desc=f"[Phase3] Epoch {epoch + 1}/{self.args.train_epochs}",
                global_step=global_step_p3,
                use_lr_warmup=(epoch == 0),  # 只在第一轮做预热
                use_mm_loss=True,
                log_interval=100
            )

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"[Phase3] Epoch {epoch + 1}: Train={train_loss:.6f} | Vali={vali_loss:.6f} | Test={test_loss:.6f}")
            
            self._log_model_stats(f"Epoch {epoch + 1}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最佳模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        
        self._log_vision_stats('Train')
        self._log_model_stats('Final')
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        self._reset_vision_stats()

        if test:
            override = getattr(self.args, 'load_checkpoint_path', None)
            ckpt_path = override if override else os.path.join(
                self.args.checkpoints, setting, 'checkpoint.pth')
            state_dict = torch.load(ckpt_path, map_location=self.device)
            msg = self.model.load_state_dict(state_dict)
            if msg.missing_keys or msg.unexpected_keys:
                print(f"Warning: Loaded checkpoint with strict=False.")
                if msg.missing_keys:
                    print(f"  Missing keys (initialized randomly): {msg.missing_keys}")
                if msg.unexpected_keys:
                    print(f"  Unexpected keys (ignored): {msg.unexpected_keys}")

        preds_final, preds_mem, preds_mm = [], [], []
        preds_final_norm, preds_mem_norm, preds_mm_norm = [], [], []
        trues, trues_norm = [], []

        folder_path = os.path.join(getattr(self.args, 'test_results_dir', './test_results'), setting) + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_ts_keys) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                label_token = batch_y[:, :self.args.label_len, :]
                zeros = torch.zeros(
                    batch_y.size(0),
                    self.args.pred_len,
                    batch_y.size(2),
                    device=self.device,
                )
                dec_inp = torch.cat([label_token, zeros], dim=1)
                target = batch_y[:, -self.args.pred_len:, :]

                outputs, memory_outputs, multimodal_outputs = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, ts_keys=batch_ts_keys
                )

                final_out = outputs[:, -self.args.pred_len:, :]
                mem_out = memory_outputs[:, -self.args.pred_len:, :] if memory_outputs is not None else None
                mm_out = multimodal_outputs[:, -self.args.pred_len:, :] if multimodal_outputs is not None else None

                final_np = final_out.detach().cpu().numpy()
                tgt_np = target.detach().cpu().numpy()
                mem_np = mem_out.detach().cpu().numpy() if mem_out is not None else None
                mm_np = mm_out.detach().cpu().numpy() if mm_out is not None else None

                preds_final_norm.append(final_np)
                trues_norm.append(tgt_np)
                if mem_np is not None:
                    preds_mem_norm.append(mem_np)
                if mm_np is not None:
                    preds_mm_norm.append(mm_np)

                # 反归一化
                if test_data.scale and self.args.inverse:
                    final_inv = test_data.inverse_transform(final_np)
                    tgt_inv = test_data.inverse_transform(tgt_np)
                    mem_inv = test_data.inverse_transform(mem_np) if mem_np is not None else None
                    mm_inv = test_data.inverse_transform(mm_np) if mm_np is not None else None
                else:
                    final_inv = final_np
                    tgt_inv = tgt_np
                    mem_inv = mem_np
                    mm_inv = mm_np

                preds_final.append(final_inv)
                trues.append(tgt_inv)
                if mem_inv is not None:
                    preds_mem.append(mem_inv)
                if mm_inv is not None:
                    preds_mm.append(mm_inv)

        if len(preds_final) == 0:
            print("[ERROR] Test set produced 0 batches — seq_len too large for test split size.")
            return {'mse': float('nan'), 'mae': float('nan'), 'rmse': float('nan'),
                    'mse_norm': float('nan'), 'mae_norm': float('nan')}

        # 合并结果
        preds_final = np.concatenate(preds_final, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_final_norm = np.concatenate(preds_final_norm, axis=0)
        trues_norm = np.concatenate(trues_norm, axis=0)

        preds_mem = np.concatenate(preds_mem, axis=0) if len(preds_mem) > 0 else None
        preds_mm = np.concatenate(preds_mm, axis=0) if len(preds_mm) > 0 else None
        preds_mem_norm = np.concatenate(preds_mem_norm, axis=0) if len(preds_mem_norm) > 0 else None
        preds_mm_norm = np.concatenate(preds_mm_norm, axis=0) if len(preds_mm_norm) > 0 else None

        print('Test shape (final):', preds_final.shape, trues.shape)
        # ======================
        # Case Study (station00 only, 8 curves)
        # ======================
        case_indices = [0, 10, 20, 30, 50, 80, 120, 150]
        N = preds_final.shape[0]

        for idx in case_indices:
            if idx >= N:
                continue
            # station00 -> index 0
            visual(
                trues[idx, :, 0],
                preds_final[idx, :, 0],
                os.path.join(getattr(self.args, 'test_results_dir', './test_results'), setting, f"case_{idx}_station00.pdf")
            )


        # Reshape
        preds_final = preds_final.reshape(-1, preds_final.shape[-2], preds_final.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        preds_final_norm = preds_final_norm.reshape(-1, preds_final_norm.shape[-2], preds_final_norm.shape[-1])
        trues_norm = trues_norm.reshape(-1, trues_norm.shape[-2], trues_norm.shape[-1])

        if preds_mem is not None:
            preds_mem = preds_mem.reshape(-1, preds_mem.shape[-2], preds_mem.shape[-1])
            preds_mem_norm = preds_mem_norm.reshape(-1, preds_mem_norm.shape[-2], preds_mem_norm.shape[-1])
        if preds_mm is not None:
            preds_mm = preds_mm.reshape(-1, preds_mm.shape[-2], preds_mm.shape[-1])
            preds_mm_norm = preds_mm_norm.reshape(-1, preds_mm_norm.shape[-2], preds_mm_norm.shape[-1])

        # 保存结果
        folder_path = os.path.join(getattr(self.args, 'results_dir', './results'), setting) + '/'
        os.makedirs(folder_path, exist_ok=True)

        # 计算指标
        mae_f, mse_f, rmse_f, mape_f, mspe_f = metric(preds_final, trues)
        mae_fn, mse_fn, rmse_fn, mape_fn, mspe_fn = metric(preds_final_norm, trues_norm)

        r2_f  = R2(preds_final, trues)
        r2_fn = R2(preds_final_norm, trues_norm)

        mse_m, mae_m = None, None
        if preds_mem is not None:
            mae_m, mse_m, rmse_m, mape_m, mspe_m = metric(preds_mem, trues)

        mse_mm, mae_mm = None, None
        if preds_mm is not None:
            mae_mm, mse_mm, rmse_mm, mape_mm, mspe_mm = metric(preds_mm, trues)
        
        print("=" * 60)
        print("[FINAL] MSE: {:.6f}, MAE: {:.6f}, RMSE: {:.6f}".format(mse_f, mae_f, rmse_f))
        print("[FINAL] R2: {:.6f}".format(r2_f))

        print("[FINAL normalized] MSE: {:.6f}, MAE: {:.6f}".format(mse_fn, mae_fn))
        print("[FINAL normalized] R2: {:.6f}".format(r2_fn))

        print("=" * 60)

        # 保存文件
        np.save(folder_path + 'metrics_final.npy',
                np.array([mae_f, mse_f, rmse_f, mape_f, mspe_f]))
        np.save(folder_path + 'pred_final.npy', preds_final)
        np.save(folder_path + 'true.npy', trues)

        if preds_mem is not None:
            np.save(folder_path + 'pred_memory.npy', preds_mem)
        if preds_mm is not None:
            np.save(folder_path + 'pred_mm.npy', preds_mm)

        # 记录结果
        with open(os.path.join(getattr(self.args, 'results_dir', '.'), "result_long_term_forecast.txt"), 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"[FINAL] MSE: {mse_f}, MAE: {mae_f}, RMSE: {rmse_f}\n")
            f.write(f"[FINAL normalized] MSE: {mse_fn}, MAE: {mae_fn}\n")
            if preds_mem is not None:
                f.write(f"[MEMORY] MSE: {mse_m}, MAE: {mae_m}\n")
            if preds_mm is not None:
                f.write(f"[MULTIMODAL] MSE: {mse_mm}, MAE: {mae_mm}\n")
            f.write("\n")

        self._log_vision_stats('Test')
        return {'mse': mse_f, 'mae': mae_f, 'rmse': rmse_f,'mse_norm': mse_fn, 'mae_norm': mae_fn} 