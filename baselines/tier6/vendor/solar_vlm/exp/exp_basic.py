import torch
from src.SolarVLM import model as SolarVLM


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SolarVLM': SolarVLM,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        if args.is_training:
            self._log_model_parameters()

    def _log_model_parameters(self):
        learnable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Learnable model parameters: {learnable:,}")
        print(f"Total model parameters: {total:,}")

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
