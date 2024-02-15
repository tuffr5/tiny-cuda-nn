import torch


def compute_l2_loss(output, targets):
    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    print(f"relative_l2_error: {relative_l2_error.mean():.6f}")
    return relative_l2_error.mean()


def compute_loss(output, targets, rate_bpp, lmbda):
    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    # print(f"output: {torch.isnan(output).any()}, targets: {torch.isnan(targets).any()}")
    # print(f"output: {torch.amax(output)}, {torch.amin(output)}")
    print(f"relative_l2_error: {relative_l2_error.mean():.6f}, rate_bpp: {rate_bpp:.6f}")
    return lmbda * rate_bpp + relative_l2_error.mean()


class Trainer():
    def __init__(self, model, image, **args):
        # some args
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.n_steps = args['n_steps']
        self.device = args['device']
        self.n_pixels = args['n_pixels']
        self.lmbda = args['lmbda']
        
        self.model = model
        parameters = [
            {'params': model.net.parameters(), 'lr': self.lr},
            {'params': model._scale, 'lr': 0.1 * self.lr},
        ]
        self.optimizer = torch.optim.Adam(parameters)

        try:
            batch = torch.rand([self.batch_size, 2], device=self.device, dtype=torch.float32)
            self.traced_image = torch.jit.trace(image, batch)
        except:
            # If tracing causes an error, fall back to regular execution
            print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
            self.traced_image = image
    
    def train(self):
        # FP training
        self._set_model_requires_grad(True)
        self._set_quant_params_requires_grad(False)
        for _ in range(self.n_steps): 
            batch, targets = self._get_batch_and_targets()
            output = self.model(batch)
            loss = compute_l2_loss(output, targets)
            self._step_optimizer(loss)

        self._set_model_requires_grad(False)
        self._set_quant_params_requires_grad(False)
        self.model.save_full_precision_param()
        self.model.init_quant_params(self.model.get_full_precision_param())
        self._set_quant_params_requires_grad(True)
        # Quantization training
        for _ in range(100):
            batch, targets = self._get_batch_and_targets() 
            # quantize and dequantize
            output = self.model(batch, quant=True)
            rate_bpp = self.model.measure_laplace_rate() / (self.n_pixels * 8)
            loss = compute_loss(output, targets, rate_bpp, self.lmbda)
            self._step_optimizer(loss)

        # quantize model
        self.model.quantize()

    def _get_batch_and_targets(self):
        batch = torch.rand([self.batch_size, 2], device=self.device, dtype=torch.float32)
        targets = self.traced_image(batch)
        return batch, targets
    
    def _step_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def _set_model_requires_grad(self, On = False):
        if On:
            self.model.net.train()
        else:
            self.model.net.eval()
        for p in self.model.net.parameters():
            p.requires_grad = On

    def _set_quant_params_requires_grad(self, On = False):
        self.model._scale.requires_grad = On