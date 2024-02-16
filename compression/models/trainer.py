import torch


def compute_l2_loss(output, targets):
    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    return relative_l2_error.mean()


def compute_loss(output, targets, rate, lmbda):
    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    return lmbda * rate + relative_l2_error.mean()


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
            {'params': model._scale, 'lr': 0.001 * self.lr},
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
        # FP training just for a few steps
        # prepare for quantization training
        self._set_model_requires_grad(True)
        self._set_quant_params_requires_grad(False)
        for _ in range(1200): 
            batch, targets = self._get_batch_and_targets()
            output = self.model(batch)
            rate = self.model.measure_laplace_rate()
            rate_per_batch = rate / self.batch_size
            # loss = compute_loss(output, targets, rate_per_batch, self.lmbda)
            loss = compute_l2_loss(output, targets)
            print(f"loss: {loss}, rate: {rate_per_batch}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Quantization training
        print("Start quantization training...")
        self._set_model_requires_grad(False)
        self._set_quant_params_requires_grad(False)
        self.model.save_full_precision_param()
        self.model.init_quant_params(self.model.get_full_precision_param())
        # self._set_model_requires_grad(True)
        # self._set_quant_params_requires_grad(True)
        # for _ in range(self.n_steps):
        #     batch, targets = self._get_batch_and_targets() 
        #     # quantize and dequantize
        #     output, rate, scale_grad = self.model(batch, quant=True)
        #     rate_per_batch = rate / self.batch_size
        #     # loss = compute_loss(output, targets, rate_per_batch, self.lmbda)
        #     loss = compute_l2_loss(output, targets)
        #     print(f"loss: {loss}, rate: {rate_per_batch}")
        #     # print(f"scale: {self.model._scale.flatten()}")
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     # accumulate gradients from rate_loss and l2_loss (after quantization)
        #     if self.model.net.params.grad is not None:
        #         self.model.net.params.grad = self.model.net_shadow.params.grad + self.model.net.params.grad
        #     else:
        #         self.model.net.params.grad = self.model.net_shadow.params.grad
        #     if self.model._scale.grad is not None:
        #         self.model._scale.grad = scale_grad + self.model._scale.grad
        #     else:
        #         self.model._scale.grad = scale_grad

        #     self.optimizer.step()

    def _get_batch_and_targets(self):
        batch = torch.rand([self.batch_size, 2], device=self.device, dtype=torch.float32)
        targets = self.traced_image(batch)
        return batch, targets
        
    def _set_model_requires_grad(self, On = False):
        if On:
            self.model.net.train()
        else:
            self.model.net.eval()
        for p in self.model.net.parameters():
            p.requires_grad = On

    def _set_quant_params_requires_grad(self, On = False):
        self.model._scale.requires_grad = On