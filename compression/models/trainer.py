import torch


def compute_l2_loss(output, targets):
    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    return relative_l2_error.mean()


def compute_loss(output, targets, rate_bpp, lmbda):
    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    return lmbda * rate_bpp + relative_l2_error.mean()


def set_requires_grad(module, On: bool=True):
    for param in module.parameters():
        param.requires_grad = On

class TrainerPhase():
    def __init__(self, 
                 start_temperature_softround, 
                 end_temperature_softround, 
                 start_kumaraswamy, 
                 end_kumaraswamy, 
                 max_step=100):
        self.start_temperature_softround = start_temperature_softround
        self.end_temperature_softround = end_temperature_softround  
        self.start_kumaraswamy = start_kumaraswamy
        self.end_kumaraswamy = end_kumaraswamy
        self.max_step = max_step


class Trainer():
    def __init__(self, model, image, **args):
        # some args
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.n_steps = args['n_steps']
        self.device = args['device']
        self.n_pixels = args['n_pixels']
        self.lmbda = args['lmbda']
        self.phase_one = TrainerPhase(start_temperature_softround=0.3,
                                      end_temperature_softround=0.1,
                                      start_kumaraswamy=2.0,
                                      end_kumaraswamy=1.0,
                                      max_step=self.n_steps)
        self.phase_two = TrainerPhase(start_temperature_softround=1e-4,
                                      end_temperature_softround=1e-4,
                                      start_kumaraswamy=1.0,
                                      end_kumaraswamy=1.0,
                                      max_step=self.n_steps)
        
        self.model = model

        parameters = [
            {'params': model.grid.parameters(), 'lr': self.lr},
            {'params': model.net.parameters(), 'lr': self.lr},
            {'params': model.arm.parameters(), 'lr': self.lr},
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
        # Phase 1: Trainingg quantized grid with noise_quantizer and arm, net
        # print("Phase 1: Trainingg quantized grid with noise_quantizer and arm, net...")
        # for i in range(self.n_steps): 
        #     self.schedule_quantizer(i, self.phase_one)
        #     self._one_training_step(STE=False)

        # Phase 2: Training quantized grid with ste_quantizer and arm, net
        print("Phase 2: Training quantized grid with ste_quantizer and arm, net...")
        self.schedule_quantizer(0, self.phase_two)
        for _ in range(self.n_steps): 
            self._one_training_step(STE=True)

        # # Phase 3: Quantized everything and retune grid params
        # print("Phase 3: Quantized everything and retune grid params...")
        # self.model.quantize_model()
        # set_requires_grad(self.model.grid, True)
        # set_requires_grad(self.model.net, False)
        # set_requires_grad(self.model.arm, False)
        # for _ in range(self.n_steps): 
        #     self._one_training_step(STE=True)
    
    def _one_training_step(self, STE=False):
        batch = torch.rand([self.batch_size, 2], device=self.device, dtype=torch.float32)
        targets = self.traced_image(batch)
        output, rate = self.model(batch, training=True, STE=STE)
        rate_bpp = rate / self.n_pixels
        # loss = compute_loss(output, targets, rate_bpp, self.lmbda)
        loss = compute_l2_loss(output, targets)
        print(f"loss: {loss}, rate_bpp: {rate_bpp}")
        self.optimizer.zero_grad()
        loss.backward()
        print(f"grid grad: {self.model.grid.params.grad.amin()}, {self.model.grid.params.grad.amax()}")

        loss = self.lmbda * rate_bpp
        loss.backward()
        # self.model.grid.params.grad += self.model.grid_shadow.grad
        self.optimizer.step()

    def schedule_quantizer(self, step, trainer_phase):
        # Custom scheduling function for the soft rounding temperature and the kumaraswamy param
        def linear_schedule(initial_value, final_value, cur_step, max_step):
            return cur_step * (final_value - initial_value) / max_step + initial_value

        # Initialize soft rounding temperature and kumaraswamy parameter
        cur_tmp = linear_schedule(
            trainer_phase.start_temperature_softround,
            trainer_phase.end_temperature_softround,
            step,
            trainer_phase.max_step
        )
        kumaraswamy_param = linear_schedule(
            trainer_phase.start_kumaraswamy,
            trainer_phase.end_kumaraswamy,
            step,
            trainer_phase.max_step
        )

        self.model.noise_quantizer.soft_round_temperature = cur_tmp
        self.model.ste_quantizer.soft_round_temperature = cur_tmp
        self.model.noise_quantizer.kumaraswamy_param = kumaraswamy_param