"""
"""

from pathlib import Path
from torch.jit import trace, save

class CheckpointSaver:
    def __init__(self,
                 checkpoint_path,
                 frequency = -1,
                 benchmark = float('inf'),
                 prefix = 'mod'):

        self.checkpoint_path = Path(checkpoint_path)
        self.frequency = frequency
        self.benchmark = benchmark
        self.prefix = prefix

    def __call__(self, model, data, epoch, metric):

        traced_model = trace(model, data)

        name = self.checkpoint_path/f'{self.prefix}_last.pth'
        save(traced_model, name)

        if self.frequency > 0:
            if epoch % self.frequency == 0:
                name = self.checkpoint_path/f'{self.prefix}_{epoch}.pth'
                save(traced_model, name)

        if metric < self.benchmark:
            self.benchmark = metric
            name = self.checkpoint_path/f'{self.prefix}_best.pth'
            save(traced_model, name)
