import torch


def save_checkpoint(step,
                    model, 
                    optimizer, 
                    scheduler,
                    path):


    checkpoint = {
        'step': step,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler
    }


    torch.save(checkpoint, path)



def load_checkpoint(path):
    return torch.load(path)