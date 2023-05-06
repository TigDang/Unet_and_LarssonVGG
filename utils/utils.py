import torch

def save_model(opt, model, epoch='latest'):
    save_path = f'artifacts\\nets\\{opt.name}_{epoch}.pt'
    torch.save(model, save_path)


def load_model(opt, epoch='latest'):
    load_path = f'artifacts\\nets\\{opt.name}_{epoch}.pt'
    model = torch.load(load_path, map_location=opt.device)
    return model
