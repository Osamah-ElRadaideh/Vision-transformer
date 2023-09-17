import numpy as np
import torch
import torch.nn as nn
from model import VIT
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
from utils import AHE, collate
import lazy_dataset
from sacred import Experiment
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
ex = Experiment('vision_transformer',save_git_info=False)
sw = SummaryWriter()


@ex.config
def defaults():
    lr= 1e-3
    batch_size = 16
    n_classes = 10  #100 for cifar, 10 for AHE,  4 for BCCD, 67 for CVPR
    patch_size = 16
    input_size = 64
    num_epochs = 500
    steps_per_eval = 500
    load_ckpt = False
    use_fp16 = True

def load_img(example):
    img = cv2.imread(example['image_path'], 1)
    example['image'] = img.astype(np.float32)
    return example

def prepare_dataset(dataset,batch_size=16):
    if isinstance(dataset,list):
        dataset = lazy_dataset.new(dataset)
    
    dataset = dataset.map(load_img)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(collate)
    return dataset
@ex.automain
def main(lr, batch_size, input_size, patch_size, n_classes, load_ckpt, use_fp16, num_epochs, steps_per_eval):
    db = AHE()
    train_ds = db.get_dataset('training_set')
    val_ds = db.get_dataset('validation_set')
    train_ds = prepare_dataset(train_ds,batch_size)
    val_ds = prepare_dataset(val_ds,batch_size=1)

    #patch_dim can be calculated as patch_size^2 * num_channels (3 for RGB, 1 for greyscale)
    model = VIT(input_size = input_size, patch_size=patch_size, patch_dim = 768, n_classes=n_classes, num_layers=4).to(device)
    if load_ckpt:
        states = torch.load('ckpt_latest.pth')
        model.load_state_dict(states)
    optim = torch.optim.Adam(model.parameters(),lr=lr)

    scaler = torch.cuda.amp.GradScaler() 

    CE = nn.CrossEntropyLoss()
    steps = 0
    running_loss = 0
    min_acc = 1e7
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch in tqdm(train_ds):
            optim.zero_grad()
            # x, y = batch
            x = torch.tensor(np.array(batch['image']))
            y = torch.tensor(batch['target'])
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = model(x)
            loss = CE(output, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running_loss += loss.item()
        if steps % steps_per_eval == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0
            with torch.inference_mode():
                for batch in tqdm(val_ds):
                    # x, y = batch
                    x = torch.tensor(np.array(batch['image']))
                    y = torch.tensor(batch['target'])
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)
                    loss = CE(output, y)
                    valid_loss += loss.item()
                    if torch.argmax(output).item() == y:
                        accuracy +=1
            accuracy /= len(val_ds)
            print(f'validation accuracy at {steps}: {accuracy:.3f}')
            sw.add_scalar('validation/accuracy', accuracy, steps)
            sw.add_scalar('validation/loss', valid_loss / len(val_ds), steps)
            sw.add_scalar('training/loss', running_loss / (steps + 1), steps)
            model.train()
            print('model set back to training mode.')
            if accuracy < min_acc:
                min_acc = accuracy
                torch.save({'VIT': model.state_dict(),
                            'optimizer': optim.state_dict(),
                            'steps': steps,
                            'min_acc': min_acc,
                            }, 'ckpt_best_loss.pth')
            torch.save({'VIT': model.state_dict(),
                        'optimizer': optim.state_dict(),
                        'steps': steps,
                        'min_acc': min_acc,
                        }, 'ckpt_latest.pth')


