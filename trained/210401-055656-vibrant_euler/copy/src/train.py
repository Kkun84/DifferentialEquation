"""
Usage:
  train.py [--max_epoch=<int>] [--batch_size=<int>] [--alpha=<float>] [--lr=<float>] [--device=<str>] [--seed=<int>] [--noise_scale=<float>]

Options:
  -h --help              Show this screen.
  --max_epoch=<int>      Epoch num [default: 10000].
  --batch_size=<int>     Batch size [default: 256].
  --lr=<float>           Learning rate [default: 0.001].
  -a, --alpha=<float>    Alpha [default: 0.1].
  -d, --device=<str>     Use device [default: cpu].
  --seed=<int>           Random seed [default: 0].
  --noise_scale=<float>  Noise scale [default: 1].
"""
from datetime import datetime
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from docopt import docopt
import mong
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm as tqdm

from src.dataset import Dataset
from src.equation import differential_equation, expression
from src.model import Model
import sys


def train(
    model,
    optimizer,
    writer,
    max_epoch: int,
    batch_size: int,
    alpha: float,
    device,
    noise_scale: float,
):
    def add_log():
        x = torch.arange(test_dataset.min, test_dataset.max, 0.001)
        fig, ax_list = plt.subplots(2, 1, figsize=[8, 8])
        ax_list = ax_list.T.flatten()

        ax = ax_list[0]
        ax.set_title('Expression of the solution')
        ax.grid()
        ax.scatter(
            train_dataset.data['x'],
            train_dataset.data['y'],
            s=2,
            c='black',
            label='data',
        )
        ax.plot(x, expression(x), label='y')
        ax.plot(x, model(x.to(device)[:, None]).detach().cpu()[:, 0], label='y_hat')
        ax.legend()

        ax = ax_list[1]
        ax.set_title('Differential equation error')
        ax.grid()
        ax.plot(
            x,
            differential_equation(x.to(device)[:, None], expression)
            .detach()
            .cpu()[:, 0],
            label='y',
        )
        ax.plot(
            x,
            differential_equation(x.to(device)[:, None], model).detach().cpu()[:, 0],
            label='y_hat',
        )
        ax.legend()

        writer.add_figure('predicted', fig, total_iterate)

    train_dataset = Dataset(300, 0, 1, expression, noise_scale=noise_scale)
    dataset_for_generate = Dataset(0, -1, 9, expression)
    test_dataset = Dataset(0, -2, 10, expression)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(train_dataset.data['x'], train_dataset.data['y'])
    writer.add_figure('dataset', fig)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    total_iterate = 0
    for epoch in tqdm(range(max_epoch)):
        for iterate, (x, y) in enumerate(tqdm(dataloader, leave=False)):
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            mse_loss = F.mse_loss(y_hat, y)

            x = dataset_for_generate.generate_x(len(x)).to(device)
            y_hat = differential_equation(x, model)
            differential_equation_loss = F.mse_loss(y_hat, torch.zeros_like(y_hat))

            loss = alpha * mse_loss + (1 - alpha) * differential_equation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('count/epoch', epoch, total_iterate)
            writer.add_scalar('count/iterate', iterate, total_iterate)
            writer.add_scalar('count/total_iterate', total_iterate, total_iterate)

            writer.add_scalar('metrics/mse_loss', mse_loss, total_iterate)
            writer.add_scalar(
                'metrics/differential_equation_loss',
                differential_equation_loss,
                total_iterate,
            )
            writer.add_scalar('metrics/loss', loss, total_iterate)

            if (total_iterate % 1000 == 0) or ((epoch + 1) == max_epoch):
                add_log()
            total_iterate += 1
    add_log()
    return


def main():
    now = datetime.now().strftime('%y%m%d-%H%M%S')
    save_dir = Path('trained', now + '-' + mong.get_random_name())
    print(save_dir)
    save_dir.mkdir(parents=True)
    all_done = False
    try:
        shutil.copytree(Path('src'), (save_dir / 'copy' / 'src'))

        args = docopt(__doc__)
        print(args)

        max_epoch = int(args['--max_epoch'])
        batch_size = int(args['--batch_size'])
        lr = float(args['--lr'])
        alpha = float(args['--alpha'])
        device = torch.device(args['--device'])
        seed = int(args['--seed'])
        noise_scale = float(args['--noise_scale'])

        torch.manual_seed(seed)

        model = Model(1, 1, 256).to(device)
        model_summary = summary(model, verbose=0)
        print(model_summary)

        optimizer = torch.optim.Adam(model.parameters(), lr)

        writer = SummaryWriter(save_dir)
        writer.add_text(
            'model_summary',
            str(model_summary).replace('\n', '  \n'),
        )
        writer.add_text('command', ' '.join(sys.argv))

        train(
            model=model,
            optimizer=optimizer,
            writer=writer,
            max_epoch=max_epoch,
            batch_size=batch_size,
            alpha=alpha,
            device=device,
            noise_scale=noise_scale,
        )

        all_done = True
    except KeyboardInterrupt as e:
        print(e)
        shutil.move(save_dir, str(save_dir) + '-__interrupt__')
        save_dir = Path(str(save_dir) + '-__interrupt__')
        print(f'Rename output directory: "{save_dir}".')
        all_done = True
    finally:
        if not all_done:
            shutil.move(save_dir, str(save_dir) + '-__error__')
            save_dir = Path(str(save_dir) + '-__error__')
            print(f'Rename output directory: "{save_dir}".')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    print('Saved model.')
    print('Done.')


if __name__ == '__main__':
    main()
