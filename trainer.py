import torch
import torch.nn.functional as F
from tqdm import tqdm


l1_loss = torch.nn.L1Loss(reduction='sum')


def loss_function(recon_x, x, mu, logvar, args):
    h, w = args.h, args.w
    # reconstruction loss
    BCE = F.binary_cross_entropy(
        recon_x.view(-1, h * w), x.view(-1, h * w), reduction='sum')
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(dataloader, model, optimizer, args=None, logger=None):
    model.train()
    losses = l1s = 0.
    pbar = tqdm(total=len(dataloader))
    for i, (input_, target, _) in enumerate(dataloader):
        bs, c, h, w = target.size()
        output, loss, l1 = step(input_, target, model, args=args)
        losses += loss.item()
        l1s += l1.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.debug: break  # noqa

        pbar.update(1)
    pbar.close()

    return {
        'loss': losses / len(dataloader.dataset), 'L1': l1s / len(dataloader.dataset),
        'pred': output, 'true': target,
    }


def validate(dataloader, model, logger=None, args=None):
    model.eval()
    losses = l1s = 0.
    pbar = tqdm(total=(len(dataloader)))
    for i, (input_, target, _) in enumerate(dataloader):
        bs, c, h, w = target.size()
        with torch.no_grad():
            output, loss, l1 = step(input_, target, model, args=args)
        losses += loss.item()
        l1s += l1.item()

        if args.debug: break  # noqa

        pbar.update(1)
    pbar.close()

    return {
        'loss': losses / len(dataloader.dataset), 'L1': l1s / len(dataloader.dataset),
        'pred': output, 'true': target,
    }


def step(input_, target, model, args):
    input_, target = input_.to(args.device), target.to(args.device)
    output, mu, logvar = model(input_)
    output = F.interpolate(output, size=(args.h, args.w), mode=args.interpolation_mode)
    loss = loss_function(output, target, mu, logvar, args=args)
    l1 = l1_loss(output, target)
    return output, loss, l1
