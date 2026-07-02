import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
# Explicit imports instead of wildcard '*' to avoid namespace pollution
from utils.utils import get_transform, ImageFolderDataset, adaptive_instance_normalization, calc_mean_std, gram_matrix
from utils.models import VGGEncoder, Decoder
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default='content_data',
                        help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default='style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default='vgg_normalised.pth',
                        help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment')
    
    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action='store_true',
                        help='Randomly crop images to final_size (default: resize only)')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5,
                        help='Style weight')
    parser.add_argument('--use_gram', action='store_true', default=False,
                        help='Use Gram matrix style loss (Gatys-style) instead of mean/std')
    parser.add_argument('--style_layers', type=str, default='all',
                        help='Layers for style loss: "all" (relu1-1 to relu4-1) or "high" (relu3-1, relu4-1 only)')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')
    

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    #Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')
    
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset,
                                    batch_size=args.batch_size,
                                    shuffle = True,
                                    pin_memory=True,
                                    drop_last=True)
    style_dataloader = DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    
    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))

    if len(content_dataloader) == 0 or len(style_dataloader) == 0:
        raise ValueError("Content or style dataset is empty. Please check the dataset paths!")

    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    # Per-iteration LR decay (original AdaIN paper implementation).
    # Old code stepped per epoch — with lr_decay=5e-5 and 1 epoch, LR barely changed.
    # Now we manually decay LR after every optimizer.step() using the iteration count.
    iteration = [0]  # Use list for closure mutability in inner function

    def _apply_lr_decay():
        """Decay LR: lr = initial_lr / (1 + lr_decay * iteration)"""
        iteration[0] += 1
        new_lr = args.lr / (1.0 + args.lr_decay * iteration[0])
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    if args.resume:
        if args.decoder_path and args.optimizer_path:
            # Load with weights_only=True for security + map_location for device compatibility
            decoder.load_state_dict(torch.load(args.decoder_path, map_location=device, weights_only=True))
            optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device, weights_only=True))
        else:
            print("[ERROR] --resume requires --decoder_path and --optimizer_path")
            return

    print('Training...')

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))

        running_loss = 0
        running_closs = 0
        running_sloss = 0
        num_batches = 0

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            with torch.no_grad():
                c_feats = encoder(content_batch)
                s_feats = encoder(style_batch)
                t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g = decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            if args.style_layers == 'high':
                layer_slice = slice(-2, None)  # only relu3-1, relu4-1
            else:
                layer_slice = slice(None)  # all layers

            if args.use_gram:
                for g_f, s_f in zip(g_feats[layer_slice], s_feats[layer_slice]):
                    g_gram = gram_matrix(g_f)
                    s_gram = gram_matrix(s_f)
                    loss_s += mse_loss(g_gram, s_gram)
            else:
                for g_f, s_f in zip(g_feats[layer_slice], s_feats[layer_slice]):
                    g_mean, g_std = calc_mean_std(g_f)
                    s_mean, s_std = calc_mean_std(s_f)
                    loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            
            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _apply_lr_decay()  # Decay LR after every iteration, not per epoch

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
            num_batches += 1
        
        # scheduler.step() removed — LR decay now happens per-iteration above

        # Guard against division by zero if dataset is empty
        if num_batches == 0:
            tqdm.write(f'[WARNING] Epoch {epoch+1}: No batches processed. Check that your dataset directories are not empty.')
            continue

        running_loss /= num_batches
        running_closs /= num_batches
        running_sloss /= num_batches

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)


    print("Training finished.")
    # Always save final model
    # NOTE: Save as 'decoder.pth' to match what app.py loads by default.
    # Also save optimizer for potential future resume.
    torch.save(decoder.state_dict(), save_dir / 'decoder_final.pth')  # Keep experiment copy
    torch.save(decoder.state_dict(), 'decoder.pth')  # Root-level copy for app.py
    torch.save(optimizer.state_dict(), save_dir / 'optimizer_final.pth')

if __name__ == '__main__':
    main()