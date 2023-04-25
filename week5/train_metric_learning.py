import csv
import random
import os
import imageio
import glob
import umap
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytorch_metric_learning.utils.logging_presets as logging_presets
import matplotlib
from natsort import natsort_keygen
import cv2
from PIL import Image

from cycler import cycler
from pytorch_metric_learning import (
    testers, samplers, losses, distances, trainers, miners)
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from metric_learning.datasets.aic19 import create_dataloader
from metric_learning.models.resnet import ResNetWithEmbedder
from metric_learning.models.vgg import VGG19

import umap
from sklearn import manifold
from sklearn.decomposition import PCA


OUTPUT_PATH = './outputs_metric_learning'
EXPERIMENT_NAME = None
tensorboard_folder = ''


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, tasks b and c. Team 1'
    )

    # General configuration
    parser.add_argument('--output_path', type=str, default='./outputs_metric_learning',
                        help='Path to the output directory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the experiment.')
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='aic19',
                        help='Dataset to use. Options: aic19.')
    parser.add_argument('--dataset_path', type=str, default='./metric_learning_dataset/',
                        help='Path to the dataset.')
    # Model configuration
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    # Loss configuration
    parser.add_argument('--loss', type=str, default='ntxent',
                        help='Loss to use. Options: contrastive, ntxent, triplet.')
    parser.add_argument('--pos_margin', type=float, default=1.0,
                        help='Positive margin for contrastive loss. Also used for triplet loss.')
    parser.add_argument('--neg_margin', type=float, default=0.0,
                        help='Negative margin for contrastive loss.')
    parser.add_argument('--triplet_margin', type=float, default=0.05,
                        help='Margin for triplet loss.')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for contrastive loss.')
    parser.add_argument('--distance', type=str, default='cosine', # euclidean is default for triplet
                        help='Distance to use. Options: euclidean, cosine.')
    # Miner configuration
    # For contrastive loss, use PairMargin or BatchEasyHard
    # For triplet loss, use TripletMargin, MultiSimilarity or BatchHard
    parser.add_argument('--miner', type=str, default="PairMargin",
                        help='Miner to use. Options: BatchEasyHard, BatchHard, MultiSimilarity, PairMargin, TripletMargin.')
    parser.add_argument('--miner_pos_strategy', type=str, default='easy',
                        help='Positive strategy for the miner. Options: all, easy, semihard, hard.')
    parser.add_argument('--miner_neg_strategy', type=str, default='hard',
                        help='Negative strategy for the miner. Options: all, easy, semihard, hard.')
    parser.add_argument('--miner_epsilon', type=float, default=0.1,
                        help='Epsilon for the miner.')
    parser.add_argument('--miner_pos_margin', type=float, default=0.2,
                        help='Positive margin for the miner.')
    parser.add_argument('--miner_neg_margin', type=float, default=0.8,
                        help='Negative margin for the miner.')
    parser.add_argument('--miner_type_of_triplets', type=str, default='all',
                        help='Type of triplets for the miner. Options: all, easy, semihard, hard.')
    # Training configuration
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use. Options: adam, sgd.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--lr_trunk', type=float, default=1e-5,
                        help='Learning rate for the trunk.')
    parser.add_argument('--lr_embedder', type=float, default=1e-4,
                        help='Learning rate for the embedder.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay.')

    args = parser.parse_args()
    return args


class CustomVisualizer:
    def __init__(self):
        self.tsne = manifold.TSNE(
            n_components=2,
            perplexity=30,
            early_exaggeration=12,
            learning_rate="auto",
            n_iter=800,
            random_state=42,
            n_jobs=-1,
        )
        self.umap = umap.UMAP(random_state=42)
        self.pca = PCA(n_components=2, svd_solver='auto', random_state=42)

    def fit_transform(self, embeddings):
        return {
            'tsne': self.tsne.fit_transform(embeddings),
            'umap': self.umap.fit_transform(embeddings),
            'pca': self.pca.fit_transform(embeddings),
            'embed': embeddings,
        }


def create_GIF(plots_dir: str, max_epoch: int):
    natsort_key = natsort_keygen(key=lambda y: y.lower())

    for embed_type in ['pca', 'tsne', 'umap']:
        embed_plot_files = glob.glob(os.path.join(plots_dir, f"{embed_type}*.png"))
        embed_plot_files.sort(key=natsort_key)
        print("embed_plot_files ", embed_plot_files)

        plot_images = []
        for idx, filename in enumerate(embed_plot_files):
            frame = cv2.imread(filename)
            height, width, _ = frame.shape
            cv2.rectangle(frame, (0, height - 25), (int(idx * (width / len(embed_plot_files))), width), (0, 255, 0), -1)

            epoch = os.path.basename(filename).split("_")[-1].split(".")[0]
            cv2.putText(
                frame, f"{epoch}/{max_epoch}",
                (1, 1),  # bottom left
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1,  # scale
                (255,0,0),  # color
                1,  # thickness
                cv2.LINE_4,  # line type
            )

            plot_images.append(frame)

        print(f"Saving GIF for {embed_type.upper()}...")
        save_path = os.path.join(plots_dir, f"{embed_type}.gif")
        imageio.mimsave(save_path, plot_images, duration=0.6)
        print(f"GIF saved at {save_path}")


def generate_sprite_image(val_ds):
    old_transform = val_ds.transform
    val_ds.transform = None  # Do not apply transforms to images when saving them to sprite

    # Gather PIL images for sprite
    images_pil = []
    for img_pt, _ in val_ds:
        img_np = img_pt.numpy().transpose(1, 2, 0) * 255
        # Save PIL image for sprite
        img_pil = Image.fromarray(img_np.astype('uint8'), 'RGB').resize((100, 100))
        images_pil.append(img_pil)

    one_square_size = int(np.ceil(np.sqrt(len(val_ds))))
    master_width = 100 * one_square_size
    master_height = 100 * one_square_size
    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0,0,0,0)  # fully transparent
    )
    for count, image in enumerate(images_pil):
        div, mod = divmod(count, one_square_size)
        h_loc = 100 * div
        w_loc = 100 * mod
        spriteimage.paste(image, (w_loc, h_loc))

    global tensorboard_folder
    spriteimage.convert('RGB').save(f'{tensorboard_folder}/sprite.jpg', transparency=0)

    val_ds.transform = old_transform


def visualizer_hook(visualizer, embeddings, labels, split_name, keyname, epoch, *args):
    global OUTPUT_PATH, EXPERIMENT_NAME, tensorboard_folder
    print(f"Visualizing {len(labels)} embeddings for {split_name} split at epoch {epoch}...")

    plots_dir = os.path.join(OUTPUT_PATH, "embedding_plots", EXPERIMENT_NAME)

    for embed_type, embed in embeddings.items():
        if embed_type == 'embed':
            # store embeddings for tensorboard's projector
            with open(f'{tensorboard_folder}/feature_vecs.tsv', 'w') as fw:
                csv_writer = csv.writer(fw, delimiter='\t')
                csv_writer.writerows(embed)
            with open(f'{tensorboard_folder}/metadata.tsv', 'w') as file:
                for label in labels:
                    file.write(f'{label}\n')

            x = """embeddings {
  tensor_path: "feature_vecs.tsv"
  sprite {
    image_path: "sprite.jpg"
    single_image_dim: 100
    single_image_dim: 100
  }
}"""
            with open(f"{tensorboard_folder}/projector_config.pbtxt","w") as f:
                f.writelines(x)

            continue

        # plot embeddings
        logging.info(
            "{} plot for the {} split and label set {}".format(
                embed_type.upper(), split_name, keyname)
        )
        label_set = np.unique(labels)
        num_classes = len(label_set)
        fig = plt.figure(figsize=(15, 9))
        frame = plt.gca()
        frame.set_prop_cycle(
            cycler(
                "color", [plt.cm.nipy_spectral(i)
                        for i in np.linspace(0, 0.9, num_classes)]
            )
        )
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(embed[idx, 0],
                     embed[idx, 1],
                     ".", markersize=2)
        fig.legend(loc='outside upper right', markerscale=15)
        plt.title(f"{embed_type.upper()} - Epoch {epoch}")
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        fig.savefig(os.path.join(plots_dir, f"{embed_type}_{split_name}_{epoch}.png"))
        plt.close()


def main(args: argparse.Namespace):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    global OUTPUT_PATH, EXPERIMENT_NAME
    os.makedirs(args.output_path, exist_ok=True)
    OUTPUT_PATH = args.output_path
    experiment_name = f"{args.model}_{args.dataset}_loss_{args.loss}_miner_{args.miner}_distance_{args.distance}"
    EXPERIMENT_NAME = experiment_name
    model_folder = os.path.join(args.output_path, "models", experiment_name)
    os.makedirs(model_folder, exist_ok=True)
    logs_folder = os.path.join(args.output_path, "logs", experiment_name)
    os.makedirs(logs_folder, exist_ok=True)
    global tensorboard_folder
    tensorboard_folder = os.path.join(
        args.output_path, "tensorboard", experiment_name)
    os.makedirs(tensorboard_folder, exist_ok=True)
    plots_dir = os.path.join(args.output_path, "embedding_plots", experiment_name)
    os.makedirs(plots_dir, exist_ok=True)
    device = 'cuda'

    # Model loading
    if args.model.split("_")[0] == 'resnet':
        model = ResNetWithEmbedder(resnet=args.model.split("_")[1], embed_size=args.embedding_size)
    elif args.model == 'vgg':
        model = VGG19()
    else:
        raise ValueError(f'Unknown model: {args.model}')

    model.to(device)

    # Dataset loading
    if args.dataset == 'aic19':
        train_dataloader, val_dataloader = create_dataloader(
            args.dataset_path,
            args.batch_size,
        )
        train_ds = train_dataloader.dataset
        val_ds = val_dataloader.dataset
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # Sampler
    # https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/
    class_sampler = samplers.MPerClassSampler(
        labels=train_ds.targets,
        m=args.batch_size // 8, 
        batch_size=args.batch_size,
        length_before_new_iter=len(train_ds),
    )

    # Loss configuration
    # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/
    if args.distance == 'euclidean':
        distance = distances.LpDistance(p=2)
    elif args.distance == 'cosine':
        distance = distances.CosineSimilarity()
    else:
        raise ValueError(f'Unknown distance: {args.distance}')

    if args.loss == 'contrastive':
        criterion = losses.ContrastiveLoss(
            pos_margin=args.pos_margin,
            neg_margin=args.neg_margin,
            distance=distance
        )
    elif args.loss == 'ntxent':
        criterion = losses.NTXentLoss(
            temperature=args.temperature,
            distance=distance,
        )
    elif args.loss == 'triplet':
        criterion = losses.TripletMarginLoss(
            margin=args.triplet_margin,
            distance=distance,
        )
    else:
        raise ValueError(f'Unknown loss: {args.loss}')

    # Miner configuration
    # https://kevinmusgrave.github.io/pytorch-metric-learning/miners/
    if args.miner == "BatchEasyHard":
        miner = miners.BatchEasyHardMiner(
            pos_strategy=args.miner_pos_strategy,
            neg_strategy=args.miner_neg_strategy,
        )
    elif args.miner == "BatchHard":
        miner = miners.BatchHardMiner()
    elif args.miner == "MultiSimilarity":
        miner = miners.MultiSimilarityMiner(
            epsilon=args.miner_epsilon,
        )
    elif args.miner == "PairMargin":
        miner = miners.PairMarginMiner(
            pos_margin=args.miner_pos_margin,
            neg_margin=args.miner_neg_margin,
        )
    elif args.miner == "TripletMargin":
        miner = miners.TripletMarginMiner(
            margin=args.miner_pos_margin,
            type_of_triplets=args.miner_type_of_triplets,
        )
    else:
        miner = None

    # Optimizer configuration
    if args.optimizer == 'adam':
        trunk_optimizer = torch.optim.Adam(
            model.trunk.parameters(), lr=args.lr_trunk, weight_decay=args.weight_decay)
        embedder_optimizer = torch.optim.Adam(
            model.embedder.parameters(), lr=args.lr_embedder, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        trunk_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_trunk)
        embedder_optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr_embedder)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    # Hooks
    record_keeper, _, _ = logging_presets.get_record_keeper(
        logs_folder, tensorboard_folder)
    hooks = logging_presets.get_hook_container(record_keeper)

    # Create the tester
    # https://kevinmusgrave.github.io/pytorch-metric-learning/testers/
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=CustomVisualizer(),
        visualizer_hook=visualizer_hook,
        dataloader_num_workers=1,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester,
        {"val": val_ds},
        model_folder,
        test_interval=1,
        patience=1,
    )

    # Training
    # https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/
    trainer = trainers.MetricLossOnly(
        models={"trunk": model.trunk, "embedder": model.embedder},
        optimizers={"trunk_optimizer": trunk_optimizer,
                    "embedder_optimizer": embedder_optimizer},
        loss_funcs={"metric_loss": criterion},
        mining_funcs={"tuple_miner": miner} if miner else None,
        data_device=device,
        dataset=train_ds,
        batch_size=args.batch_size,
        sampler=class_sampler,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    trainer.train(num_epochs=args.epochs)

    # Save model
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join(
        model_folder, f'model_final.pth'))

    # Create GIF from embedding plots
    create_GIF(
        plots_dir=plots_dir,
        max_epoch=args.epochs,
    )

    generate_sprite_image(val_ds)


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)
