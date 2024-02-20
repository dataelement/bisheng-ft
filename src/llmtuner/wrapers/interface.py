import argparse
import os
import traceback

from .trainval import trval_main


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(help='sub commands')

    trval_parser = sub_parsers.add_parser('train', help='train on train dataset')
    trval_parser.add_argument('--subcommand', default='train')
    trval_parser.add_argument(
        '--model_name_or_path', help='base model name', required=True, type=str)
    trval_parser.add_argument(
        '--dataset', help='train dataset path: data1,data2,data3', required=True, type=str)
    trval_parser.add_argument(
        '--each_max_samples', help='truncate the number of examples for each dataset: num1,num2,num3', required=True, type=str)
    trval_parser.add_argument(
        '--output_dir', help='save model path', required=True, type=str)
    trval_parser.add_argument(
        '--finetuning_type', help='finetuning type: full, freeze, lora', required=True, type=str)
    trval_parser.add_argument(
        '--gpus', help='gpus to use: 0,1,2,3', required=True, type=str)
    trval_parser.add_argument(
        '--val_ratio', help='val dataset ratio', default=0.1, type=float)
    trval_parser.add_argument(
        '--per_device_train_batch_size', help='batch size', default=1, type=int)
    trval_parser.add_argument(
        '--learning_rate', help='learning rate', default=0.00005, type=float)
    trval_parser.add_argument(
        '--num_train_epochs', help='train epochs', default=3, type=int)
    trval_parser.add_argument(
        '--max_seq_len', help='max seq len', default=8192, type=int)
    trval_parser.add_argument(
        '--cpu_load', help='if gpu memory is not enough params and optimizer in cpu', action='store_true')
    
    args = parser.parse_args()
    if args.subcommand == 'train':
        trval_main(args)


if __name__ == '__main__':
    main()
