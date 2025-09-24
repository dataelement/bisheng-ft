import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(help='sub commands')

    trval_parser = sub_parsers.add_parser('train', help='train on train dataset')
    trval_parser.add_argument('--subcommand', default='train', help=argparse.SUPPRESS)
    trval_parser.add_argument('--model_name_or_path',
                              help='base model name',
                              required=True,
                              type=str)
    trval_parser.add_argument('--model_template', help='model template', required=True, type=str)
    trval_parser.add_argument('--dataset',
                              help='train dataset path: data1,data2,data3',
                              required=True,
                              type=str)
    trval_parser.add_argument(
        '--each_max_samples',
        help='truncate the number of examples for each dataset: num1,num2,num3',
        required=True,
        type=str)
    trval_parser.add_argument('--output_dir', help='save model path', required=True, type=str)
    trval_parser.add_argument('--finetuning_type',
                              choices=["lora", "full", "freeze"],
                              help='微调方式: lora(低秩适应), full(全参数微调), freeze(冻结部分参数)',
                              required=True,
                              type=str)
    trval_parser.add_argument('--gpus', help='gpus to use: 0,1,2,3', default='', type=str)
    trval_parser.add_argument('--val_ratio', help='val dataset ratio', default=0.1, type=float)
    trval_parser.add_argument('--per_device_train_batch_size',
                              help='batch size',
                              default=2,
                              type=int)
    trval_parser.add_argument('--learning_rate', help='learning rate', default=0.00005, type=float)
    trval_parser.add_argument('--num_train_epochs', help='train epochs', default=3, type=int)
    trval_parser.add_argument('--max_seq_len', help='max seq len', default=8192, type=int)
    trval_parser.add_argument('--cpu_load',
                              help='if gpu memory is not enough params and optimizer in cpu',
                              action='store_true')

    args = parser.parse_args()

    if not hasattr(args, 'subcommand'):
        parser.print_help()
        exit(1)

    if hasattr(args, 'gpus') and args.gpus != '':

        system_env = os.environ.get('SYSTEM_ENV', 'CUDA').upper()
        if system_env not in ['CUDA', 'ROCM', 'NPU']:
            raise ValueError(f'unknown system env {system_env}')
        if system_env == 'NPU':
            os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.gpus
        elif system_env == 'CUDA':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        elif system_env == 'ROCM':
            os.environ["HIP_VISIBLE_DEVICES"] = args.gpus

    if args.subcommand == 'train':
        from llmtuner.wrapers.trainval import trval_main
        trval_main(args)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    main()
