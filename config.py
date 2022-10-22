import argparse

def get_train_config(args):
    model = args.model
    gpu_id = args.gpu_id
    scale = args.scale
    continue_train = args.continue_train

    if scale not in [2, 3, 4]:
        raise Exception('scale {} is not supported!'.format(scale))

    opt_parser = argparse.ArgumentParser(description='Training module')
    opt_parser.add_argument('--gpu_id', type=str, default=gpu_id)
    opt_parser.add_argument('--init_iters', type=int, default=0)
    opt_parser.add_argument('--num_iters', type=int, default=300e3)
    opt_parser.add_argument('--lr', type=float, default=1e-4)
    opt_parser.add_argument('--N_frames', type=int, default=5)
    opt_parser.add_argument('--batch_size', type=int, default=4)
    opt_parser.add_argument('--valid_batch_size', type=int, default=1)
    opt_parser.add_argument('--n_workers', type=int, default=4)
    opt_parser.add_argument('--LR_size', type=int, default=128)
    opt_parser.add_argument('--scale', type=int, default=args.scale)

    opt_parser.add_argument('--train_paths_LR_RAW', type=str, default='./datasets/Train/{0:d}X/lr_{1:d}_raw/'.format(scale, scale))
    opt_parser.add_argument('--train_paths_LR_RGB', type=str, default='./datasets/Train/{0:d}X/lr_{1:d}_rgb/'.format(scale, scale))
    opt_parser.add_argument('--train_paths_HR_RGB', type=str, default='./datasets/Train/{0:d}X/hr_{1:d}_rgb/'.format(scale, scale))

    opt_parser.add_argument('--test_paths_LR_RAW', type=str, default='./datasets/Test/{0:d}X/lr_{1:d}_raw/'.format(scale, scale))
    opt_parser.add_argument('--test_paths_LR_RGB', type=str, default='./datasets/Test/{0:d}X/lr_{1:d}_rgb/'.format(scale, scale))
    opt_parser.add_argument('--test_paths_HR_RGB', type=str, default='./datasets/Test/{0:d}X/hr_{1:d}_rgb/'.format(scale, scale))
    
    opt_parser.add_argument('--weight_savepath', type=str, default='./weight_checkpoints/')
    opt_parser.add_argument('--model', type=str, default=model)
    opt_parser.add_argument('--continue_train', type=bool, default=continue_train)
    opt = opt_parser.parse_args()

    return opt

def get_test_config(args):
    model = args.model
    gpu_id = args.gpu_id
    scale = args.scale
    save_image = args.save_image

    if scale not in [2, 3, 4]:
        raise Exception('scale {} is not supported!'.format(scale))

    opt_parser = argparse.ArgumentParser(description='Testing module')
    opt_parser.add_argument('--gpu_id', type=str, default=gpu_id)
    opt_parser.add_argument('--N_frames', type=int, default=5)
    opt_parser.add_argument('--batch_size', type=int, default=1)
    opt_parser.add_argument('--n_workers', type=int, default=4)
    opt_parser.add_argument('--scale', type=int, default=args.scale)

    opt_parser.add_argument('--test_paths_LR_RAW', type=str, default='./datasets/Test/{0:d}X/lr_{1:d}_raw/'.format(scale, scale))
    opt_parser.add_argument('--test_paths_LR_RGB', type=str, default='./datasets/Test/{0:d}X/lr_{1:d}_rgb/'.format(scale, scale))
    opt_parser.add_argument('--test_paths_HR_RGB', type=str, default='./datasets/Test/{0:d}X/hr_{1:d}_rgb/'.format(scale, scale))

    opt_parser.add_argument('--weight_savepath', type=str, default='./weight_checkpoints/', help='weight_savepath')
    opt_parser.add_argument('--model', type=str, default=model, help='base model')
    opt_parser.add_argument('--save_image', type=bool, default=save_image)
    opt = opt_parser.parse_args()

    return opt
