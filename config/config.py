import argparse
import numpy as np

def init_args(return_parser=False): 
    parser = argparse.ArgumentParser(description="""Configure""")

    # basic configuration 
    parser.add_argument('--exp', type=str, default='test101',
                        help='checkpoint folder')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs to run (default: 90)')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--resume_optim', default=False, action='store_true')
    parser.add_argument('--save_step', default=1, type=int)
    parser.add_argument('--valid_step', default=1, type=int)
    

    # Dataloader parameter
    parser.add_argument('--max_sample', default=-1, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', default=24, type=int)

    # network parameters
    parser.add_argument('--setting', type=str, default='obstacle_mono', required=False)

    parser.add_argument('--backbone', type=str, default='vggish', required=False)
    parser.add_argument('--pretrained', default=False, action='store_true')

    parser.add_argument('--no_bn', default=False, action='store_true')
    # optimizer parameters
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--optim', type=str, default='Adam',
                        choices=['SGD', 'Adam'])
    parser.add_argument('--schedule', type=str, default='cos', choices=['none', 'cos', 'step'], required=False)
    # Loss parameters
    parser.add_argument('--loss', type=str, default='MAE', choices=['BCE', 'CE', 'MSE', 'MAE', 'Huber'], required=False)

    parser.add_argument('--aug_spec', default=False, action='store_true')
    parser.add_argument('--aug_wave', default=False, action='store_true')
    parser.add_argument('--aug_img', default=False, action='store_true')

    parser.add_argument('--valid_by_step', default=False, action='store_true')
    parser.add_argument('--min_distance', type=float, default=0.1, choices=[0.1, 0.2, 0.3, 0.4], required=False)
    parser.add_argument('--delta', type=float, default=1.0, required=False)
    
    parser.add_argument('--input', type=str, default='image', choices=['audio', 'image', 'both'], required=False)
    parser.add_argument('--freeze', default=False, action='store_true')

    parser.add_argument('--test_mode', default=False, action='store_true')
    parser.add_argument('--list_test', type=str, default=None, required=False)
    parser.add_argument('--load_model', type=str, default=None, required=False)
    parser.add_argument('--scale2meter', default=False, action='store_true')

    if return_parser:
        return parser

    # global args
    args = parser.parse_args()

    return args
