import argparse


def parse_args():
    """Parse input arguments.

    Returns:
        args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(prog='Deep Attention MIL',
                                     description='Trains a deep attention-based multiple instance learning system')

    # General
    parser.add_argument('-m', '--model', dest='model',
                        help='choice of model [string]',
                        default='butkej-attention', type=str)

    parser.add_argument('-e', '--epochs', dest='epochs',
                        help='amount of epochs to train for [integer]',
                        default=100, type=int)

    # Model-related settings
    parser.add_argument('-g', '--gpu', dest='multi_gpu',
                        help='choice of gpu amount [integer]',
                        default=0, type=int)

    parser.add_argument('-o', '--optimizer', dest='optimizer',
                        help='choice of optimizer [string]',
                        default='adam', type=str)

    # parser.add_argument('-d', '--dropout', dest='dropout',
    #                    help='dropout rate [float 0-1]',
    #                    default=0.5, type=float)

    parser.add_argument('--use_gated', dest='use_gated',
                        help='use gated Attention. is False by default [boolean]',
                        action='store_true')

    parser.add_argument('--no_bias', dest='use_bias',
                        help='use bias. is True by default [boolean]',
                        action='store_false')

    args = parser.parse_args()
    return args


def main():
    print("Thanks for running this function, but it actually does nothing. Have a nice day. =)")
    print("But, this function checks and validates your input args:")

    parse_args()


if __name__ == "__main__":
    main()
