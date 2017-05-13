import argparse
from cnn import LeNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('src_train_path', type=str,
                        help='train imgs paths')
    parser.add_argument('-t', '--test', dest='src_test_path', type=str, default=None,
                        help='test imgs paths')
    parser.add_argument('--hdf5', dest='dst_hdf5_path', type=str, default='trained_weight.hdf5',
                        help='destination hdf5 path')
    args = parser.parse_args()
    lenet = LeNet(width=200, height=200, depth=1, classes=26)
    lenet.train(args.src_train_path, args.src_test_path, args.dst_hdf5_path)

