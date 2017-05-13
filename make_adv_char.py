import argparse
import adv_char
from cnn import LeNet
from adv_char import AdversarialCharacter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make adversarial character')
    parser.add_argument('src_img_path', type=str,
                        help='source img path')
    parser.add_argument('src_alph', type=str,
                        help='source character')
    parser.add_argument('dst_alph', type=str,
                        help='destination character')
    parser.add_argument('-d', '--dst_path', dest='dst_path', type=str, default='output',
                        help='destination path')
    parser.add_argument('--cxpb', dest='cxpb', type=float, default=0.5,
                        help='crossover probability')
    parser.add_argument('--mutpb', dest='mutpb', type=float, default=0.2,
                        help='mutation probability')
    parser.add_argument('--ngen', dest='ngen', type=int, default=100,
                        help='num of generation')
    parser.add_argument('--npop', dest='npop', type=int, default=100,
                        help='num of population')
    parser.add_argument('--breakacc', dest='breakacc', type=float, default=0.99,
                        help='accuracy of break')
    args = parser.parse_args()

    lenet = LeNet(width=200, height=200, depth=1, classes=26, weight_path='./train_weights.hdf5')
    ac = AdversarialCharacter(src_img_path=args.src_img_path, src_alph=args.src_alph, 
                              dst_alph=args.dst_alph, dst_path=args.dst_path, 
                              cxpb=args.cxpb, mutpb=args.mutpb, ngen=args.ngen, 
                              npop=args.npop, breakacc=args.breakacc, model=lenet)
    ac.train()
    ac.make_animation()

