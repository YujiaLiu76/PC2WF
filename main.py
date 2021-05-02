import argparse
from train_end2end import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training PC2WF.')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='data path')
    parser.add_argument('-p', '--patch_size', type=int, default=50, required=True, help='patch size, e.g., 50.')
    parser.add_argument('-b', '--mini_batch', type=int, default=512, required=False, help='batch size for training patchNet, vertexNet, and lineNet.')
    parser.add_argument('-nt', '--nms_th', type=float, default=0.01, required=True, help='NMS threshold for filtering redundant vertices, used after vertexNet and before lineNet.')
    parser.add_argument('-lpt', '--line_positive_th', type=float, default=0.01, required=True, help='Threshold for positive line endpoints.')
    parser.add_argument('-lnt', '--line_negative_th', type=float, default=0.05, required=True, help='Threshold for negative line endpoints.')
    parser.add_argument('-lwP', '--loss_weight_patch', type=float, default=1.0, required=False, help='loss weight of patchNet.')
    parser.add_argument('-lwV', '--loss_weight_vertex', type=float, default=50.0, required=False, help='loss weight of vertexNet.')
    parser.add_argument('-lwL', '--loss_weight_line', type=float, default=1.0, required=False, help='loss weight of lineNet.')
    parser.add_argument('-s', '--sigma', type=float, default=0.01, required=True, help='sigma of noise.')
    parser.add_argument('-c', '--clip', type=float, default=0.01, required=True, help='clip of noise.')
    args = parser.parse_args()

    loss_weight = [args.loss_weight_patch, args.loss_weight_vertex, args.loss_weight_line]
    train(args.data_path, patch_size=args.patch_size, mini_batch=args.mini_batch, nms_th=args.nms_th, line_positive_th=args.line_positive_th, line_negative_th=args.line_negative_th, loss_weight=loss_weight, sigma=args.sigma, clip=args.clip)

