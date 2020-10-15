
from trainer import LOTClassTrainer
import argparse


def main():
    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset_dir', default='datasets/agnews/',
                        help='dataset directory')
    parser.add_argument('--label_names_file', default='label_names.txt',
                        help='file containing label names (under dataset directory)')
    parser.add_argument('--train_file', default='train.txt',
                        help='unlabeled text corpus for training (under dataset directory); one document per line')
    parser.add_argument('--test_file', default=None,
                        help='test corpus to conduct model predictions (under dataset directory); one document per line')
    parser.add_argument('--test_label_file', default=None,
                        help='test corpus ground truth label; if provided, model will be evaluated during self-training')
    parser.add_argument('--final_model', default='final_model.pt',
                        help='the name of the final classification model to save to')
    parser.add_argument('--out_file', default='out.txt',
                        help='model predictions on the test corpus if provided')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='batch size per GPU for evaluation; bigger batch size makes training faster')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='batch size per GPU for training')
    parser.add_argument('--top_pred_num', type=int, default=50,
                        help='language model MLM top prediction cutoff')
    parser.add_argument('--category_vocab_size', type=int, default=100,
                        help='category vocabulary size for each class')
    parser.add_argument('--match_threshold', type=int, default=20,
                        help='category indicative words matching threshold')
    parser.add_argument('--max_len', type=int, default=512,
                        help='length that documents are padded/truncated to')
    parser.add_argument('--update_interval', type=int, default=50,
                        help='self training update interval; 50 is good in general')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='gradient accumulation steps during training')
    parser.add_argument('--mcp_epochs', type=int, default=3,
                        help='masked category prediction training epochs; 3-5 usually is good depending on dataset size (smaller dataset needs more epochs)')
    parser.add_argument('--self_train_epochs', type=float, default=1,
                        help='self training epochs; 1-5 usually is good depending on dataset size (smaller dataset needs more epochs)')
    parser.add_argument('--early_stop', action='store_true',
                        help='whether or not to enable early stop of self-training')
    parser.add_argument('--gpus', default=3, type=int,
                        help='number of gpus to use')
    parser.add_argument('--dist_port', type=int, default=12345,
                        help='distributed training port id; any number between 10000 and 20000 will work')

    args = parser.parse_args()
    print(args)
    trainer = LOTClassTrainer(args)
    # Construct category vocabulary
    trainer.category_vocabulary(top_pred_num=args.top_pred_num, category_vocab_size=args.category_vocab_size)
    # Training with masked category prediction
    trainer.mcp(top_pred_num=args.top_pred_num, match_threshold=args.match_threshold, epochs=args.mcp_epochs)
    # Self-training 
    trainer.self_train(epochs=args.self_train_epochs, loader_name=args.final_model)
    # Write test set results
    if args.test_file is not None:
        trainer.write_results(loader_name=args.final_model, out_file=args.out_file)


if __name__ == "__main__":
    main()
    