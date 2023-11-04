import argparse

from cond_BERT import load_condBERT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detoxify text.')
    parser.add_argument('-f', '--file', type=str, help='filename to be detoxified')
    parser.add_argument('-o', '--out', type=str, help='filename to be store detox, only in pair with --file')
    parser.add_argument('-s', '--string', type=str, help='string to be detoxified')
    parser.add_argument('-t', '--threshold', type=float, help='threshold value of CondBERT.mask()')
    parser.add_argument('-w', '--min_words', type=int, help='min_words value of CondBERT.mask()')
    parser.add_argument('-n', '--top_n', type=int, help='top_n value of CondBERT.translate()')

    args = parser.parse_args()
    if args.file:
        condBERT = load_condBERT()
        with open(args.file, 'r') as f:
            text = f.read()
        out = condBERT.detox(text, args.threshold, args.min_words, args.top_n)

        if args.out:
            out_path = args.out
        else:
            fname_split = args.file.split('.')
            out_path = fname_split[-2] + '-detox' + fname_split[-1]
        with open(out_path, 'w') as f:
            f.write(out)
    elif args.string:
        condBERT = load_condBERT()
        print(condBERT.detox(args.string, args.threshold, args.min_words, args.top_n))
    else:
        raise ValueError('Nothing to detoxify')
