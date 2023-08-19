from abc import abstractproperty
import argparse
import os
import io
from collections import Counter


def load_source_labeled(out, source_domain, target_domain):
    with open(out + '%s-%s/source_labeled.txt' % (source_domain, target_domain), 'w')as fin:
        with open('./raw_data/%s_train.txt' % (source_domain), 'r')as fout:
            for line in fout:
                in_line = []
                label = ['O']
                line = line.strip().split("####")[1]
                words_tags = line.split()
                pre = 'O'
                for wt in words_tags:
                    if wt.count("=") >= 2:
                        _, _, tag = wt.split("=")
                        if (tag != 'O'):
                            if (pre != 'O'):
                                label.append('I' + tag[1:])
                            else:
                                label.append('B' + tag[1:])
                        else: label.append('O')
                        in_line.append("=")
                        pre = tag
                    else:
                        w, tag = wt.split("=")
                        if len(w)==0:
                            continue
                        if (tag != 'O'):
                            if (pre != 'O'):
                                label.append('I' + tag[1:])
                            else:
                                label.append('B' + tag[1:])
                        else: label.append('O')
                        in_line.append(w)
                        pre = tag

                fin.write('[source] ' + ' '.join(in_line) + '####' + ' '.join(label) + "\n")


def load_target_pseudo_labeled(rfile, wfile):
    with open(wfile, 'w')as fin:
        with open(rfile, 'r')as fout:
            for line in fout:
                words = line.strip().split("####")[0].split()
                tags = line.strip().split("####")[1].split()

                # remove '##' and convert 'BI' to 'T'
                new_words = []
                new_tags = []
                for j in range(len(words)):
                    w, t = words[j], tags[j]
                    t = t.replace("B",'T')
                    t = t.replace("I",'T')
                    if '##' in w:
                        if (new_words == []):
                            new_tags.append(t)
                            new_words.append(w[2:])
                        else:
                            new_words[-1] = new_words[-1] + w[2:]
                    else:
                        new_tags.append(t)
                        new_words.append(w)

                # convert 'T' to 'BI'
                pre_tag = 'O'
                text = []
                label = ['O']
                for i in range(len(new_words)):
                    w, t = new_words[i], new_tags[i]
                    if 'T' not in t:
                        tag = 'O'
                        cur_tag = 'O'
                    else:
                        cur_tag, sentiment = t.split('-')
                        if pre_tag == 'O':
                            tag = 'B-' + sentiment
                        else:
                            tag = 'I-' + sentiment
                    pre_tag = cur_tag

                    text.append(w)
                    label.append(tag)

                fin.write('[target] ' + ' '.join(text) + '####' + ' '.join(label) + "\n")


def load(inp_file):
    data = []
    with io.open(inp_file, encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            data.append(line)
    return data


def merge_data(new_file, ori_file0, ori_file1):
    ori_data0 = load(ori_file0)
    ori_data1 = load(ori_file1)

    merge_data = ori_data0 + ori_data1

    with io.open(new_file, "w", encoding="utf-8", errors="ignore") as fout:
        for data in merge_data:
            fout.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_domain", default='service',type=str,required=False)
    parser.add_argument("--target_domain", default='rest',type=str,required=False)
    parser.add_argument("--in_path", default='./pseudo_outputs/',type=str,required=False)
    parser.add_argument("--out_path", default='./process_data/',type=str,required=False)
    args = parser.parse_args()

    print(args.source_domain, "====================================>", args.target_domain)
    if not os.path.exists(args.out_path + "%s-%s" % (args.source_domain, args.target_domain)):
        os.makedirs(args.out_path + "%s-%s" % (args.source_domain, args.target_domain))

    load_target_pseudo_labeled(args.in_path + '%s-%s/pre.txt' % (args.source_domain, args.target_domain),
                               args.out_path + '%s-%s/target_pseudo.txt' % (args.source_domain, args.target_domain))

    load_source_labeled(args.out_path, args.source_domain, args.target_domain)

    merge_data(args.out_path + '%s-%s/final_train.txt' % (args.source_domain, args.target_domain),
               args.out_path + '%s-%s/target_pseudo.txt' % (args.source_domain, args.target_domain),
               args.out_path + '%s-%s/source_labeled.txt' % (args.source_domain, args.target_domain))

if __name__ == '__main__':
    main()