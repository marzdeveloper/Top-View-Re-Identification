import os
import sys
import argparse
from shutil import copyfile
import subprocess


def move(result_file, src, dst, filter, result_path):

    dst = os.path.join(os.path.dirname(__file__), dst, result_file.split('.')[0][7:] + '_clean')
    os.mkdir(dst)
    src = os.path.join(os.path.dirname(__file__), src, result_file.split('.')[0][7:])
    with open(os.path.join(result_path,result_file), 'rb') as file_handler:
        rows = file_handler.readlines()[1:]
        for i, line in enumerate(rows):
            fields = line.decode().rstrip().split(';', 3)
            old_id = fields[0]
            new_id = fields[1]
            if not filter:
                new_id = old_id
            images = fields[3].split(';')
            if not os.path.exists(os.path.join(os.path.dirname(__file__), dst, new_id)):
                os.mkdir(os.path.join(os.path.dirname(__file__), dst, new_id))
            for x in images:
                copyfile(os.path.join(os.path.dirname(__file__), src, old_id, x),
                         os.path.join(os.path.dirname(__file__), dst, new_id, x))
                x = x.split('_')[0]
                x += '_depth.png'
                copyfile(os.path.join(os.path.dirname(__file__), src, old_id, x),
                         os.path.join(os.path.dirname(__file__), dst, new_id, x))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    p.add_argument('-result', dest='result', action='store', default='results', help='result path folder')
    p.add_argument('-src', dest='source', action='store', default='src', help='source path')
    p.add_argument('-dst', dest='destination', action='store', default='dataset', help='output path')
    p.add_argument('--filter', default=False, action='store_true')

    args = p.parse_args()
    if not os.path.exists(os.path.join(os.path.dirname(__file__), args.source)):
        print("Source path does not exist! Aborting")
        sys.exit(1)

    list_result_files = os.listdir(os.path.join(os.path.dirname(__file__), args.result))

    files_archive = []
    for result_f in list_result_files:
        print('Moving folders for file', result_f)
        move(result_f, args.source, args.destination, args.filter, args.result)
        files_archive.append(os.path.join(args.destination, result_f.split('.')[0][7:] + '_clean'))

    print('All files moved, now archiving')
    subprocess.call(['7z', 'a', 'archive.7z', '-v1900m'] + files_archive)


if __name__ == '__main__':
    main()
