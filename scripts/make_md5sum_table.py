import argparse
import os
import os.path as osp
import hashlib
import isl_utils as islutil

@islutil.skip_if_processed()
def calculate_md5(file_path):
    try: 
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            hash_md5.update(f.read())
        return hash_md5.hexdigest()
    except Exception :
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create md5sum table videos')
    parser.add_argument('--video_dir', required=True, help='Directory containing videos')
    parser.add_argument('--table_path', required=True, help='Path of the table')
    args = parser.parse_args()

    video_files = list(islutil.allFiles(args.video_dir))
    md5sums = list(islutil.pmap(calculate_md5, video_files))

    # remove the files which couldn't be computed
    video_files = [v for v, md5 in zip(video_files, md5sums) if md5 is not None]
    md5sums = [md5 for md5 in md5sums if md5 is not None]

    with open(args.table_path, 'w+') as fp : 
        fp.write('path,hash\n')
        fp.write('\n'.join([f'{a},{b}' for a, b in zip(video_files, md5sums)]))
