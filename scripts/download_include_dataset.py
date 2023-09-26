import argparse
import os
import urllib.request
import zipfile
from isl_utils import skip_if_processed, pmap

output_dir = None

@skip_if_processed()
def download_and_extract_zip(url):
    global output_dir
    filename = url.split('/')[-1]
    
    # Download ZIP file
    zip_path = os.path.join(output_dir, filename)
    urllib.request.urlretrieve(url, zip_path)
    
    # Unzip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Remove zip
    os.remove(zip_path)

def download_and_unzip(url_list_file):
    global output_dir
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Read the URLs from the file and strip them
    with open(url_list_file, 'r') as f:
        urls = [_.strip() for _ in f.readlines()]

    list(pmap(download_and_extract_zip, urls))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the INCLUDE dataset')
    parser.add_argument('--url_list_file', required=True, help='Path to the file containing URLs.')
    parser.add_argument('--output_dir', required=True, help='Directory to store the downloaded and unzipped files.')
    
    args = parser.parse_args()
    output_dir = args.output_dir
    download_and_unzip(args.url_list_file)

