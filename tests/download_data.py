import os
import requests
import argparse
import yaml
import shutil
from zipfile import ZipFile
import gzip
from tqdm import tqdm
from git import Repo, RemoteProgress

class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm(unit='objects', leave=True)

    def update(self, op_code, cur_count, max_count=None, message=''):
        if max_count:
            self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.set_description(f'{message}')
        self.pbar.refresh()

    def __del__(self):
        self.pbar.close()

def load_datasets(yaml_file):
    """
    Load datasets from a YAML file.
    
    Args:
        yaml_file (str): Path to the YAML file.
    
    Returns:
        dict: A dictionary containing the root folder and a list of datasets.
    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

def download_file(url, dest_folder, is_github=False, progress_bar=None):
    """
    Downloads a file from the given URL to the specified destination folder.
    
    Args:
        url (str): The URL of the file to download.
        dest_folder (str): The directory where the file should be saved.
        is_github (bool): Indicates if the file is from GitHub.
        progress_bar (tqdm): A tqdm progress bar instance to update.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    filename = url.split('/')[-1].split('?')[0]
    file_path = os.path.join(dest_folder, filename)
    
    # Check if file already exists
    if os.path.exists(file_path):
        if progress_bar:
            progress_bar.update(1)
        return file_path
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            block_size = 1024
            if is_github:
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        f.write(chunk)
                if progress_bar:
                    progress_bar.update(1)
            else:
                total_size = int(r.headers.get('content-length', 0))
                t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        t.update(len(chunk))
                        f.write(chunk)
                t.close()
                downloaded_size = os.path.getsize(file_path)
                if total_size != 0 and downloaded_size != total_size:
                    print(f"WARNING: {filename} download incomplete. Downloaded {downloaded_size} bytes; expected {total_size} bytes")
            return file_path
    except requests.RequestException as e:
        print(f'Failed to download {filename} from {url}: {e}')
        return None

def unzip_file(zip_path, dest_folder):
    """
    Unzips a ZIP file to the specified destination folder.
    
    Args:
        zip_path (str): The path to the ZIP file.
        dest_folder (str): The directory where the contents should be extracted.
    """
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    print(f'Extracted {zip_path} to {dest_folder}')

def ungzip_file(gz_path, dest_folder):
    """
    Unzips a GZ file to the specified destination folder and moves the extracted contents
    to the destination folder.
    
    Args:
        gz_path (str): The path to the GZ file.
        dest_folder (str): The directory where the contents should be extracted and moved.
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    # Create the path for the extracted file
    filename = os.path.basename(gz_path).replace('.gz', '')
    file_path = os.path.join(dest_folder, filename+".log")
    # Extract the GZ file
    with gzip.open(gz_path, 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f'Extracted {gz_path} to {file_path}')
    
    # If the extracted file is a directory, move its contents to the destination folder
    if os.path.isdir(file_path):
        for item in os.listdir(file_path):
            item_path = os.path.join(file_path, item)
            shutil.move(item_path, dest_folder)
        # Remove the now-empty directory
        os.rmdir(file_path)
        print(f'Moved contents of {file_path} to {dest_folder}')
    else:
        print(f'{file_path} is not a directory, no additional moving needed.')

def un7z_file(sevenz_path, dest_folder):
    import py7zr
    """
    Extracts a 7z file to the specified destination folder.
    
    Args:
        sevenz_path (str): The path to the 7z file.
        dest_folder (str): The directory where the contents should be extracted.
    """
    with py7zr.SevenZipFile(sevenz_path, mode='r') as z:
        z.extractall(path=dest_folder)
    print(f'Extracted {sevenz_path} to {dest_folder}')

def clone_github_repo(repo_url, dest_folder):
    """
    Clones a GitHub repository to the specified destination folder.
    
    Args:
        repo_url (str): The URL of the GitHub repository.
        dest_folder (str): The directory where the repository should be cloned.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    print(f'Cloning repository {repo_url} into {dest_folder}')
    progress = CloneProgress()
    Repo.clone_from(repo_url, dest_folder, progress=progress)

def move_github_folder(repo_folder, source_path, dest_folder):
    """
    Moves a folder from the cloned GitHub repository to the specified destination folder.
    
    Args:
        repo_folder (str): The path to the cloned GitHub repository.
        source_path (str): The path within the repository to the folder to move.
        dest_folder (str): The directory where the folder should be moved.
    """
    source_folder = os.path.join(repo_folder, source_path)
    if os.path.exists(source_folder):
        print(f'Moving folder {source_folder} to {dest_folder}')
        shutil.move(source_folder, dest_folder)
    else:
        print(f'Folder {source_folder} does not exist in the repository.')

def transform_github_url(url):
    """
    Transforms a GitHub URL to the corresponding repository URL and folder path.
    
    Args:
        url (str): The GitHub URL to transform.
    
    Returns:
        tuple: A tuple containing the repository URL and the folder path.
    """
    repo_path = url.split('github.com/')[-1].replace('tree/main/', '')
    repo_url = 'https://github.com/' + '/'.join(repo_path.split('/')[:2]) + '.git'
    folder_path = '/'.join(repo_path.split('/')[2:])
    return repo_url, folder_path

def download_all_parts(name, urls, dest_folder):
    """
    Downloads all parts of a dataset from multiple URLs.
    
    Args:
        name (str): The name of the dataset.
        urls (list): A list of URLs to download parts from.
        dest_folder (str): The destination folder for the dataset.
    """
    cloned_repos = {}
    for url in urls:
        if url is None:
            print(f'No valid URL provided for {name}. Skipping download.')
            continue

        # If the github link already points to a single file, we don't need a temp folder
        file_extensions = ['.zip', '.gz', '.7z', '.tar', '.rar', '.csv', '.txt']
        if 'github.com' in url and not any(url.lower().endswith(ext) for ext in file_extensions):
            try:
                repo_url, folder_path = transform_github_url(url)
                last_folder_name = folder_path.split('/')[-1]
                full_dest_folder = os.path.join(dest_folder, last_folder_name)
                temp_repo_folder = os.path.join(dest_folder, 'temp_repo')
                
                # Clone the repository if not already cloned
                if repo_url not in cloned_repos:
                    cloned_repos[repo_url] = temp_repo_folder
                    clone_github_repo(repo_url, temp_repo_folder)
                
                # Move the specific folder to the destination
                move_github_folder(temp_repo_folder, folder_path, full_dest_folder)
            except Exception as e:
                print(f'Failed to download GitHub folder {url}: {e}')
        else:
            print(f'Downloading {url} into {dest_folder}')
            file_path = download_file(url, dest_folder)
            if file_path:
                if file_path.endswith('.zip'):
                    unzip_file(file_path, dest_folder)
                elif file_path.endswith('.gz'):
                    ungzip_file(file_path, dest_folder)
                elif file_path.endswith('.7z'):
                    un7z_file(file_path, dest_folder)
                os.remove(file_path)  # Remove the zip or gz file after extraction
    
    # Clean up cloned repositories
    for repo_url, repo_folder in cloned_repos.items():
        shutil.rmtree(repo_folder)

def main(dest_base_folder, yaml_file):
    """
    Downloads and unzips all datasets to the specified base folder.
    
    Args:
        dest_base_folder (str): The base folder where datasets should be downloaded.
        yaml_file (str): Path to the YAML file containing dataset information.
    """
    data = load_datasets(yaml_file)
    root_folder = dest_base_folder if dest_base_folder else os.path.expanduser(data['root_folder'])
    
    for dataset in data['datasets']:
        name = dataset['name']
        skip_download = not dataset.get('download', True)
        if skip_download:
            print(f'Skipping download for {name}.')
            continue
        urls = dataset.get('urls', [dataset.get('url')])
        if not any(urls):
            print(f'No URLs provided for {name}. Skipping download.')
            continue
        dataset_folder = os.path.join(root_folder, name)
        
        # Check if the dataset folder already exists
        if os.path.exists(dataset_folder):
            print(f'Folder {dataset_folder} already exists. Skipping download for {name}.')
            continue
        
        download_all_parts(name, urls, dataset_folder)

#if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Download datasets to a specified location.')
parser.add_argument('--location', type=str, help='The base folder where datasets should be downloaded. This overrides the location in the YAML file.')
parser.add_argument('--config', type=str, default='datasets.yml', help='Path to the YAML file containing dataset information. Default is datasets.yml.')
args = parser.parse_args()

main(args.location, args.config)
