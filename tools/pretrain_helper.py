import os
import json
import requests

# fetch_pretrains returns an array with the names
# download_pretrain downloads the pretrain with the name

PRETRAINS_URL = "https://gist.githubusercontent.com/miqu-s/35042d5eaaebe311ecc2530040434192/raw/687ea4c2158aa20a8ba52aa2d16f12578ec4197b/pretrains"

def fetch_pretrains():
    try:
        response = requests.get(PRETRAINS_URL)
        if response.status_code == 200:
            pretrains_data = json.loads(response.text)
            return list(pretrains_data.keys())
        else:
            print("Failed to fetch pretrains data. Status code:", response.status_code)
            return []
    except Exception as e:
        print("An error occurred while fetching pretrains data:", e)
        return []

def download_pretrain(pretrain_name):
    try:
        response = requests.get(PRETRAINS_URL)
        if response.status_code == 200:
            pretrains_data = json.loads(response.text)
            if pretrain_name in pretrains_data:
                pretrain_urls = pretrains_data[pretrain_name]
                total_files = len(pretrain_urls)
                print(f"Downloading {total_files} files for {pretrain_name}:")
                for index, (key, url) in enumerate(pretrain_urls.items(), 1):
                    filename = f"{pretrain_name}_{key}.pth"
                    download_file(url, filename, index, total_files)
            else:
                print("Pretrain not found.")
        else:
            print("Failed to fetch pretrains data. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred while downloading pretrain:", e)

def download_file(url, filename, current_file, total_files):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            directory = "assets/pretrains_v2"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'wb') as f:
                total_length = int(response.headers.get('content-length'))
                print(f"Downloading {filename} [{current_file}/{total_files}]")
                dl = 0
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    print("\r[%s%s]" % ('=' * done, ' ' * (50 - done)), end='', flush=True)
            print(f"\rDownloaded {filename} successfully.")
        else:
            print(f"Failed to download {filename}. Status code:", response.status_code)
    except Exception as e:
        print(f"An error occurred while downloading {filename}:", e)