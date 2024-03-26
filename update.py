import os
import requests
import hashlib

def download_file(url, destination):
    try:
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {destination}")
    except Exception as e:
        print(f"Error downloading {destination}: {e}")

def get_file_sha(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return hashlib.sha256(response.content).hexdigest()
        else:
            print(f"Failed to get SHA for {url}")
            return None
    except Exception as e:
        print(f"Error fetching SHA for {url}: {e}")
        return None

def update_file(file_info):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_path = os.path.join(script_dir, file_info['path'], file_info['name'])
    remote_sha = get_file_sha(file_info['download_url'])

    if remote_sha:
        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                local_sha = hashlib.sha256(f.read()).hexdigest()
                print(f'Skipping {local_path}, up to date.')
            if local_sha != remote_sha:
                download_file(file_info['download_url'], local_path)
        else:
            if not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path))
            download_file(file_info['download_url'], local_path)


def main():
    try:
        response = requests.get('https://api.github.com/repos/TheStingerX/Ilaria-RVC-Mainline/contents')
        if response.status_code == 200:
            files_info = response.json()
            for file_info in files_info:
                if file_info['type'] == 'file':
                    file_info['path'] = ''
                    update_file(file_info)
                elif file_info['type'] == 'dir':
                    folder_url = file_info['url']
                    folder_response = requests.get(folder_url)
                    if folder_response.status_code == 200:
                        folder_files_info = folder_response.json()
                        for folder_file_info in folder_files_info:
                            if folder_file_info['type'] == 'file':
                                folder_file_info['path'] = file_info['name']
                                update_file(folder_file_info)
        else:
            print("Failed to fetch SHAs from the Github API.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
