import os
import zipfile
import subprocess
from tqdm import tqdm

# 다운로드할 데이터셋 경로 리스트
datasets = [
    'uwrfkaggler/ravdess-emotional-speech-audio',
    'ejlok1/cremad',
    'ejlok1/toronto-emotional-speech-set-tess',
    'barelydedicated/savee-database'
]

# 각 데이터셋을 순회하며 다운로드 및 압축 해제
for dataset_path in tqdm(datasets, desc="Downloading datasets"):
    # 데이터셋 다운로드 명령어
    download_command = ["kaggle", "datasets", "download", "-d", dataset_path]
    
    # subprocess.run()을 사용하여 명령어 실행
    try:
        print(f"Downloading {dataset_path}...")
        subprocess.run(download_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dataset_path}: {e}")
        continue
    
    # 다운로드된 파일명 (데이터셋 이름으로 zip 파일이 생성됨)
    zip_file_name = dataset_path.split('/')[-1] + '.zip'
    
    # 다운로드된 zip 파일의 압축을 해제
    if os.path.exists(zip_file_name):
        print(f"Unzipping {zip_file_name}...")
        
        # 압축 해제할 폴더 이름 생성
        destination_folder = zip_file_name.replace('.zip', '')
        
        # 폴더가 없으면 생성
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        # 압축을 새로운 폴더에 해제
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        
        # 압축 해제 후 zip 파일 삭제 (선택 사항)
        os.remove(zip_file_name)
        print(f"Successfully downloaded and unzipped {dataset_path} to {destination_folder}")
    else:
        print(f"Error: {zip_file_name} not found after download.")

print("\nAll datasets processed successfully!")