import paramiko
import os
import asyncio
from constants import *
import shutil
import datetime

timenow = datetime.datetime.now()
session = datetime.datetime.strftime(timenow, "%y%m%d-%H%M%S") + "-train"
dataset_name = "dataset610v3"
train_script = "train_lora.py" 
# train_script = "good-train.py"

SLURM_SCRIPT = f"""#!/bin/bash

#SBATCH --job-name=MOGWAI
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=02:59:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --mem=40gb

source /etc/profile.d/modules.sh
module load nvidia/cuda-12.4
source /home/s2425823/test8/bin/activate

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128

export CUDA_HOME=/deepstore/software/nvidia/cuda-12.4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 {train_script}"""
    
async def create_ssh_client():
    """Создание SSH-клиента (для отправки команд на сервере)"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USERNAME, password=PASSWORD)
    return client

async def create_sftp_client():
    """Создание SFTP-клиента (для загрузки и скачивания файлов)"""
    transport = paramiko.Transport((HOST, 22))
    transport.connect(username = USERNAME, password = PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp

async def upload_files(sftp: paramiko.sftp):
    """Загрузка файлов на кластер."""

    # Загрузка датасета:
    # for i in range(1,30):
    #     sftp.put(f"{dataset_name}/{i}.mp4", f"{REMOTE_DIR}/{dataset_name}/{i}.mp4")
    # sftp.put(f"{dataset_name}/train.json", f"{REMOTE_DIR}/{dataset_name}/train.json")

    # Загрузка train скрипта
    sftp.put(train_script, f"{REMOTE_DIR}/sessions/{session}/{train_script}")

    # Загрузка bash скрипта
    sftp.put(f"sessions/{session}/submit.sh", f"{REMOTE_DIR}/sessions/{session}/submit.sh")

async def submit_job(ssh, session):
    """Запуск задания через sbatch и проверка файла."""
    command = f"cd {REMOTE_DIR}/sessions/{session} && sbatch submit.sh"
    stdin, stdout, stderr = ssh.exec_command(command)
    error_output = stderr.read().decode()
    if error_output:
        print(f"Ошибка: {error_output}")
        return None

    job_id = stdout.read().decode().strip().split()[-1]  # Получение ID задания
    print(f"Задание отправлено. ID: {job_id}")
    return job_id

async def monitor_job(ssh, job_id):
    """Мониторинг статуса задания."""
    score = 0
    while True:
        stdin, stdout, stderr = ssh.exec_command(f"squeue -j {job_id}")
        status = stdout.read().decode()
        if job_id not in status:
            print("Задание завершено.")
            break
        print("Задание в очереди или выполняется...", job_id, "Прошло времени:", score, "sec")
        await asyncio.sleep(60)
        score += 60

async def download_results(sftp, session, job_id):
    """Скачивание результатов с кластера."""
    local_dir = f"sessions/{session}"  # Локальная директория для сохранения
    remote_files = [
        f"{REMOTE_DIR}/sessions/{session}/output.log",
        f"{REMOTE_DIR}/sessions/{session}/error.log",
        f"{REMOTE_DIR}/sessions/{session}/finish.txt",
        f"{REMOTE_DIR}/sessions/{session}/{train_script}",     
    ]

    # Создать локальную директорию, если её нет
    os.makedirs(local_dir, exist_ok=True)

    for remote_file in remote_files:
        try:
            # Скачать файл
            local_path = os.path.join(local_dir, os.path.basename(remote_file))
            sftp.get(remote_file, local_path)
            print(f"Файл {remote_file} скачан в {local_path}")
        except BaseException as err:
            print(f"Ошибка при скачивании {remote_file}: {str(err)}")

# Основной процесс
async def process():
    # ssh
    ssh = await create_ssh_client()
    # sftp
    sftp = await create_sftp_client()

    # Создание папок на сервере, если их нет
    folders = [
        f"sessions/{session}"
    ]
    for path in folders:
        try:
            sftp.chdir(path)  # Проверка есть ли путь 
        except IOError as err:
            sftp.mkdir(path)  # Если нет, то создается папка session
            sftp.chdir(path)

    # создать на пк папку sessions/session если ее нет
    os.makedirs(f"sessions/{session}", exist_ok=True)

    with open(f"sessions/{session}/submit.sh", "wb") as f:
        f.write(bytes(SLURM_SCRIPT, "utf-8"))

    # Загрузка файлов на кластер
    await upload_files(sftp)

    # Отправка задания
    job_id = await submit_job(ssh, session)
    if not job_id: return

    # Мониторинг статуса
    await monitor_job(ssh, job_id)
    
    # Скачивание результатов
    await download_results(sftp, session, job_id)

    # Закрытие подключений
    sftp.close()
    ssh.close()

    print("END", session)

asyncio.run(process())