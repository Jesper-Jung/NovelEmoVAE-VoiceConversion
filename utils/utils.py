import os

def makedirs(path):
    try:
        original_umask = os.umask(0)
        os.makedirs(path, mode=0o755, exist_ok=False)
    finally:
        os.umask(original_umask)