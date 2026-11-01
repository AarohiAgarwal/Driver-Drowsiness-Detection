import os
import time
import ctypes
from pathlib import Path

def test_winmm(audio_file):
    print(f"Testing winmm on {audio_file}...")
    abs_path = str(Path(audio_file).resolve())
    winmm = ctypes.windll.winmm
    alias = "test_audio_alias"
    
    # Close any existing
    winmm.mciSendStringW(f"close {alias}", None, 0, None)
    
    ret_open = winmm.mciSendStringW(f'open "{abs_path}" type mpegvideo alias {alias}', None, 0, None)
    ret_play = winmm.mciSendStringW(f'play {alias}', None, 0, None)
    print(f"ret_open={ret_open}, ret_play={ret_play}")
    
    # Query length if supported
    buffer = ctypes.create_unicode_buffer(128)
    winmm.mciSendStringW(f"status {alias} length", buffer, 128, None)
    length_ms = buffer.value
    print(f"Audio length: {length_ms} ms")
    
    # Let it play briefly
    time.sleep(1.0)
    winmm.mciSendStringW(f"stop {alias}", None, 0, None)
    winmm.mciSendStringW(f"close {alias}", None, 0, None)
    print("Stopped and closed cleanly.")

if __name__ == "__main__":
    test_winmm("test_hi.mp3")
