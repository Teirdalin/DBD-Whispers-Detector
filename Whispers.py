import cv2
import numpy as np
import pyautogui
import pygame
import time
import os
import sys

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, "resources", relative_path)

pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()

sound_on = pygame.mixer.Sound(resource_path("whispers_on.wav"))
sound_off = pygame.mixer.Sound(resource_path("whispers_off.wav"))

template_paths = [
    "whispers_yellow_face.jpg",
    "whispers_green_face.jpg",
    "whispers_purple_face.jpg"
]

spinner = ["Whispers Detector Running.  ", "Whispers Detector Running.. ", "Whispers Detector Running..."]

templates = []
for filename in template_paths:
    path = resource_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing template: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    templates.append(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY))

THRESHOLD = 0.95
BRIGHTNESS_THRESHOLD = 110

was_active = False

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def capture_screen(region_only=True):
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if region_only:
        h, w = gray.shape
        roi = gray[int(h * 2/3):, int(w * 2/3):]
        return roi, (int(w * 2/3), int(h * 2/3))
    else:
        return gray, (0, 0)

def is_icon_present(screen, offset):
    for template in templates:
        th, tw = template.shape
        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= THRESHOLD:
            top_left = max_loc
            match_region = screen[top_left[1]:top_left[1]+th, top_left[0]:top_left[0]+tw]
            brightness = np.mean(match_region)
            if brightness >= BRIGHTNESS_THRESHOLD:
                return True
    return False

def main():
    global was_active
    spinner_index = 0

    screen, offset = capture_screen(region_only=True)
    _ = is_icon_present(screen, offset)

    clear_terminal()
    print("Starting Whispers Detector...")
    clear_terminal()

    while True:
        screen, offset = capture_screen(region_only=True)
        is_active = is_icon_present(screen, offset)

        if is_active and not was_active:
            clear_terminal()
            print("[+] Whispers activated.")
            sound_on.play()
            time.sleep(0.5)
            spinner_index = 0
            clear_terminal()
            print(spinner[spinner_index], end="\r", flush=True)
        elif not is_active and was_active:
            clear_terminal()
            print("[-] Whispers deactivated.")
            sound_off.play()
            time.sleep(0.5)
            clear_terminal()
            spinner_index = 0
            print(spinner[spinner_index], end="\r", flush=True)
        else:
            print(spinner[spinner_index], end="\r", flush=True)
            spinner_index = (spinner_index + 1) % len(spinner)

        was_active = is_active
        time.sleep(1)

if __name__ == "__main__":
    main()
