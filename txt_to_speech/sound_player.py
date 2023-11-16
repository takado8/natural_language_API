import pygame
import time


def play_sound(file_path):
    pygame.init()
    pygame.mixer.init()

    try:
        pygame.mixer.music.load(file_path)
        print(f"Playing: {file_path}")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            # Keep the program running while the music is playing
            time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.mixer.quit()
        pygame.quit()
