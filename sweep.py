import pygame
import time

def play_mp3(mp3_file):
    # Initialize Pygame
    pygame.init()

    try:
        # Load the MP3 file
        pygame.mixer.music.load(mp3_file)

        # Play the MP3 file
        pygame.mixer.music.play()

        # Wait while the music is playing
        while pygame.mixer.music.get_busy():
            time.sleep(1)

    except pygame.error as e:
        print(f"Error: {e}")
    finally:
        # Quit Pygame
        pygame.quit()

if __name__ == "__main__":
    mp3_file_path = "sweep_gottasweep.mp3"  # Replace with the path to your MP3 file
    play_mp3(mp3_file_path)
