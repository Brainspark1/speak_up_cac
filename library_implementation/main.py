import sys
import logging
import os

import stable_retro as retro
import pygame

from ..library_implementation.AutoTracking import AutoTracking
from ..library_implementation.ManualActionHandler import ManualActionHandler
from ..library_implementation.TrackingActionHandler import TrackingActionHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Initialize Pygame to handle window rendering and keyboard input
pygame.init()

JSON_CONFIG_PATH = "main/configs/mario_config.json"
ENV_NAME = "SuperMarioBros-Nes-v0"

try:
    env = retro.make(game=ENV_NAME, state=retro.State.DEFAULT)
except Exception as e:
    print(f"Error loading environment: {e}")
    print("Ensure your ROM is imported using: python3 -m retro.import /path/to/roms")
    sys.exit(1)

# voice commands via ManualActionHandler set target onto shared AutoTracking instance, TrackingActionHandler reads same state every frame to decide how to get character to move
auto_tracking_class = AutoTracking(JSON_CONFIG_PATH)

manual_action_handler = ManualActionHandler(
    mapping_json_path=JSON_CONFIG_PATH,
    auto_tracking_class=auto_tracking_class,
    device_backend="mps",
    whisper_model_size="tiny.en",
)

manual_action_handler.initial_prompt = f"Look out for words related to this game, especially {manual_action_handler.names_string}."

tracking_action_handler = TrackingActionHandler(
    env, JSON_CONFIG_PATH, auto_tracking_class
)

ram = env.unwrapped.get_ram()

# starting background listening thread from parent class NESVoiceController: whisper --> NESBERT --> process_game_commands()
stop_listening = manual_action_handler.start_listening()

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0, 0, 0, 0, 0])

SCREEN_SCALE = 3
screen_width = obs.shape[1] * SCREEN_SCALE
screen_height = obs.shape[0] * SCREEN_SCALE
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Super Mario Bros - Voice Control")

clock = pygame.time.Clock()
running = True

current_command = None  # currently executing auto-tracking command
step_index = 0  # which step in sequence for succession mode
step_remaining = 0  # frames left on current step for succession mode
sustain_remaining = 0  # frames left for together mode

BUTTON_INDICES = manual_action_handler.button_to_index_mapping

while running:
    # Handles window close or escape key
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # blank NES controller array: [B, None, MODE, START, UP, DOWN, LEFT, RIGHT, A]
    # All buttons default to False (0)
    action = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # voice actions start live here, being set and decremented in duration each frame
    duration = manual_action_handler.get_duration_array()

    # keyboard states
    keys = pygame.key.get_pressed()

    # WASD to D-pad
    if keys[pygame.K_w]:
        action[4] = 1  # UP
    if keys[pygame.K_s]:
        action[5] = 1  # DOWN
    if keys[pygame.K_a]:
        action[6] = 1  # LEFT
    if keys[pygame.K_d]:
        action[7] = 1  # RIGHT

    # E to Attack
    if keys[pygame.K_e]:
        action[0] = 1  # B

    # Spacebar to Jump (Button A)
    if keys[pygame.K_SPACE]:
        action[1] = 1  # A

    # counting down durations for each button in 9 slots
    for i in range(9):
        if duration[i] > 0:  # if time still left on action
            action[i] = 1  # continue pressing button/index = 1
            duration[i] -= 1  # decrease duration by one

    # if no auto tracking attack is going on, check if need to move toward target or actually execute action
    if current_command is None:
        positions = tracking_action_handler.auto_tracking_class.get_game_positions(env)

        profiles = tracking_action_handler.auto_tracking_class.get_distances_to_targets(
            env, positions
        )

        command = tracking_action_handler.go_to_target(
            profiles,
            left_index=BUTTON_INDICES.get("left"),
            right_index=BUTTON_INDICES.get("right"),
            action=action,
        )

        # if there is a command to track returned from go_to_target()
        if command:
            current_command = command

            if command["succession"]:
                step_index = 0
                step_remaining = command["sequence_presses"][0]["duration"]

                if command["direction"] == "right":
                    action[BUTTON_INDICES.get("right")] = 1
                else:
                    action[BUTTON_INDICES.get("left")] = 1
            else:
                sustain_remaining = command[
                    "hold_frames"
                ]  # the amount of time to hold is the max duration passed in initially

                for button in command["buttons"]:
                    index = BUTTON_INDICES.get(button.lower())

                    if index is not None:
                        action[index] = 1

    # conditional handling the variables set above if there is a command to execute
    if current_command:
        if current_command["succession"]:  # succession
            if current_command["direction"] == "right":
                action[BUTTON_INDICES.get("right")] = 1
            else:
                action[BUTTON_INDICES.get("left")] = 1

            current_step = current_command["sequence_presses"][step_index]
            button_index = BUTTON_INDICES.get(current_step["button"].lower())

            if button_index is not None:
                action[button_index] = 1

            # decreasing duration for button in current step_index
            step_remaining -= 1

            if step_remaining <= 0:
                # moving to next step
                step_index += 1

                if step_index >= len(
                    current_command["sequence_presses"]
                ):  # if have reached the end/current step index is at the number of steps/buttons to be pressed
                    current_command = None
                else:
                    step_remaining = current_command["sequence_presses"][step_index][
                        "duration"  # getting the duration of the current button at the current step index being checked
                    ]

        else:  # together
            for button in current_command["buttons"]:
                index = BUTTON_INDICES.get(button.lower())

                if index is not None:
                    action[index] = 1

            # decreasing amount of time pressing all the buttons together
            sustain_remaining -= 1

            if sustain_remaining <= 0:
                current_command = None

    # Step the environment forward with custom actions
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        current_command = None
        action = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    current_death_state = int(
        ram[auto_tracking_class.general_dies_state]
    )  # getting current state of life according to emulator

    if current_death_state == auto_tracking_class.character_dies_state:
        current_command = None
        action = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        print("Character death detected")

    # Converts the environment's RGB frame array to a Pygame surface and display it
    # Transpose frame array from (Height, Width, Channel) to Pygame's (Width, Height, Channel)
    frame = obs.transpose(1, 0, 2)
    surf = pygame.surfarray.make_surface(frame)

    # Scale and draw the game frame
    scaled_surf = pygame.transform.scale(surf, (screen_width, screen_height))
    screen.blit(scaled_surf, (0, 0))
    pygame.display.flip()

    clock.tick(45)

stop_listening(wait_for_stop=False)
env.close()
pygame.quit()
sys.exit()
