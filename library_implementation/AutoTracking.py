import logging

from ..library.SemanticMapper import SemanticMapper

logger = logging.getLogger("AutoTracking")


class AutoTracking:
    def __init__(self, json_path):
        self.data = self.read_json_file(json_path)
        self.character_data = self.data["characters"]
        self.enemy_data = self.data["targets"]["enemy"]
        self.item_data = self.data["items"]
        self.env_data = self.data["environment"]

        self.semantic_mapper = SemanticMapper(
            json_path
        )  # not using in this file, but called in others

        self.target_type_lookup = {
            name: int(
                info["address"], 16
            )  # 16 to interpret as base-sixteen/hexadecimal format
            for name, info in self.data["targets"].items()
            if name != "enemy"
        }

        self.is_tracking = False  # stores if auto tracking is currently active
        self.target_type_address = (
            None  # holds the memory address for the target going to be tracked
        )
        self.target_type_name = None  # holds the name for the target going to be tracked to find the closest candidates/one

        (
            self.character_absolute_page_number,
            self.character_vertical_screen_position,
            self.character_horizontal_position,
            self.general_dies_state,
            self.character_dies_state,
        ) = self.initialize_character_variables()
        (
            self.enemy_active_state,
            self.enemy_type_address,
            self.enemy_horizontal_velocity,
            self.enemy_absolute_map_num,
            self.enemy_horizontal_page_position,
            self.enemy_vertical_screen_position,
        ) = self.initialize_enemy_variables()
        self.initialize_item_veriables()
        self.env_time_hundred, self.env_time_ten, self.env_time_one = (
            self.initialize_environment_variables()
        )

    def initialize_character_variables(self):
        character_absolute_page_number = int(
            self.character_data["absolute_page_number"], 16
        )
        character_vertical_screen_position = int(
            self.character_data["vertical_screen_position"], 16
        )
        character_horizontal_position = int(
            self.character_data["horizontal_position"], 16
        )
        general_dies_state = int(self.character_data["dies_state"], 16)
        character_dies_state = int(self.character_data["character_state"], 16)

        return (
            character_absolute_page_number,
            character_vertical_screen_position,
            character_horizontal_position,
            general_dies_state,
            character_dies_state,
        )

    def initialize_enemy_variables(self):
        enemy_active_state = int(self.enemy_data["active_state"], 16)
        enemy_type_address = int(self.enemy_data["enemy_type"], 16)
        enemy_horizontal_velocity = int(self.enemy_data["horizontal_velocity"], 16)
        enemy_absolute_map_num = int(self.enemy_data["absolute_map_num"], 16)
        enemy_horizontal_page_position = int(
            self.enemy_data["horizontal_page_position"], 16
        )
        enemy_vertical_screen_position = int(
            self.enemy_data["enemy_vertical_screen_position"], 16
        )

        return (
            enemy_active_state,
            enemy_type_address,
            enemy_horizontal_velocity,
            enemy_absolute_map_num,
            enemy_horizontal_page_position,
            enemy_vertical_screen_position,
        )

    def initialize_item_veriables(self):
        pass

    def initialize_environment_variables(self):
        env_time_hundred = int(self.env_data["time_hundred"], 16)
        env_time_ten = int(self.env_data["time_ten"], 16)
        env_time_one = int(self.env_data["time_one"], 16)

        return env_time_hundred, env_time_ten, env_time_one

    def activate_tracking(self):
        self.is_tracking = True

    def deactivate_tracking(self):
        self.is_tracking = False
        self.target_type_address = None
        self.target_type_name = None

    def set_target_from_similarity(self, transcript_sentence, min_confidence=0.17):
        name, score = self.semantic_mapper.find_max_target_similarity(
            transcript_sentence
        )

        if score < min_confidence:
            self.target_type_address = None
            self.target_type_name = None

            print("Not confident enough to identify target")

            return None, score

        self.target_type_address = self.target_type_lookup.get(name)
        self.target_type_name = name

        return name, score

    # method to activate auto tracking while setting the target name
    def activate_set_target(self, transcript_sentence):
        self.activate_tracking()
        name, score = self.set_target_from_similarity(transcript_sentence)

        return name, score

    # method to find closest enemy target to get
    def pick_target(self, enemy_profiles):
        # starting point by assuming first enemy in enemy list is the closest to mario
        closest_enemy = enemy_profiles[0]
        # getting amount of time until collision
        closest_time = closest_enemy.get("time_to_collision_frames", float("inf"))

        # checking every other enemy from index 1 to the end of the list
        for enemy in enemy_profiles[1:]:
            # getting corresponding time to collision
            enemy_time = enemy.get(
                "time_to_collision_frames", float("inf")
            )  # defaulting to infinite frames until collision to prevent any future crashes

            # if new enemy time is less than closest found time yet, set enemy and time values accordingly
            if enemy_time < closest_time:
                closest_enemy = enemy
                closest_time = enemy_time

        # return which enemy is the closest one
        return closest_enemy

    # method to get positions of mario and enemy from memory
    def get_game_positions(self, env):
        ram = env.unwrapped.get_ram()

        # print(type(env.unwrapped), env.unwrapped.get_ram())

        # page on which mario currently is at (width of 256 pixels)
        character_x_page = int(ram[self.character_absolute_page_number])

        # horizontal position of mario in current page
        character_x_screen = int(ram[self.character_horizontal_position])

        # gets position of mario from very start of level, since each page = 256 pixels long,
        # so multiplying page number by page width then adding mario x position on current page produces distance from start of level
        character_x_pos = (character_x_page * 256) + character_x_screen

        # mario y position
        character_y_pos = int(ram[self.character_vertical_screen_position])

        # initializing empty list to hold data for any enemies currently on screen
        enemy_slots = []

        # looping through the 5 available hardware slots that the nes uses to track active enemies (from data crystal)
        for i in range(5):

            # checking to see if current enemy slot contains a live enemy (note that 00 means empty/dead)
            enemy_active = ram[
                self.enemy_active_state + i
            ]  # + i allows active enemies to be checked for in all 5 possible enemy ram slots

            # if there is an active enemy
            if enemy_active:

                # tracking absolute positions of enemies in each slot
                enemy_x_page = int(ram[self.enemy_absolute_map_num + i])
                enemy_x_screen = int(ram[self.enemy_horizontal_page_position + i])

                # getting absolute position of enemy from start of level to properly match mario's coordinate scale
                enemy_x_pos = (enemy_x_page * 256) + enemy_x_screen

                # enemy y position
                enemy_y_pos = int(ram[self.enemy_vertical_screen_position + i])

                # enemy type
                enemy_type = int(ram[self.enemy_type_address + i])

                # adding to currently looped over slot the x and y positions of an enemy if one is there
                enemy_slots.append(
                    {"slot": i, "x": enemy_x_pos, "y": enemy_y_pos, "type": enemy_type}
                )

        # returning dictionary containing mario's x and y positions, and the positions of enemies in each slot
        return {
            "character": {"x": character_x_pos, "y": character_y_pos},
            "enemies": enemy_slots,
        }

    # method to get the horizontal and vertical distances between mario and enemies
    def get_distances_to_targets(self, env, positions):
        ram = env.unwrapped.get_ram()

        # dictionary to hold distance metrics for every active enemy found on screen
        enemy_metrics = []

        # extracting mario coordinates
        character_x = positions["character"]["x"]
        character_y = positions["character"]["y"]

        # looping through each active enemy dictionary stored in the provided list
        for enemy in positions["enemies"]:
            # calculating horizontal distance (note that positive means enemy is right, negative means enemy is left)
            horizontal_distance = enemy["x"] - character_x

            # calculating vertical distance (positive means enemy is below mario, negative means above mario)
            vertical_distance = enemy["y"] - character_y

            # finding distance using pythagorean theorem
            distance = (horizontal_distance**2 + vertical_distance**2) ** 0.5

            enemy_raw_speed_byte = int(
                ram[self.enemy_horizontal_velocity + enemy["slot"]]
            )

            # raw speed byte idea fine-tuned with AI autocomplete
            # raw speed byte is able to hold direction and speed in same value, where 129-255 = left and 0-128 = right, where higher values in each range = speed

            # if byte raw value is above 128/represents negative values/moving left
            if enemy_raw_speed_byte > 128:
                enemy_direction = "left"  # set direction
                enemy_speed = abs(
                    256
                    - enemy_raw_speed_byte  # the speed of the enemy then must be 256 - enemy's raw speed byte, using Two's Complement system - negative values stored in this range to prevent using extra space to hold the sign
                )

            # if byte raw value is below 128/represents positive values/moving right
            else:
                enemy_direction = "right"  # set direction
                enemy_speed = enemy_raw_speed_byte  # speed of enemy is just the raw speed byte - no need for Two's Complement trick, as not negative values/direction is right

            # fallback safety check to avoid any zero division errors if an enemy is momentarily stationary
            if enemy_speed == 0:
                enemy_speed = 1

            # block determining if velocity direction means the gap is actively closing between enemy and mario
            is_moving_towards_character = False

            # if enemy is to the right of mario (positive distance) and its physics vector is going left
            if horizontal_distance > 0 and enemy_direction == "left":
                is_moving_towards_character = True

            # if enemy is to the left of mario (negative distance) and its physics vector is going right
            elif horizontal_distance < 0 and enemy_direction == "right":
                is_moving_towards_character = True

            # calculating time to collision with time = distance / speed - note that if enemy moving away from mario, set time to infinity to avoid any errors
            if is_moving_towards_character:
                time_to_collision_frames = abs(horizontal_distance) / enemy_speed
            else:
                time_to_collision_frames = float("inf")

            # appending spatial calculations, distance, and real time collision predictions to output dictionary
            enemy_metrics.append(
                {
                    "enemy_slot_number": enemy["slot"],
                    "enemy_type": enemy["type"],
                    "horizontal_distance": horizontal_distance,
                    "vertical_distance": vertical_distance,
                    "distance": round(distance, 2),
                    "speed_per_frame": enemy_speed,
                    "is_coming_towards": is_moving_towards_character,
                    "time_to_collision_frames": time_to_collision_frames,
                }
            )

        # returning calculated metrics
        return enemy_metrics

    def read_json_file(self, json_path):
        import json

        with open(json_path, "r") as file:
            data = json.load(file)

        return data
