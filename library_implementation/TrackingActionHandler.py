import json


class TrackingActionHandler:
    def __init__(self, env, json_path, auto_tracking_class):
        self.data = self.read_json_file(json_path)
        self.action_data = self.data["actions"]

        self.env = env

        self.auto_tracking_class = auto_tracking_class

        self.track_distance_lookup = {
            name: info["tracking"]["track_distance"]
            for name, info in self.data["targets"].items()
            if name != "enemy" and "tracking" in info
        }

    # called once per frame from main loop with new enemy profiles to return metrics needed to execute a together or successive tracking down action
    def go_to_target(self, target_profiles, left_index, right_index, action):
        auto_tracking = self.auto_tracking_class

        # nothing to track down if no target has been selected/is not auto tracking
        if not auto_tracking.is_tracking or auto_tracking.target_type_address is None:
            return None

        candidates = [
            profile
            for profile in target_profiles
            if profile["enemy_type"] == auto_tracking.target_type_address
        ]

        if not candidates:
            self.auto_tracking_class.deactivate_tracking()
            return None

        target = auto_tracking.pick_target(candidates)

        target_name = auto_tracking.target_type_name

        # getting all the supplied information to track the passed in target name
        tracking_info = self.data["targets"][target_name].get("tracking")

        # no configured information to do once tracked down
        if not tracking_info:
            return None

        tracking_distance = tracking_info["track_distance"]
        function_name = tracking_info["action"]

        action_info = self.action_data[function_name]  # getting info for custom action
        buttons = list(action_info.get("button_array", []))
        durations = list(action_info.get("duration_array", []))

        horizontal_distance = target["horizontal_distance"]

        if horizontal_distance >= 0:
            direction = "right"
        else:
            direction = "left"

        # if still out of range, keep walking in direction of target
        if abs(horizontal_distance) > tracking_distance:
            if direction == "right":
                action[right_index] = 1
            else:
                action[left_index] = 1

            return None

        # if in range, perform set action from config file
        # if mode is not successive button presses (together mode)
        if action_info["succession"] == "False":
            if durations:
                max_duration = max(
                    durations
                )  # can only hold for max duration, so ever button needs to be held for that max_duration
            else:
                max_duration = 0  # default

            # returns dictionary of necessary metrics needed to execute a together action
            return {
                # appending direction to front of buttons to press
                "buttons": [direction] + buttons,
                "hold_frames": max_duration,
                "succession": False,
            }

        # if successive button presses
        sequence_presses = [
            {"button": buttons[i], "duration": durations[i]}
            for i in range(len(buttons))
        ]  # creating dictionary of sequences of actions to occur - all the buttons to be pressed, with all their corresponding durations

        # returns diciontary of necessary metrics needed to execute a successive action
        return {
            "direction": direction,  # not appending direction to front of sequence_presses as may not want to move in that direction while performing the action
            "sequence_presses": sequence_presses,  # dictionary of buttons to durations to allow main method to cycle through each button index, take the corresponding duration, then move on to the next once duration = 0
            "succession": True,
        }

    def read_json_file(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)

        return data
