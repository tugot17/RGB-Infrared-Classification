from rgb_only import run_rgb_only_experiments
from rgb_infrared_combined import run_rgb_infrared_combined_experiments
from infrared_only import run_infrared_only_experiments
from rgb_infrared_as_two_separate_inputs import (
    run_rgb_infrared_as_two_separate_inputs_experiments,
)

if __name__ == "__main__":
    # pass
    run_rgb_infrared_as_two_separate_inputs_experiments()
    run_rgb_infrared_combined_experiments()
    run_rgb_only_experiments()
    run_infrared_only_experiments()
