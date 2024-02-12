from lib.act_tubelet_generator import ActTubeletGenerator


ATOMIC_ACTION_CLASSES = [
    "activity_walking",
    "activity_running",
    "activity_standing",
    "activity_sitting"
]

CONCURRENT_ACTION_CLASSES = [
    # "activity_carrying",
    "activity_gesturing",
    "Talking",
    "specialized_talking_phone",
    "specialized_using_tool",
    "Open_Trunk",
    "Interacts",
    "Unloading",
    "Pull",
    "Entering",
    "specialized_texting_phone",
    "Talking",
    "SetDown",
    "Loading",
    "Riding",
    "Closing_Trunk",
    "Push",
    "Person_Person_Interaction",
    "Closing"
    "Opening"
]

def main(config_file) :
    generator = ActTubeletGenerator(config_file)
    generator.generate_dataset()
    generator.get_train_test_split()
    # generator.get_dataset_stats(CONCURRENT_ACTION_CLASSES,"concurrent_action")



if __name__ == "__main__" :
    config_file = "generator_config.json"
    main(config_file)