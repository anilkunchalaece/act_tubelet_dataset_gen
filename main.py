from lib.act_tubelet_generator import ActTubeletGenerator




def main(config_file) :
    generator = ActTubeletGenerator(config_file)
    # generator.generate_dataset()
    generator.get_train_test_split()



if __name__ == "__main__" :
    config_file = "generator_config.json"
    main(config_file)