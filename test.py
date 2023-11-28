if __name__ == "__main__":
    from src import submit_to_kaggle
    from src.configs.load_config import ConfigLoader
    config = ConfigLoader("config.json")
    submit_to_kaggle.submit(config, submit=True)

