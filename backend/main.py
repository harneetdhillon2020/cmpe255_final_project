from scripts.clean_data import clean_all_sites
from scripts.train_model import train_models_per_site

if __name__ == "__main__":
    clean_all_sites()
    train_models_per_site()
