from feat_generator import FeatGenerator
from classifier import Clasifier


if __name__ == '__main__':
    table_save_paths = {
        'targets': '/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/v2_data/dashapi250821.csv',
        'influencers': '/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/v2_data/influencers250821.csv',
        'data_center': '/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/v2_data/datacenter250821.csv',
        'pool_user': '/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/v2_data/pooluser250821.csv'
    }

    feat_gen = FeatGenerator(
        users=None,
        fill_na=False,
        table_save_paths=table_save_paths,
        mode='Train'
    )

    predictor = Clasifier(
        gender_model_path=None,
        age_model_path=None, 
        scaler_path=None, 
        mode='Train'
    )
    
    target_dfs = feat_gen.get_target_data()
    feat_dfs = feat_gen.get_all_feats()
    df_feats_all = feat_gen.combine_feats()
    
    df_gender = feat_gen.generate_gender_train_data()
    df_age = feat_gen.generate_age_train_data()
    df_bot = feat_gen.generate_bot_train_data()

    scaler = predictor.train_scaler(df_feats_all)
    knn_gender = predictor.train_gender_model(df_gender, 8)
    knn_age = predictor.train_age_model(df_age, 15)
    knn_bot = predictor.train_bot_model(df_bot, 6)