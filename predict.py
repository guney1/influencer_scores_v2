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
        users=['aykutelmas', 'danlabilic', 'fatihyasinim'],
        fill_na=True,
        table_save_paths=table_save_paths,
        mode='Predict'
    )

    predictor = Clasifier(
        gender_model_path='/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/models/v2_250930_gender_knn_8_neigh',
        age_model_path='/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/models/v2_250930_age_knns_15_neigh.pickle', 
        bot_user_model_path='/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/models/v2_250930_bot_user_knn_6_neigh',
        scaler_path='/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/models/v2_250930_minmax_scaler', 
        users=['aykutelmas', 'danlabilic', 'fatihyasinim'],
        mode='Predict'
    )

    feat_dfs = feat_gen.get_all_feats()
    df_feats_all = feat_gen.combine_feats()
    pred_dict = predictor.predict(df_feats_all)


