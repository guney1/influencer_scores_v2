from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import re

@dataclass
class FeatGenerator:
    users: list = None
    fill_na: bool = False
    table_save_paths: dict = None
    mode: str = 'Train'

    def __post_init__(self):
        self.feat_dict = {}

    def get_target_data(self):
        # query = {"query": f"SELECT source, username, male, female, age1, age2, age3, age4, age5, real_user, bot_user from dashapi"}
        # self.db_query(query, self.table_paths['targets'])
        df_target = pd.read_csv(self.table_save_paths['targets'])

        # ----Generate Gender Target Data----
        df_target_gender = df_target[['source', 'username', 'male', 'female']]
        df_target_gender = df_target_gender.loc[df_target_gender[['male', 'female']].sum(axis=1).round()==100]
        df_target_gender = df_target_gender.drop_duplicates(subset=['source', 'username'])

        for user in df_target_gender.loc[df_target_gender.duplicated('username')].username:
            df_sub = df_target_gender.loc[df_target_gender['username'] == user]
            drop_idx = df_sub.loc[df_sub['source'] == 'Modash'].index
            df_target_gender = df_target_gender.drop(drop_idx, axis=0)

        df_target_gender = df_target_gender.drop_duplicates('username')

        # ----Generate Age Target Data----
        df_target_age = df_target[['source', 'username', 'age1', 'age2', 'age3', 'age4', 'age5']]
        df_target_age = df_target_age.loc[df_target_age[['age1', 'age2', 'age3', 'age4','age5']].sum(axis=1).round() >= 90]
        df_target_age = df_target_age.drop_duplicates(subset=['source', 'username'])

        for user in df_target_age.loc[df_target_age.duplicated('username')].username:
            df_sub = df_target_age.loc[df_target_age['username'] == user]
            drop_idx = df_sub.loc[df_sub['source'] == 'Modash'].index
            df_target_age = df_target_age.drop(drop_idx, axis=0)

        df_target_age = df_target_age.drop_duplicates('username')

        # ----Generate Bot Target Data----
        df_target_bot = df_target[['source', 'username', 'real_user', 'bot_user']]
        df_target_bot = df_target_bot.loc[df_target_bot[['real_user', 'bot_user']].sum(axis=1)!=0]
        df_target_bot = df_target_bot.drop_duplicates(subset=['source', 'username'])

        self.train_users = np.unique(list(df_target_gender['username'].values) + list(df_target_age['username'].values) + list(df_target_bot['username'].values))
        target_dfs = {
            'gender': df_target_gender, 
            'age': df_target_age,
            'bot': df_target_bot
        }
        self.feat_dict['target'] = target_dfs

        if self.mode == 'Train':
            self.users = self.train_users

        return target_dfs
    def influencers_table(self):
        # query = {"query": f"select username, post_count, follow_count, follower_count, sc_avg_daily_stories, sc_avg_daily_swipeup from influencers"}
        # self.db_query(query, self.table_paths['influencers'])
        df_infs = pd.read_csv(self.table_save_paths['influencers'])
        df_infs = df_infs.loc[df_infs.username.isin(self.users)]
        df_infs = df_infs.drop(['category_tags', 'bio'], axis=1)
    
        df_infs['post_count_pct'] = (df_infs['post_count']/df_infs['post_count'].mean())
        df_infs['follow_count_pct'] = (df_infs['follow_count']/df_infs['follow_count'].mean())
        df_infs['follower_count_pct'] = (df_infs['follower_count']/df_infs['follower_count'].mean())

        df_infs.index = df_infs.username
        df_infs = df_infs.drop('username', axis=1)
        df_infs.loc['DROPME', :] = np.nan
        self.feat_dict['influencers'] = df_infs

    def data_center_table(self):
        # query = {"query": f"SELECT username, type, create_date_new, view, likecount, comments, hashtags, mentions, is_mentioned_brand_brands, is_mentioned_brand_sectors FROM datacenter"}
        # self.db_query(query, self.table_paths['data_center'])

        df_dc = pd.read_csv(self.table_save_paths['data_center'])
        df_dc = df_dc.loc[df_dc.username.isin(self.users)]

        df_dc['create_date'] = pd.to_datetime(df_dc.create_date_new)
        df_dc = df_dc.drop('create_date_new', axis=1)
        df_dc = df_dc.sort_values(['username', 'type', 'create_date'])

        # Split into Reel, post, story features
        df_dc_reel = df_dc.loc[df_dc.type == 'Reel']
        df_dc_post = df_dc.loc[df_dc.type == 'Post']
        df_dc_story = df_dc.loc[df_dc.type == 'Story']

        # ----Story Feats----
        df_dc_story_count = df_dc_story.groupby(['username', 'create_date'], as_index=False)[['type']].count().sort_index().rename(columns={'type': 'story_count'})
        df_dc_avg_story = pd.DataFrame(df_dc_story_count.groupby('username')['story_count'].mean().sort_values())
        df_dc_avg_story.loc['DROPME', :] = np.nan
        self.feat_dict['dc_stories'] = df_dc_avg_story

        # ----Reels Feats----
        df_dc_reel = df_dc_reel.loc[df_dc_reel.view != 0]
        df_dc_reel_stats = df_dc_reel.groupby('username')[['view', 'likecount', 'comments']].mean()
        df_dc_reel_stats.loc['DROPME', :] = np.nan

        # Avg like count to avg reel view
        df_dc_reel_stats['like_to_view_ratio'] = df_dc_reel_stats['likecount']/df_dc_reel_stats['view']
        # Avg comment count to avg reel view
        df_dc_reel_stats['comment_to_view_ratio'] = df_dc_reel_stats['comments']/df_dc_reel_stats['view']

        assign_idx = df_dc_reel_stats.index.intersection(self.feat_dict['influencers'].index)
        df_dc_reel_stats.loc[assign_idx, 'like_to_follower'] = df_dc_reel_stats.loc[assign_idx, 'likecount']/self.feat_dict['influencers'].loc[assign_idx, 'follower_count']
        df_dc_reel_stats.loc[assign_idx, 'comment_to_follower'] = df_dc_reel_stats.loc[assign_idx, 'comments']/self.feat_dict['influencers'].loc[assign_idx, 'follower_count']

        df_dc_reel_stats.columns = [f'reel_{col}' for col in df_dc_reel_stats.columns]
        self.feat_dict['dc_reels'] = df_dc_reel_stats

        # ----Post Feats----
        df_dc_post['is_sponsored'] = (~df_dc_post['is_mentioned_brand_brands'].isna()).astype(int)

        df_dc_post['hash_count'] = df_dc_post.hashtags.dropna().apply(lambda x: len(x.split()))
        df_dc_post['mention_count'] = df_dc_post.mentions.dropna().apply(lambda x: len(x.split()))

        df_dc_post['hash_count'] = df_dc_post['hash_count'].fillna(0)
        df_dc_post['mention_count'] = df_dc_post['mention_count'].fillna(0)

        df_dc_post_mean = df_dc_post[['username', 'likecount', 'comments', 'hash_count', 'mention_count', 'is_sponsored']].groupby('username').mean()
        df_dc_post_mean.loc['DROPME', :] = np.nan
        
        assign_idx = df_dc_post_mean.index.intersection(self.feat_dict['influencers'].index)

        df_dc_post_mean.loc[assign_idx, 'like_to_follower'] = df_dc_post_mean.loc[assign_idx, 'likecount']/self.feat_dict['influencers'].loc[assign_idx, 'follower_count']
        df_dc_post_mean.loc[assign_idx, 'comment_to_follower'] = df_dc_post_mean.loc[assign_idx, 'comments']/self.feat_dict['influencers'].loc[assign_idx, 'follower_count']

        df_dc_post_mean.columns = [f'post_{col}' for col in df_dc_post_mean.columns]
        self.feat_dict['dc_table_post'] = df_dc_post_mean

    def pool_user_table(self):
        # query = {"query": f"SELECT from_influencer, username, gender, age, follows, followers FROM pool_user"}
        # self.db_query(query, self.table_paths['pool_user'])
        df_pool = pd.read_csv(self.table_save_paths['pool_user'])

        df_pool['from_influencer'] = df_pool.from_influencer.astype(str).apply(lambda x: re.sub(',', ' ', x).split())
        df_pool['user_to_inf_count'] = df_pool['from_influencer'].apply(lambda x: len(x))

        # Split DF to single influencer and multiple influencer follows
        df_pool_single = df_pool.loc[df_pool.user_to_inf_count == 1].reset_index(drop=True)
        df_pool_mult = df_pool.loc[df_pool.user_to_inf_count != 1].reset_index(drop=True)
        # Remove the list format from influencers
        df_pool_single['from_influencer'] = df_pool_single.from_influencer.apply(lambda x: x[0])

        from tqdm import tqdm
        new_rows = []
        for idx in tqdm(df_pool_mult.index):
            infs = df_pool_mult.loc[idx].from_influencer
            row = df_pool_mult.loc[idx]
            modified_row = pd.concat([row] * len(infs), axis=1).T
            modified_row['from_influencer'] = infs
            new_rows.append(modified_row)

        df_pool_mult_new = pd.concat(new_rows, axis=0).reset_index(drop=True)
        df_pool_whole = pd.concat([df_pool_single, df_pool_mult_new], axis=0).reset_index(drop=True)

        # Map the ages and genders to integers
        gender_map = {'M': 1, 'W': 0}
        age_map = {'0_17':0, '18_34':1, '35_59': 2, '60_inf': 3}
        df_pool_whole['gender'] = df_pool_whole.gender.map(gender_map)
        df_pool_whole['age'] = df_pool_whole.age.map(age_map)

        df_pool_whole['gender'] = df_pool_whole['gender'].fillna(2)
        df_pool_whole['age'] = df_pool_whole['age'].fillna(4)

        df_pool_whole = df_pool_whole.loc[df_pool_whole.from_influencer.isin(self.users)]
        df_pool_age_count = df_pool_whole.groupby('from_influencer', as_index=False)['age'].value_counts()

        unq_infs = df_pool_age_count.from_influencer.unique()
        for inf in unq_infs:
            exist_ages = df_pool_age_count.loc[df_pool_age_count.from_influencer == inf].age.unique()
            missing_ages = list(set(list(age_map.values())) - set(list(exist_ages)))
            for age in missing_ages:
                df_pool_age_count.loc[max(df_pool_age_count.index)+1] = {'from_influencer': inf, 'age': age, 'count': 0}

        df_pool_age_count = df_pool_age_count.sort_values('from_influencer').reset_index(drop=True)

        df_pool_age_count = (df_pool_age_count.pivot(index='from_influencer', columns='age', values='count').fillna(0)/df_pool_age_count.pivot(index='from_influencer', columns='age', values='count').fillna(0).sum(axis=1).values.reshape(-1,1))
        df_pool_avg_inf_follow = df_pool_whole.groupby('from_influencer')[['user_to_inf_count']].mean()
        df_pool_feat = pd.concat([df_pool_age_count, df_pool_avg_inf_follow], axis=1)

        df_pool_gender_count = df_pool_whole[['from_influencer', 'gender']].replace(2, np.nan).groupby('from_influencer')[['gender']].mean()
        df_pool_feat = pd.concat([df_pool_feat, df_pool_gender_count], axis=1)
        df_pool_feat.loc['DROPME', :] = np.nan

        self.feat_dict['pool_user'] = df_pool_feat

    def get_all_feats(self):
        self.influencers_table()
        self.data_center_table()
        self.pool_user_table()
        return self.feat_dict

    def combine_feats(self):
        self.df_feats_all = pd.concat({key: self.feat_dict[key] for key in self.feat_dict if key != 'target'}, axis=1)
        self.df_feats_all.columns = [f'{str(a[0])}_{str(a[1])}' for a in self.df_feats_all.columns]
        if self.fill_na:
            self.df_feats_all = self.df_feats_all.fillna(0)
        else:
            self.df_feats_all = self.df_feats_all.dropna()

        return self.df_feats_all.drop(['DROPME'], axis=0)

    def generate_gender_train_data(self):
        df_target = self.feat_dict['target']['gender'].copy()
        
        df_target = df_target.loc[df_target['source'] == 'IGA']
        df_target.index = df_target.username
        df_target.index.name = None
        df_target = df_target.drop(['username', 'source'], axis=1)

        df_model = pd.concat([self.df_feats_all, df_target[['male']]], axis=1, join='inner').dropna()
        return df_model
    
    def generate_age_train_data(self):
        df_target = self.feat_dict['target']['age'].copy()

        df_target = df_target.loc[df_target['source'] == 'IGA']
        df_target.index = df_target.username
        df_target.index.name = None
        df_target = df_target.drop(['username', 'source'], axis=1)

        df_model = pd.concat([self.df_feats_all, df_target], axis=1, join='inner').dropna()
        return df_model
    
    def generate_bot_train_data(self):
        df_target = self.feat_dict['target']['bot'].copy()
        df_target = df_target.drop_duplicates('username')
        df_target.index = df_target['username']
        df_target.index.name = None
        df_target = df_target.drop(['source', 'username'], axis=1)

        df_model = pd.concat([self.df_feats_all, df_target['bot_user']], axis=1, join='inner').dropna()
        return df_model


if __name__ == '__main__':
    table_save_paths = {
        'targets': '/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/v2_data/dashapi250821.csv',
        'influencers': '/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/v2_data/influencers250821.csv',
        'data_center': '/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/v2_data/datacenter250821.csv',
        'pool_user': '/Users/guneykanozkaya/Desktop/projects/projects_structered/data/influncer_scores/v2_data/pooluser250821.csv'
    }
    feat_gen = FeatGenerator(table_save_paths=table_save_paths)
    target_dfs = feat_gen.get_target_data()
    feat_dfs = feat_gen.get_all_feats()
    df_feats_all = feat_gen.combine_feats()
    df_gender = feat_gen.generate_gender_train_data()
    df_age = feat_gen.generate_age_train_data()
    df_bot = feat_gen.generate_bot_train_data()
    print('ok')

















        




        


































