from datetime import datetime as dt
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

INPUT_FILE_PATH = "/Users/baset/PycharmProjects/sentiment_analysis_all_models/processed_data/merged_df_AAPL.csv"

def print_report(df):
    posts_num = len(df)
    print("-The dataset contains: {} observations".format(posts_num))

    first_date = dt.strptime(min(df["posting_date"]), "%Y-%m-%d")
    last_date =  dt.strptime(max(df["posting_date"]), "%Y-%m-%d")
    delta = last_date - first_date
    time_period_in_days = delta.days

    print("-The oldest post in the dataset was published in : {} ".format(first_date))
    print("-The newest post in the dataset was published in : {} ".format(last_date))
    print("-The time period is : {} day".format(time_period_in_days))


    days_with_posts = (df.groupby(['posting_date']).sum()).index.values.tolist()
    num_of_days_with_posts = len(df.groupby(['posting_date']).sum())
    avg_num_post_per_day = posts_num / time_period_in_days

    print("-There is {} days with posts in the dataset".format(num_of_days_with_posts))
    print("-The average number of posts per day is: {}".format(avg_num_post_per_day))

    posts_distribution_over_time_df = df.groupby(['posting_date'])['id'].count().reset_index(name ='number_of_posts')
    posts_distribution_over_time_df['posting_date'] = pd.to_datetime(posts_distribution_over_time_df['posting_date'])

    biggest_number_of_posts_in_one_day = max(posts_distribution_over_time_df['number_of_posts'])
    smallest_number_of_posts_in_one_day = min(posts_distribution_over_time_df['number_of_posts'])

    print("-The biggest number of posts in one day is: {}".format(biggest_number_of_posts_in_one_day))
    print("-The smallest number of posts in one day is: {}".format(smallest_number_of_posts_in_one_day))

    complete_text = ""
    df.fillna('', inplace=True)
    for i, row in df.iterrows():
        complete_text = complete_text + row['title'] + " " + row['selftext']

    complete_text_words = complete_text.split()
    complete_text_words_count = len(complete_text_words)

    avg_num_of_words_per_post = complete_text_words_count // posts_num

    print("-The average number of words per post is: {}".format(avg_num_of_words_per_post))

    # num_days_with_posts_2008 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2008-01-01") & (sentiment_by_day['NY_Day'] <= "2008-12-31")]
    # num_days_with_posts_2009 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2009-01-01") & (sentiment_by_day['NY_Day'] <= "2009-12-31")]
    # num_days_with_posts_2010 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2010-01-01") & (sentiment_by_day['NY_Day'] <= "2010-12-31")]
    # num_days_with_posts_2011 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2011-01-01") & (sentiment_by_day['NY_Day'] <= "2011-12-31")]
    # num_days_with_posts_2012 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2012-01-01") & (sentiment_by_day['NY_Day'] <= "2012-12-31")]
    # num_days_with_posts_2013 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2013-01-01") & (sentiment_by_day['NY_Day'] <= "2013-12-31")]
    # num_days_with_posts_2014 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2014-01-01") & (sentiment_by_day['NY_Day'] <= "2014-12-31")]
    # num_days_with_posts_2015 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2015-01-01") & (sentiment_by_day['NY_Day'] <= "2015-12-31")]
    # num_days_with_posts_2016 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2016-01-01") & (sentiment_by_day['NY_Day'] <= "2016-12-31")]
    # num_days_with_posts_2017 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2017-01-01") & (sentiment_by_day['NY_Day'] <= "2017-12-31")]
    # num_days_with_posts_2018 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2018-01-01") & (sentiment_by_day['NY_Day'] <= "2018-12-31")]
    # num_days_with_posts_2019 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2019-01-01") & (sentiment_by_day['NY_Day'] <= "2019-12-31")]
    # num_days_with_posts_2020 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2020-01-01") & (sentiment_by_day['NY_Day'] <= "2020-12-31")]
    # num_days_with_posts_2021 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2021-01-01") & (sentiment_by_day['NY_Day'] <= "2021-12-31")]
    # num_days_with_posts_2022 = sentiment_by_day.loc[
    #     (sentiment_by_day['NY_Day'] >= "2022-01-01") & (sentiment_by_day['NY_Day'] <= "2022-12-31")]
    #
    # num_of_posts_2008 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2008-01-01") & (eval_df_merged['NY_Day'] <= "2008-12-31")]
    # num_of_posts_2009 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2009-01-01") & (eval_df_merged['NY_Day'] <= "2009-12-31")]
    # num_of_posts_2010 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2010-01-01") & (eval_df_merged['NY_Day'] <= "2010-12-31")]
    # num_of_posts_2011 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2011-01-01") & (eval_df_merged['NY_Day'] <= "2011-12-31")]
    # num_of_posts_2012 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2012-01-01") & (eval_df_merged['NY_Day'] <= "2012-12-31")]
    # num_of_posts_2013 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2013-01-01") & (eval_df_merged['NY_Day'] <= "2013-12-31")]
    # num_of_posts_2014 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2014-01-01") & (eval_df_merged['NY_Day'] <= "2014-12-31")]
    # num_of_posts_2015 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2015-01-01") & (eval_df_merged['NY_Day'] <= "2015-12-31")]
    # num_of_posts_2016 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2016-01-01") & (eval_df_merged['NY_Day'] <= "2016-12-31")]
    # num_of_posts_2017 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2017-01-01") & (eval_df_merged['NY_Day'] <= "2017-12-31")]
    # num_of_posts_2018 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2018-01-01") & (eval_df_merged['NY_Day'] <= "2018-12-31")]
    # num_of_posts_2019 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2019-01-01") & (eval_df_merged['NY_Day'] <= "2019-12-31")]
    # num_of_posts_2020 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2020-01-01") & (eval_df_merged['NY_Day'] <= "2020-12-31")]
    # num_of_posts_2021 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2021-01-01") & (eval_df_merged['NY_Day'] <= "2021-12-31")]
    # num_of_posts_2022 = eval_df_merged.loc[
    #     (eval_df_merged['NY_Day'] >= "2022-01-01") & (eval_df_merged['NY_Day'] <= "2022-12-31")]

    fig,ax = plt.subplots()
    # make a plot
    ax.plot(posts_distribution_over_time_df['posting_date'],
            posts_distribution_over_time_df['number_of_posts'],
            color="red")
    # set x-axis label
    ax.set_xlabel("date", fontsize = 14)
    # set y-axis label
    ax.set_ylabel("number_of_posts",
                  color="red",
                  fontsize=14)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE_PATH)
    print_report(df)
