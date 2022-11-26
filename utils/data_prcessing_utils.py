import pandas as pd
import json
import redditcleaner

emojis_mapping = {'<U+0001F319>':'\U0001F319','<U+0001F98D>':'\U0001F98D','<U+0001F36D>':'\U0001F36D','<U+0001F9AF>':'\U0001F9AF','<U+0001F938>':'\U0001F938','<U+0001F9D1>':'\U0001F9D1','<U+0001F435>':'\U0001F435','<U+0001F919>':'\U0001F919','<U+0001F3A2>':'\U0001F3A2','<U+0001F90C>':'\U0001F90C','<U+0001F579>':'\U0001F579','<U+0001F928>':'\U0001F928','<U+0001F389>':'\U0001F389','<U+0001F3F0>':'\U0001F3F0','<U+0001F35E>':'\U0001F35E','<U+0001F39F>':'\U0001F39F','<U+0001F92C>':'\U0001F92C','<U+0001F4A5>':'\U0001F4A5','<U+0001F351>':'\U0001F351','<U+0001F404>':'\U0001F404','<U+0001F312>':'\U0001F312','<U+0001F4B9>':'\U0001F4B9','<U+0001F525>':'\U0001F525','<U+0001F4BA>':'\U0001F4BA','<U+0001F356>':'\U0001F356','<U+0001F96D>':'\U0001F96D','<U+0001F631>':'\U0001F631','<U+0001F4F1>':'\U0001F4F1','<U+0001FA73>':'\U0001FA73','<U+0001F620>':'\U0001F620','<U+0001F4A9>':'\U0001F4A9','<U+0001F41F>':'\U0001F41F','<U+0001F408>':'\U0001F408','<U+0001F42E>':'\U0001F42E','<U+0001F1E6>':'\U0001F1E6','<U+0001F37E>':'\U0001F37E','<U+0001F633>':'\U0001F633','<U+0001F468>':'\U0001F468','<U+0001F649>':'\U0001F649','<U+0001F40B>':'\U0001F40B','<U+0001F44C>':'\U0001F44C','<U+0001F47E>':'\U0001F47E','<U+0001FAA8>':'\U0001FAA8','<U+0001F320>':'\U0001F320','<U+0001F5E3>':'\U0001F5E3','<U+0001F910>':'\U0001F910','<U+0001F37B>':'\U0001F37B','<U+0001F433>':'\U0001F433','<U+0001F36B>':'\U0001F36B','<U+0001F61A>':'\U0001F61A','<U+0001F6D2>':'\U0001F6D2','<U+0001F354>':'\U0001F354','<U+0001F63A>':'\U0001F63A','<U+0001F476>':'\U0001F476','<U+0001F6AA>':'\U0001F6AA','<U+0001F512>':'\U0001F512','<U+0001FA82>':'\U0001FA82','<U+0001F30D>':'\U0001F30D','<U+0001F6A8>':'\U0001F6A8','<U+0001F9DB>':'\U0001F9DB','<U+0001F3FD>':'\U0001F3FD','<U+0001F480>':'\U0001F480','<U+0001F600>':'\U0001F600','<U+0001F648>':'\U0001F648','<U+0001F495>':'\U0001F495','<U+0001F40A>':'\U0001F40A','<U+0001F971>':'\U0001F971','<U+0001F418>':'\U0001F418','<U+0001F450>':'\U0001F450','<U+0001F926>':'\U0001F926','<U+0001F171>':'\U0001F171','<U+0001F606>':'\U0001F606','<U+0001F980>':'\U0001F980','<U+0001F4E2>':'\U0001F4E2','<U+0001F64C>':'\U0001F64C','<U+0001F469>':'\U0001F469','<U+0001F1EC>':'\U0001F1EC','<U+0001F357>':'\U0001F357','<U+0001F51D>':'\U0001F51D','<U+0001F31C>':'\U0001F31C','<U+0001F643>':'\U0001F643','<U+0001F440>':'\U0001F440','<U+0001F970>':'\U0001F970','<U+0001FA90>':'\U0001FA90','<U+0001F5D3>':'\U0001F5D3','<U+0001F4B8>':'\U0001F4B8','<U+0001F9D3>':'\U0001F9D3','<U+0001F966>':'\U0001F966','<U+0001F92B>':'\U0001F92B','<U+0001F647>':'\U0001F647','<U+0001F3CB>':'\U0001F3CB','<U+0001F393>':'\U0001F393','<U+0001F927>':'\U0001F927','<U+0001F3F4>':'\U0001F3F4','<U+0001F349>':'\U0001F349','<U+0001F634>':'\U0001F634','<U+0001F625>':'\U0001F625','<U+0001F55C>':'\U0001F55C','<U+0001F93A>':'\U0001F93A','<U+0001F474>':'\U0001F474','<U+0001F1EA>':'\U0001F1EA','<U+0001F330>':'\U0001F330','<U+0001F91B>':'\U0001F91B','<U+0001F411>':'\U0001F411','<U+0001F6AB>':'\U0001F6AB','<U+0001F62C>':'\U0001F62C','<U+0001F60F>':'\U0001F60F','<U+0001F4C3>':'\U0001F4C3','<U+0001F414>':'\U0001F414','<U+000E0062>':'\U000E0062','<U+0001F347>':'\U0001F347','<U+0001F60B>':'\U0001F60B','<U+0001F507>':'\U0001F507','<U+0001F923>':'\U0001F923','<U+0001F924>':'\U0001F924','<U+0001F601>':'\U0001F601','<U+0001F3F9>':'\U0001F3F9','<U+0001F622>':'\U0001F622','<U+0001F645>':'\U0001F645','<U+0001F311>':'\U0001F311','<U+0001F921>':'\U0001F921','<U+0001F7E2>':'\U0001F7E2','<U+0001F31A>':'\U0001F31A','<U+0001F3AA>':'\U0001F3AA','<U+0001F575>':'\U0001F575','<U+0001F438>':'\U0001F438','<U+0001FA93>':'\U0001FA93','<U+0001F3FF>':'\U0001F3FF','<U+0001F315>':'\U0001F315','<U+0001F4DE>':'\U0001F4DE','<U+0001F4C9>':'\U0001F4C9','<U+0001F364>':'\U0001F364','<U+0001FAD0>':'\U0001FAD0','<U+0001F965>':'\U0001F965','<U+0001F36C>':'\U0001F36C','<U+0001F92A>':'\U0001F92A','<U+0001F644>':'\U0001F644','<U+0001F479>':'\U0001F479','<U+0001F3A4>':'\U0001F3A4','<U+0001F443>':'\U0001F443','<U+0001F420>':'\U0001F420','<U+0001F449>':'\U0001F449','<U+0001F44B>':'\U0001F44B','<U+0001F97A>':'\U0001F97A','<U+000E007F>':'\U000E007F','<U+0001F31F>':'\U0001F31F','<U+0001F537>':'\U0001F537','<U+0001F605>':'\U0001F605','<U+0001F1FE>':'\U0001F1FE','<U+0001F64F>':'\U0001F64F','<U+0001F699>':'\U0001F699','<U+0001F9DF>':'\U0001F9DF','<U+0001F3FB>':'\U0001F3FB','<U+0001F590>':'\U0001F590','<U+0001F635>':'\U0001F635','<U+0001F314>':'\U0001F314','<U+0001F6E9>':'\U0001F6E9','<U+0001F3B6>':'\U0001F3B6','<U+0001F6A9>':'\U0001F6A9','<U+0001F4F0>':'\U0001F4F0','<U+0001F4BB>':'\U0001F4BB','<U+0001F95C>':'\U0001F95C','<U+0001F682>':'\U0001F682','<U+0001F914>':'\U0001F914','<U+0001F1EE>':'\U0001F1EE','<U+0001F987>':'\U0001F987','<U+0001F62B>':'\U0001F62B','<U+0001F494>':'\U0001F494','<U+0001F9B4>':'\U0001F9B4','<U+0001F615>':'\U0001F615','<U+0001FA99>':'\U0001FA99','<U+0001F968>':'\U0001F968','<U+0001F9E0>':'\U0001F9E0','<U+0001F5A4>':'\U0001F5A4','<U+0001F624>':'\U0001F624','<U+0001F61C>':'\U0001F61C','<U+0001F947>':'\U0001F947','<U+0001F436>':'\U0001F436','<U+0001F447>':'\U0001F447','<U+0001F1F0>':'\U0001F1F0','<U+0001F539>':'\U0001F539','<U+0001F932>':'\U0001F932','<U+0001F1F8>':'\U0001F1F8','<U+0001F4C4>':'\U0001F4C4','<U+0001F6F0>':'\U0001F6F0','<U+0001F49C>':'\U0001F49C','<U+0001F916>':'\U0001F916','<U+0001F608>':'\U0001F608','<U+0001F412>':'\U0001F412','<U+0001FAA6>':'\U0001FAA6','<U+0001F918>':'\U0001F918','<U+0001F39E>':'\U0001F39E','<U+0001F419>':'\U0001F419','<U+0001F368>':'\U0001F368','<U+0001F6E5>':'\U0001F6E5','<U+0001F31B>':'\U0001F31B','<U+0001F1F7>':'\U0001F1F7','<U+0001F6AC>':'\U0001F6AC','<U+0001F3E6>':'\U0001F3E6','<U+0001F549>':'\U0001F549','<U+0001F975>':'\U0001F975','<U+0001F929>':'\U0001F929','<U+0001F48D>':'\U0001F48D','<U+0001F5FC>':'\U0001F5FC','<U+0001F32D>':'\U0001F32D','<U+0001F691>':'\U0001F691','<U+0001F9C3>':'\U0001F9C3','<U+0001F49A>':'\U0001F49A','<U+0001F47F>':'\U0001F47F','<U+0001F570>':'\U0001F570','<U+0001F47D>':'\U0001F47D','<U+0001F680>':'\U0001F680','<U+0001F3C3>':'\U0001F3C3','<U+0001F60A>':'\U0001F60A','<U+0001F984>':'\U0001F984','<U+0001F4AD>':'\U0001F4AD','<U+0001F346>':'\U0001F346','<U+0001F451>':'\U0001F451','<U+0001F943>':'\U0001F943','<U+0001F969>':'\U0001F969','<U+0001F4DC>':'\U0001F4DC','<U+0001F4DD>':'\U0001F4DD','<U+0001FA80>':'\U0001FA80','<U+0001F3A5>':'\U0001F3A5','<U+0001F4DA>':'\U0001F4DA','<U+0001F446>':'\U0001F446','<U+0001F36F>':'\U0001F36F','<U+0001F4FA>':'\U0001F4FA','<U+0001F529>':'\U0001F529','<U+0001F3FE>':'\U0001F3FE','<U+000E0063>':'\U000E0063','<U+0001F604>':'\U0001F604','<U+0001F62F>':'\U0001F62F','<U+0001F1F9>':'\U0001F1F9','<U+0001F4AC>':'\U0001F4AC','<U+0001F6CC>':'\U0001F6CC','<U+0001F61E>':'\U0001F61E','<U+0001F9B6>':'\U0001F9B6','<U+0001F4F6>':'\U0001F4F6','<U+0001F340>':'\U0001F340','<U+0001F33A>':'\U0001F33A','<U+0001F402>':'\U0001F402','<U+0001F30A>':'\U0001F30A','<U+0001F413>':'\U0001F413','<U+0001F627>':'\U0001F627','<U+0001F6CD>':'\U0001F6CD','<U+0001F996>':'\U0001F996','<U+0001F192>':'\U0001F192','<U+0001F632>':'\U0001F632','<U+0001F91A>':'\U0001F91A','<U+0001F91C>':'\U0001F91C','<U+0001F44F>':'\U0001F44F','<U+0001F33F>':'\U0001F33F','<U+0001F50E>':'\U0001F50E','<U+0001F4A8>':'\U0001F4A8','<U+0001F610>':'\U0001F610','<U+0001F954>':'\U0001F954','<U+0001F370>':'\U0001F370','<U+0001F9B0>':'\U0001F9B0','<U+0001F6F8>':'\U0001F6F8','<U+0001F4BC>':'\U0001F4BC','<U+0001F4A3>':'\U0001F4A3','<U+0001F4B2>':'\U0001F4B2','<U+0001F6CE>':'\U0001F6CE','<U+0001F9A6>':'\U0001F9A6','<U+0001F3E0>':'\U0001F3E0','<U+0001F3AF>':'\U0001F3AF','<U+0001F602>':'\U0001F602','<U+0001F6E1>':'\U0001F6E1','<U+0001F90F>':'\U0001F90F','<U+0001F611>':'\U0001F611','<U+0001F986>':'\U0001F986','<U+0001F4AF>':'\U0001F4AF','<U+0001F4B6>':'\U0001F4B6','<U+0001F37D>':'\U0001F37D','<U+0001F1F2>':'\U0001F1F2','<U+0001F974>':'\U0001F974','<U+0001F92E>':'\U0001F92E','<U+0001F94A>':'\U0001F94A','<U+0001F614>':'\U0001F614','<U+0001F53B>':'\U0001F53B','<U+0001FA78>':'\U0001FA78','<U+0001F62D>':'\U0001F62D','<U+0001F623>':'\U0001F623','<U+0001F30E>':'\U0001F30E','<U+0001F911>':'\U0001F911','<U+0001F60D>':'\U0001F60D','<U+0001F43F>':'\U0001F43F','<U+0001F9F8>':'\U0001F9F8','<U+0001F552>':'\U0001F552','<U+0001F98B>':'\U0001F98B','<U+0001F6EB>':'\U0001F6EB','<U+0001F3AE>':'\U0001F3AE','<U+0001F1ED>':'\U0001F1ED','<U+0001F44D>':'\U0001F44D','<U+0001F6F6>':'\U0001F6F6','<U+0001F366>':'\U0001F366','<U+0001F9FB>':'\U0001F9FB','<U+0001F44A>':'\U0001F44A','<U+0001F485>':'\U0001F485','<U+0001F61D>':'\U0001F61D','<U+0001F607>':'\U0001F607','<U+0001F6D1>':'\U0001F6D1','<U+0001F91F>':'\U0001F91F','<U+0001F431>':'\U0001F431','<U+0001F34B>':'\U0001F34B','<U+0001F52D>':'\U0001F52D','<U+0001F30C>':'\U0001F30C','<U+0001F48B>':'\U0001F48B','<U+0001F609>':'\U0001F609','<U+0001F3EA>':'\U0001F3EA','<U+0001F951>':'\U0001F951','<U+0001F687>':'\U0001F687','<U+0001F9FC>':'\U0001F9FC','<U+0001F48E>':'\U0001F48E','<U+0001F4B0>':'\U0001F4B0','<U+0001F44E>':'\U0001F44E','<U+0001F34C>':'\U0001F34C','<U+0001F37A>':'\U0001F37A','<U+0001F92F>':'\U0001F92F','<U+0001F978>':'\U0001F978','<U+0001F4C8>':'\U0001F4C8','<U+0001F37F>':'\U0001F37F','<U+0001F596>':'\U0001F596','<U+0001F403>':'\U0001F403','<U+0001F3E2>':'\U0001F3E2','<U+0001F64A>':'\U0001F64A','<U+0001F47B>':'\U0001F47B','<U+0001F3B5>':'\U0001F3B5','<U+0001F355>':'\U0001F355','<U+0001F384>':'\U0001F384','<U+0001F437>':'\U0001F437','<U+0001F913>':'\U0001F913','<U+0001F198>':'\U0001F198','<U+0001F317>':'\U0001F317','<U+0001F3F3>':'\U0001F3F3','<U+0001F5E1>':'\U0001F5E1','<U+0001F409>':'\U0001F409','<U+0001F40E>':'\U0001F40E','<U+0001F4A6>':'\U0001F4A6','<U+0001F9BC>':'\U0001F9BC','<U+0001F1FA>':'\U0001F1FA','<U+0001F30B>':'\U0001F30B','<U+0001F4B5>':'\U0001F4B5','<U+0001F31D>':'\U0001F31D','<U+0001F4B4>':'\U0001F4B4','<U+0001F972>':'\U0001F972','<U+0001F69A>':'\U0001F69A','<U+0001F937>':'\U0001F937','<U+0001F43B>':'\U0001F43B','<U+0001F51C>':'\U0001F51C','<U+0001F973>':'\U0001F973','<U+0001F636>':'\U0001F636','<U+0001F308>':'\U0001F308','<U+0001F313>':'\U0001F313','<U+0001F58D>':'\U0001F58D','<U+0001F950>':'\U0001F950','<U+0001F33D>':'\U0001F33D','<U+0001F52A>':'\U0001F52A','<U+0001F920>':'\U0001F920','<U+0001F199>':'\U0001F199','<U+0001F445>':'\U0001F445','<U+0001F1E9>':'\U0001F1E9','<U+000E0067>':'\U000E0067','<U+000E0074>':'\U000E0074','<U+0001F1E8>':'\U0001F1E8','<U+0001F630>':'\U0001F630','<U+0001F9A7>':'\U0001F9A7','<U+0001F1F3>':'\U0001F1F3','<U+000E0073>':'\U000E0073','<U+0001F4AA>':'\U0001F4AA','<U+0001F4AB>':'\U0001F4AB','<U+0001F3FC>':'\U0001F3FC','<U+0001F917>':'\U0001F917','<U+0001F697>':'\U0001F697','<U+0001F41D>':'\U0001F41D','<U+0001F61B>':'\U0001F61B','<U+0001F4F7>':'\U0001F4F7','<U+0001F52E>':'\U0001F52E','<U+0001F1EB>':'\U0001F1EB','<U+0001F441>':'\U0001F441','<U+0001F9D0>':'\U0001F9D0','<U+0001F30F>':'\U0001F30F','<U+0001F618>':'\U0001F618','<U+0001F994>':'\U0001F994','<U+0001F535>':'\U0001F535','<U+0001F9E4>':'\U0001F9E4','<U+0001F629>':'\U0001F629','<U+0001F942>':'\U0001F942','<U+0001F933>':'\U0001F933','<U+0001F9E8>':'\U0001F9E8','<U+0001F985>':'\U0001F985','<U+0001F316>':'\U0001F316','<U+0001F9E2>':'\U0001F9E2','<U+0001F448>':'\U0001F448','<U+0001F386>':'\U0001F386','<U+0001F621>':'\U0001F621','<U+0001F496>':'\U0001F496','<U+0001F91E>':'\U0001F91E','<U+0001F6EC>':'\U0001F6EC','<U+0001F91D>':'\U0001F91D','<U+0001F595>':'\U0001F595','<U+0001F53A>':'\U0001F53A','<U+0001F1E7>':'\U0001F1E7','<U+0001F9C0>':'\U0001F9C0','<U+0001F976>':'\U0001F976','<U+0001F41C>':'\U0001F41C','<U+0001F60E>':'\U0001F60E','<U+0001F3AB>':'\U0001F3AB','<U+0001F9F4>':'\U0001F9F4','<U+0001F444>':'\U0001F444','<U+0001F514>':'\U0001F514','<U+0001F49B>':'\U0001F49B','<U+0001F603>':'\U0001F603','<U+0001F318>':'\U0001F318','<U+0001F52B>':'\U0001F52B','<U+0001F31E>':'\U0001F31E'}

def sub_emojis(text):
    for key, value in emojis_mapping.items():
        text = text.replace(key, value)
    return text

def prepare_df(df, exclude_list=None, sample_size=None, without_na=True):
    if without_na:
        df = df.loc[df['selftext'] != '[deleted]']
        df = df.loc[df['selftext'].notna()]
    if exclude_list:
        df = df[~df['id'].isin(exclude_list)]
    if sample_size:
        df_sample = df.sample(n=sample_size)
    else:
        df_sample = df
    df_sample['title'] = df_sample['title'].apply(lambda v: str(v) if not pd.isnull(v) else '')
    df_sample['selftext'] = df_sample['selftext'].apply(lambda v: str(v) if not pd.isnull(v) else '')
    df_sample['selftext'] = df_sample['selftext'].apply(lambda v: str(v) if v != '[deleted]' else '')
    df_sample['full_text'] = df_sample['title'] + ': ' + df_sample['selftext']
    df_sample['text_processed'] = df_sample['full_text'].apply(lambda x: sub_emojis(x) if isinstance(x, str) else x)
    df_sample['text_processed'] = df_sample['text_processed'].apply(lambda x: x.replace('<U+0092>', "'") if isinstance(x, str) else x)
    df_sample["text_cleaned"] = df_sample['full_text'].map(redditcleaner.clean)
    df_sample_evaluate = df_sample[['id', 'Ticker', 'full_text', 'text_processed','text_cleaned']]
    return df_sample_evaluate