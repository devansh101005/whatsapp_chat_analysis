from urlextract import URLExtract
extract=URLExtract()


def fetch_stats(selected_user,df):

    # if(selected_user=='Overall'):
    #     #fetching number of messages
    #     num_messages=df.shape[0]
    #     #now fetching number of words
    #     words=[]
    #     for message in df['message']:
    #         words.extend(message.split())
    #     return num_messages,len(words)
    # else:
    #     new_df=df[df['user'] == selected_user]
    #     num_messages=new_df.shape[0]
    #     words=[]
    #     for message in new_df['message']:
    #         words.extend(message.split())
    #         return num_messages,len(words)

    #---------------Restructing code as above code is too bulky-------------#

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

        # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())


        # fetch number of media messages
        num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
        
        #fetch number of links shared
        links=[]
        for message in df['message']:
            links.extend(extract.find_urls(message))


        return num_messages,len(words),num_media_messages,len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df
