import re
import pandas as pd

def preprocessor(data):

    # Fix invisible unicode spaces before am/pm
    data = data.replace('\u202f', ' ').replace('\u200f', ' ')

    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}\s?[apAP][mM])\s-\s'

    messages = re.split(pattern, data)[1:]
    message_dates = []
    user_messages = []

    for i in range(0, len(messages), 3):
        date_time = f"{messages[i]} {messages[i+1]}"
        message = messages[i+2]
        message_dates.append(date_time)
        user_messages.append(message)

    df = pd.DataFrame({'date_time': message_dates, 'user_message': user_messages})

    # Convert datetime formats safely
    df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['date_time'])

    users = []
    msgs = []

    for msg in df['user_message']:
        parts = re.split(r'^(.*?):\s', msg, maxsplit=1)
        if len(parts) > 2:
            users.append(parts[1])
            msgs.append(parts[2])
        else:
            users.append("group_notification")
            msgs.append(parts[0])

    df['user'] = users
    df['message'] = msgs

    df['year'] = df['date_time'].dt.year
    df['month_num'] = df['date_time'].dt.month
    df['month'] = df['date_time'].dt.month_name()
    df['only_date'] = df['date_time'].dt.date
    df['day_name'] = df['date_time'].dt.day_name()
    df['hour'] = df['date_time'].dt.hour
    df['minute'] = df['date_time'].dt.minute
    df.drop(columns=['user_message'], inplace=True)

    df['period'] = df['hour'].apply(lambda x: f'{x}-{(x+1)%24}')

    return df
