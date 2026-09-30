import re


import pandas as pd


def preprocessor(data):
    # WhatsApp exports come in two common formats:
    #   Android:  "12/25/23, 2:30 PM - User: message"   (time can also be 24-hour like 14:30)
    #   iPhone :  "[25/12/23, 2:30:45 PM] User: message"
    # The patterns also allow 2 or 4 digit years and an optional AM/PM.
    android_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[APap][Mm])?\s-\s'
    ios_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?[APap][Mm]\]\s'

    # Use whichever format matches the uploaded file.
    # iPhone dates are usually day-first, Android (US export) is month-first.
    if re.search(ios_pattern, data):
        pattern = ios_pattern
        dayfirst = True
    else:
        pattern = android_pattern
        dayfirst = False

    messages= re.split(pattern,data)[1:]
    dates=re.findall(pattern,data)

    df=pd.DataFrame({'user_message':messages,'message_date':dates})
    #df['message_date'] = pd.to_datetime(df['message_date'].str.strip(), format='%m/%d/%y, %H:%M -', errors='coerce')

    # Clean the date text: drop the brackets and the " - " separator so pandas can read it
    df['message_date'] = df['message_date'].astype(str)
    df['message_date'] = df['message_date'].str.replace('[', '', regex=False)
    df['message_date'] = df['message_date'].str.replace(']', '', regex=False)
    df['message_date'] = df['message_date'].str.replace(' - ', '', regex=False)
    df['message_date'] = df['message_date'].str.strip()

    # Let pandas parse the date (works for both 24-hour and AM/PM times)
    df['message_date'] = pd.to_datetime(df['message_date'], format='mixed', dayfirst=dayfirst, errors='coerce')
    df = df.dropna(subset=['message_date'])

    df.rename(columns={'message_date':'date'},inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'^(.*?):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df.drop(columns=['user_message'], inplace=True)

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period



    return df