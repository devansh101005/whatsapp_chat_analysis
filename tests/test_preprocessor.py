import preprocessor

# A small sample chat in the same format the app expects
sample_chat = """12/25/23, 10:30 - John: Hello everyone
12/25/23, 10:31 - Alice: Hi John
12/25/23, 10:32 - John: <Media omitted>
12/25/23, 10:33 - Messages and calls are end-to-end encrypted.
"""


def test_preprocessor_returns_all_rows():
    df = preprocessor.preprocessor(sample_chat)
    # 4 lines should become 4 rows
    assert df.shape[0] == 4


def test_preprocessor_extracts_users():
    df = preprocessor.preprocessor(sample_chat)
    users = df['user'].tolist()
    assert "John" in users
    assert "Alice" in users


def test_system_message_becomes_group_notification():
    df = preprocessor.preprocessor(sample_chat)
    # The "end-to-end encrypted" line has no "user:" part
    assert "group_notification" in df['user'].tolist()


def test_preprocessor_has_time_columns():
    df = preprocessor.preprocessor(sample_chat)
    for col in ['year', 'month', 'day_name', 'hour', 'period']:
        assert col in df.columns


# A chat that uses 12-hour AM/PM time instead of 24-hour time
ampm_chat = """12/25/23, 2:30 PM - John: Afternoon message
12/25/23, 11:05 AM - Alice: Morning message
"""


def test_preprocessor_handles_am_pm():
    df = preprocessor.preprocessor(ampm_chat)
    assert df.shape[0] == 2
    # 2:30 PM should be stored as hour 14 (24-hour clock)
    assert 14 in df['hour'].tolist()


# A chat in the iPhone export format with square brackets
ios_chat = "[25/12/23, 2:30:45 PM] John: Hello from iPhone\n[25/12/23, 2:31:00 PM] Alice: Hi\n"


def test_preprocessor_handles_ios_format():
    df = preprocessor.preprocessor(ios_chat)
    assert df.shape[0] == 2
    assert "John" in df['user'].tolist()
