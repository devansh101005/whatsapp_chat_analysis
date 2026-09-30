import pandas as pd
import pytest

# helper.py imports heavy libraries (gensim, transformers, torch...).
# If they are not installed, skip these tests instead of failing.
helper = pytest.importorskip("helper")


def make_df():
    return pd.DataFrame({
        'user': ['A', 'B', 'A', 'group_notification'],
        'message': ['hello world', 'hi there', 'https://example.com', 'A joined'],
    })


def test_fetch_stats_counts():
    df = make_df()
    num_messages, words, media, links = helper.fetch_stats("Overall", df)
    assert num_messages == 4
    assert links == 1  # one URL in the messages


def test_slang_sentiment_positive():
    # Positive Hinglish words should give a positive score
    assert helper.slang_sentiment("mast badiya") > 0


def test_slang_sentiment_negative():
    # Negative Hinglish words should give a negative score
    assert helper.slang_sentiment("bakwas ganda") < 0


def test_build_conversation_graph_edges():
    df = make_df()
    G = helper.build_conversation_graph(df)
    # A talked, then B replied
    assert ("A", "B") in G.edges()
    # group_notification rows should be ignored
    assert "group_notification" not in G.nodes()
