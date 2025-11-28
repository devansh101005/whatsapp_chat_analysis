import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
from streamlit.components.v1 import html
import networkx as nx

st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat file")

if uploaded_file is not None:
    data = uploaded_file.getvalue().decode("utf-8")
    df = preprocessor.preprocessor(data)

    # User selection
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for", user_list, key="user_select")

    # Page selection
    page = st.sidebar.selectbox(
        "Select Feature",
        [
            "Chat Statistics",
            "Sentiment Analysis",
            "Topic Modeling",
            "Toxicity Detection",
            "Social Network Graph"
        ],
        key="page_select"
    )

    # ---------------------- PAGE 1: CHAT STATISTICS ---------------------- #
    st.title("DataSet Used For Logistic Regression and SVM Classifier")
    st.subheader("https://www.kaggle.com/datasets/kazanova/sentiment140")

    if page == "Chat Statistics":
        st.title("Chat Statistics & Activity")

        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Messages")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly timeline
        st.subheader("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        if not timeline.empty:
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        else:
            st.info("Not enough data for monthly timeline.")

        # Daily timeline
        st.subheader("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        if not daily_timeline.empty:
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        else:
            st.info("Not enough data for daily timeline.")

        # Activity map
        st.subheader("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.caption("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            if not busy_day.empty:
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            else:
                st.info("No data for busy day chart.")

        with col2:
            st.caption("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            if not busy_month.empty:
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            else:
                st.info("No data for busy month chart.")

        # Weekly heatmap
        st.subheader("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        if user_heatmap.empty:
            st.warning("Not enough data to show heatmap.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, ax=ax)
            st.pyplot(fig)

        # Most busy users (group level)
        if selected_user == "Overall":
            st.subheader("Most Busy Users in Group")
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # Wordcloud
        st.subheader("WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        ax.axis("off")
        st.pyplot(fig)

        # Most common words
        st.subheader("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Emoji
        st.subheader("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            if not emoji_df.empty:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f%%")
                st.pyplot(fig)
            else:
                st.info("No emojis found.")

    # ---------------------- PAGE 2: SENTIMENT ANALYSIS ---------------------- #
    elif page == "Sentiment Analysis":
        st.title("Sentiment Analysis")

        model_choice = st.sidebar.radio(
            "Select Sentiment Model",
            ("Logistic Regression", "SVM", "BERT"),
            key="sentiment_model_choice"
        )

        if model_choice == "Logistic Regression":
            sentiment_result = helper.ml_sentiment(selected_user, df)
        elif model_choice == "SVM":
            sentiment_result = helper.svm_sentiment(selected_user, df)
        else:
            sentiment_result = helper.bert_sentiment(selected_user, df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            st.dataframe(sentiment_result)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(sentiment_result.values, labels=sentiment_result.index, autopct="%0.2f%%")
            st.pyplot(fig)

        # Optional: sentiment trend over time if you added it
        try:
            from helper import sentiment_timeline
            st.subheader("Sentiment Trend Over Time")
            mood_timeline = sentiment_timeline(selected_user, df, model_choice)
            if not mood_timeline.empty:
                fig, ax = plt.subplots()
                ax.plot(mood_timeline['only_date'], mood_timeline['sentiment_score'], marker='o')
                plt.xticks(rotation='vertical')
                plt.ylabel("Average Sentiment Score")
                plt.grid(True)
                st.pyplot(fig)
            else:
                st.info("Not enough data to show sentiment timeline.")
        except Exception:
            pass


    # ---------------------- PAGE: TOPIC MODELING ---------------------- #
    elif page == "Topic Modeling":
        st.title("Topic Modeling ")

        st.caption(
            "This module uses Latent Dirichlet Allocation (LDA) "
            "to discover hidden discussion themes in the WhatsApp chat."
        )

        num_topics = st.slider(
            "Select number of topics",
            min_value=2,
            max_value=8,
            value=5
        )

        topics = helper.topic_modeling(selected_user, df, num_topics=num_topics)

        if len(topics) == 0:
            st.warning("Not enough data to generate topics.")
        else:
            for i, topic in topics:
                st.subheader(f"Topic {i + 1}")
                st.write(topic)


    # ---------------------- PAGE 3: TOXICITY DETECTION (INNOVATION) ---------------------- #
    elif page == "Toxicity Detection":
        st.title("Toxicity & Abuse Detection")

        tox_df, summary = helper.toxicity_analysis(selected_user, df)

        if tox_df.empty:
            st.warning("No messages found to analyze toxicity for this selection.")
        else:
            st.subheader("Average Toxicity by Category")
            st.dataframe(summary)

            fig, ax = plt.subplots()
            ax.bar(summary.index, summary.values)
            plt.xticks(rotation='vertical')
            plt.ylabel("Average Toxicity Score (0-1)")
            st.pyplot(fig)

            st.subheader("Most Toxic Messages")
            top_toxic = tox_df.sort_values("max_toxic", ascending=False).head(10)[["message", "max_toxic"]]
            st.dataframe(top_toxic)

    # ---------------------- PAGE 4: SOCIAL NETWORK GRAPH (INNOVATION) ---------------------- #
    elif page == "Social Network Graph":
        st.title("Social Network Graph of Conversations")

        # For social graph, always use full chat (group-level insight)
        G = helper.build_conversation_graph(df)

        if len(G.nodes) == 0:
            st.warning("Not enough data to build a conversation graph.")
        else:
            st.subheader("Static Network Graph (NetworkX)")

            fig, ax = plt.subplots(figsize=(8, 5))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            nx.draw(
                G, pos, with_labels=True, ax=ax,
                node_size=2000, font_size=10,
                width=[w * 0.2 for w in weights]
            )
            st.pyplot(fig)

            st.subheader("Interactive Network Graph (PyVis)")

            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
            net.from_nx(G)
            net.repulsion(node_distance=200, central_gravity=0.33, spring_length=110, spring_strength=0.10)

            net.save_graph("chat_network.html")
            with open("chat_network.html", "r", encoding="utf-8") as f:
                html_data = f.read()
            html(html_data, height=600, scrolling=True)

else:
    st.write(" Upload a WhatsApp chat file (.txt) from 'Export Chat' to start analysis.")
