import streamlit as st
import preprocessor,helper

st.sidebar.title("Whatsapp chat analyzer")


uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    st.text(data)
    df = preprocessor.preprocessor(data)

    st.dataframe(df)


#fetching using users
    user_list=df['user'].unique().tolist()
    #removing group notification
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user=st.sidebar.selectbox("Show ananlysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages,words,num_media_messages=helper.fetch_stats(selected_user,df)

        col1,col2,col3,col4=st.columns(4)

        with col1:
            st.header("Total messages")
            st.title(num_messages)

        with col2:
            st.header("Words")
            st.title(words)

        with col3:
            st.header("Media messages")
            st.title(num_media_messages)


