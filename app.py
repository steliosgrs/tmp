import streamlit as st
from io import StringIO

title = st.selectbox(
    "Which exercise you want to test?",
    ("Exercise 1", "Exercise 2", "Exercise 3"),
    index=None,
    placeholder="Choose your exercise..."
    )
print(title)
with st.form("upload_form"):
    st.write("The current movie title is", title)
    uploaded_files = st.file_uploader("Upload Exercise", accept_multiple_files=True)

    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.write(stringio)

            # To read file as string:
            string_data = stringio.read()
            st.write(string_data)

        # st.write("filename:", uploaded_file.name)
        # st.write(bytes_data)

    # st.file_uploader("Upload Exercise")
    st.form_submit_button("Submit")