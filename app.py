import json
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient import discovery
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

all_comments = []

# Function to get replies in JSON-like structure
def get_replies(comment_thread_id):
    replies = []
    request = youtube.comments().list(
        part="snippet",
        parentId=comment_thread_id,
        maxResults=100
    )

    while request:
        response = request.execute()
        for item in response.get('items', []):
            snippet = item['snippet']
            replies.append({
                "type": "reply",
                "author": snippet['authorDisplayName'],
                "publishedAt": snippet['publishedAt'],
                "likes": snippet['likeCount'],
                "text": snippet['textOriginal']
            })
        request = youtube.comments().list_next(request, response)

    return replies
# Function to get all comments and replies
def get_comments(video_id):
    comments_json = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )

    while request:
        response = request.execute()
        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            total_reply_count = item['snippet']['totalReplyCount']

            comment_obj = {
                "type": "comment",
                "id": item['id'],
                "author": snippet['authorDisplayName'],
                "publishedAt": snippet['publishedAt'],
                "likes": snippet['likeCount'],
                "text": snippet['textOriginal'],
                "replies": []
            }

            if total_reply_count > 0:
                comment_obj["replies"] = get_replies(item['id'])

            comments_json.append(comment_obj)

        request = youtube.commentThreads().list_next(request, response)

    return comments_json
# Function to flatten the comment structure
def flatten_comment_structure(comment_list):
    text_block = ""

    for comment in comment_list:
        if isinstance(comment, dict):
            author = comment.get("author", "Unknown")
            published = comment.get("published_at", "Unknown date")
            text = comment.get("text", "")
            likes = comment.get("likes", 0)

            text_block += f"🟦 Comment by {author} on {published} (Likes: {likes}):\n{text}\n\n"

            # Handle replies if present
            replies = comment.get("replies", [])
            if isinstance(replies, list) and replies:
                for reply in replies:
                    reply_author = reply.get("author", "Unknown")
                    reply_published = reply.get("published_at", "Unknown date")
                    reply_text = reply.get("text", "")
                    reply_likes = reply.get("likes", 0)

                    text_block += f"    ↳ Reply by {reply_author} on {reply_published} (Likes: {reply_likes}):\n    {reply_text}\n\n"

    return text_block

# Function to vector embedding
def vector_embedding(GEMINI_API_KEY, transcript_text):
    if "vectors" not in st.session_state:
        # Use Google Generative AI Embeddings for the model
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GEMINI_API_KEY)

        with open("data/youtube_comments.json", "r") as f:
            json_data = json.load(f)
        full_text = flatten_comment_structure(json_data)
        context = f"video transcript: {transcript_text}\n\n, comments: {full_text}"

        document = Document(page_content=context, metadata={"source": "video transcript and comments"})
        # Split the documents into chunks for easier processing
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = splitter.split_documents([document])
        # Create vector embeddings using FAISS
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        if st.session_state.vectors:
            st.session_state.completed_vector_embedding = True
            st.success("Vector Store DB is ready!")
        else:
            st.session_state.completed_vector_embedding = False
            st.warning("⚠️ Vector Store DB is not ready yet, please check the API keys and video URL.")

        

if "required_data_collected" not in st.session_state:
    st.session_state.required_data_collected = False
if "api_keys_collected" not in st.session_state:
    st.session_state.api_keys_collected = False
if "video_setup_collected" not in st.session_state:
    st.session_state.video_setup_collected = False
if "Video_title" not in st.session_state:
    st.session_state.Video_title = None
if "video_description" not in st.session_state:
    st.session_state.video_description = None
if "channel_title" not in st.session_state:
    st.session_state.channel_title = None
if "completed_vector_embedding" not in st.session_state:
    st.session_state.completed_vector_embedding = False
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""

st.title("VIEWER LENS")
st.write("See What Your Audience Thinks and Asks")
st.caption("This application allows you to ask questions about a YouTube video, and it will provide answers based on the video transcript and user comments.")


st.divider()

with st.sidebar:
    st.header("🔐 Enter Your API Keys")
    st.session_state.YT_API_KEY = st.text_input("YouTube API Key", type="password")
    st.session_state.GROQ_API_KEY = st.text_input("Groq API Key", type="password")
    st.session_state.GEMINI_API_KEY = st.text_input("Google Gemini API Key", type="password")
    api_submit = st.button("Save API Keys")
    if api_submit:
        required_keys = ["YT_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"]
        missing_keys = [key for key in required_keys if key not in st.session_state or not st.session_state[key]]

        if missing_keys:
            st.warning(f"⚠️ Please Enter all Api keys before proceeding: {', '.join(missing_keys)}")
        else:
            st.success("✅ API Keys saved successfully!")
            st.session_state.api_keys_collected = True
    st.header("🔗 Enter Video and Model Details")
    st.session_state.URL = st.text_input("Enter the YouTube Video URL", value="")
    st.session_state.MODEL_NAME = st.text_input("Model Name", value="")
    video_submit = st.button("Save Video Setup")
    if video_submit:
        required_keys = ["URL", "MODEL_NAME"]
        missing_keys = [key for key in required_keys if key not in st.session_state or not st.session_state[key]]

        if missing_keys:
            st.warning(f"⚠️ Please Enter all Video details before proceeding: {', '.join(missing_keys)}")
        else:
            st.success("✅ Video and model info saved!")
            st.session_state.video_setup_collected = True



if st.session_state.required_data_collected and st.session_state.api_keys_collected and st.session_state.video_setup_collected:

    st.subheader(st.session_state.channel_title+" : "+ st.session_state.Video_title)
    st.image(st.session_state.thumbnail_url)          

    # Define the model
    llm = ChatGroq(groq_api_key=st.session_state.GROQ_API_KEY, model_name=st.session_state.MODEL_NAME)

    # Define the prompt template that now accepts a single "context" variable
    prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the context provided below. Consider both the video transcript and user comments to provide an accurate and well-rounded answer.

    <Context>
    {context}

    Questions: {input}
    """
    )

    # User input for asking questions
    prompt1 = st.text_input("Enter Your Question About the Video")


    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        
        st.write("Response time: ", time.process_time() - start)
        st.write(response['answer'])
        
        # Show similar documents that were used for context (from the retrieval)
        st.subheader("🔍 Top Relevant Chunks")
        similar_docs = retriever.get_relevant_documents(prompt1)
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(similar_docs):
                st.write(doc.page_content)
                st.write("--------------------------------")

st.divider()

st.subheader("Document Embedding and Fetching Details")
st.write("Click the button below to fetch the transcript and comments for the YouTube video, and then create vector embeddings for them.")
st.caption("This will allow the application to answer questions based on the video content and user comments.")

Fetch_details, Vector_embedding = st.columns(2)

with Fetch_details:
    fetch_btn = st.button("📡 Fetch Transcript and Comments")
    if fetch_btn:
        if st.session_state.api_keys_collected and st.session_state.video_setup_collected:

            try:
                # Extract the video ID from the URL
                YT_VIDEOID = st.session_state.URL.split("v=")[-1]
                # Initialize the YouTube API client
                api_service_name = "youtube"
                api_version = "v3"
                youtube = discovery.build(
                    api_service_name, api_version, developerKey=st.session_state.YT_API_KEY)
                # Request to get video details, including the title.
                video_request = youtube.videos().list(
                    part="snippet",
                    id=YT_VIDEOID
                )

                # Execute the video request.
                aboutvideo_response = video_request.execute()
                with open('/Users/gautambr/Documents/Machine Learning Projects/ViewerLens Application/data/About.json', "w", encoding="utf-8") as f:
                    json.dump(aboutvideo_response, f, indent=4, ensure_ascii=False)
                # Extract the title.
                st.session_state.Video_title = aboutvideo_response['items'][0]['snippet']['title']
                # Extract the description.
                st.session_state.video_description = aboutvideo_response['items'][0]['snippet']['description']
                # Extract the channel title.
                st.session_state.channel_title = aboutvideo_response['items'][0]['snippet']['channelTitle']

                st.session_state.thumbnail_url = aboutvideo_response['items'][0]['snippet']['thumbnails']['maxres']['url']
            
            except Exception as e:
                st.error(f"⚠️ Error fetching video details: {e}")
                st.session_state.required_data_collected = False
                st.session_state.video_setup_collected = False

            try:
                transcript = YouTubeTranscriptApi.get_transcript(YT_VIDEOID)
                with open(f"data/transcript.txt", "w") as file:
                    st.session_state.transcript_text = ""
                    for line in transcript:
                        file.write(f"{line['text']} ")
                        st.session_state.transcript_text += f"(line['text']) "

                # Function to get comments from a YouTube video
                youtube = discovery.build(api_service_name, api_version, developerKey=st.session_state.YT_API_KEY)
            except Exception as e:
                st.error(f"⚠️ Error fetching transcript: {e}")
                st.session_state.required_data_collected = False
            
            try:
                # Fetch and save to JSON
                all_comments = get_comments(YT_VIDEOID)
                with open("data/youtube_comments.json", "w", encoding="utf-8") as f:
                    json.dump(all_comments, f, indent=2, ensure_ascii=False)

                st.success("✅ All Required Data is Collected!")
                st.session_state.required_data_collected = True
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Error fetching comments: {e}")
                st.session_state.required_data_collected = False 
        else:
            st.warning("⚠️ Please Enter all the API keys, video URL, and model name before proceeding.")
    
with Vector_embedding:
        vector_embedding_btn = st.button("Documents Embedding")
        if vector_embedding_btn:
            vector_embedding(GEMINI_API_KEY=st.session_state.GEMINI_API_KEY, transcript_text=st.session_state.transcript_text)

st.caption("This is a YouTube Video Q&A application that uses the Groq API and Google Generative AI to answer questions based on video transcripts and user comments Using Retrieval-Augmented Generation (RAG) techniques.")       



