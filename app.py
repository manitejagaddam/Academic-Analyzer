import streamlit as st
from PIL import Image
import asyncio
import tempfile

from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import Image as AGImage

# Import your agents
from agents.answers_extractor import answer_extractor
from agents.answers_analyser import answer_analyser

# Page config
st.set_page_config(
    page_title="🧠 Multi-Agent Image Analyzer",
    layout="centered"
)

# Initialize agent team
team = RoundRobinGroupChat(
    participants=[answer_extractor(), answer_analyser()],
    max_turns=2
)

# Async image processor
async def process_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    pil_image = Image.open(temp_path)
    ag_image = AGImage(pil_image)

    task = MultiModalMessage(content=["give the response accordingly", ag_image], source='user')
    result = await team.run(task=task)
    return result.messages

# Streamlit UI
st.title("🧠 Multi-Agent Image Analyzer")

uploaded_file = st.file_uploader("📁 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            try:
                result = asyncio.run(process_image(uploaded_file))
                st.success("✅ Done!")

                st.subheader("📋 Final Output")
                for i, msg in enumerate(result):
                    sender = getattr(msg, "sender", None)
                    sender_name = sender.name if sender else "Unknown"

                    content = msg.content
                    st.markdown(f"**{sender_name}:**")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, str):
                                st.markdown(part)
                    else:
                        st.markdown(str(content))

            except Exception as e:
                st.error(f"❌ Error: {e}")
