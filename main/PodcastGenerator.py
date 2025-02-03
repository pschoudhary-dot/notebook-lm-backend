from datetime import datetime
import os
import os
import time
from langchain_core.prompts import ChatPromptTemplate
import requests
from models import ModelManager
from voice_config import VOICES

class PodcastProcessor:
    def __init__(self, user_id: str, secret_key: str):
        self.model_manager = ModelManager()
        self.llm = self.model_manager.get_model("deepseek-r1-distill-llama-70b")
        self.user_id = user_id
        self.secret_key = secret_key
        self.headers = {
            'X-USER-ID': user_id,
            'Authorization': f'Bearer {secret_key}',
            'Content-Type': 'application/json',
        }

    async def process_podcast(self, document_content: str) -> str:
        """Generate podcast transcript from document content"""
        prompt = ChatPromptTemplate.from_template(
            """
            ### Podcast Dialogue Generator
            Create a compelling podcast conversation between two hosts that transforms complex content into engaging dialogue. Follow these guidelines:

            ## Host Personas
            1. **Dynamic Duo:**
            - Host 1: "The Enthusiast" - Curious, energetic, asks probing questions
            - Host 2: "The Expert" - Knowledgeable, witty, delivers key insights
            - Natural back-and-forth flow without interruptions

            ## Content Requirements
            1. **Conversation Style:**
            - Pure dialogue format: "Host 1:..." "Host 2:..." 
            - No segment titles, music cues, or formatting
            - Seamless topic transitions through dialogue

            2. **Content Adaptation:**
            - Convert technical terms using relatable analogies
            - Use "Imagine if..." explanations for complex ideas
            - Highlight 2-3 key insights from document

            ## Dialogue Rules
            1. **Flow & Structure:**
            - Alternate every 1-3 sentences
            - Use conversational connectors: "Right...", "But consider...", "Here's why..."
            - Include 3 audience engagement phrases per 500 words: "Ever wondered...", "Picture this..."
            - Create engaging and dependend sentences and also add human like interactions like hmm, okay, right, aah, got it, etc.

            2. **Tone Balance:**
            - 30 percent humor/references, 50 percent insights, 20 percent banter
            - Professional foundation with witty spikes
            - Example: "So it's like TikTok for neurons? (laughs) But seriously..."

            ## Technical Specs
            - Length: you are technically free to make it as long as you want but try to keep it around 12 mins to 22 mins
            - Complexity: Grade 8-10 reading level
            - Format: Strictly "Host 1: [text]" lines without empty lines

            ## Required Content
            {document_content}

            ## Anti-Requirements
            - No markdown/formatting
            - No section headers or labels
            - No monologues >3 sentences
            - No passive voice

            ## Example Format:
            Host 1: Welcome to The Tech Tomorrow Podcast! Today we're diving into AI voice technology.
            Host 2: And what a topic this is. The progress from basic commands to full conversations is staggering.
            Host 1: Remember when asking phones to set timers felt revolutionary?
            Host 2: Now AI understands context and nuance. But does that raise ethical questions?
            Host 1: Exactly! Where should we draw the privacy line with these voice assistants?

            ## Example format2:
            Speaker 1: Alright so, instead of our regularly scheduled programming, there's something hairier that came across my feed last night that I just need to discuss.
            Speaker 2: Wait, where is this going?
            Speaker 1: I uh, I just thought we'd take a little... detour. You know, take the scenic route down the path of mystery. Specifically, into the thick, mossy woods where something like, oh I don't know… *Bigfoot* himself might be lurking.
            Speaker 2: Oh, for the love of, again, Briggs? Really? We did this last month. And the month before that. This is basically the podcast equivalent of your sad karaoke go-to.
            Speaker 1: What? No! I'm just... providing the people what they want! Listen, There's new evidence, and the public demands more attention on it.
            Speaker 2: The "public" is just you, Briggs. You're the one emailing us suggestions under fake names. We've all seen "Biggie O Footlore" in our inbox.
            """
        )

        
        chain = prompt | self.llm
        response = await chain.ainvoke({"document_content": document_content})
        transcript = response.content
        
        self._save_transcript(transcript)
        return transcript

    def _save_transcript(self, transcript: str):
        """Save generated transcript to transcripts folder"""
        os.makedirs("transcripts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.txt"
        
        # Extract only the host dialogues
        host_transcript = []
        in_think_tag = False
        for line in transcript.splitlines():
            if "<think>" in line:
                in_think_tag = True
            elif "</think>" in line:
                in_think_tag = False
            elif not in_think_tag:
                host_transcript.append(line)
        
        host_transcript = "\n".join(host_transcript)
        
        with open(f"transcripts/{filename}", "w") as f:
            f.write(host_transcript)
        print(f"Transcript saved to transcripts/{filename}")

    def generate_audio(self, transcript: str, host1_voice: str, host2_voice: str) -> str:
        """Convert transcript to audio using specified voices"""
        # Get voice IDs from configuration
        voice1_id = VOICES.get(host1_voice, {}).get("id")
        voice2_id = VOICES.get(host2_voice, {}).get("id")
        
        if not voice1_id or not voice2_id:
            raise ValueError("Invalid voice selection")

        payload = {
            'model': 'PlayDialog',
            'text': transcript,
            'voice': voice1_id,
            'voice2': voice2_id,
            'turnPrefix': 'Host 1:',
            'turnPrefix2': 'Host 2:',
            'outputFormat': 'mp3',
        }

        response = requests.post(
            'https://api.play.ai/api/v1/tts/',
            headers=self.headers,
            json=payload
        )

        if response.status_code != 201:
            raise Exception(f"TTS API error: {response.text}")

        job_id = response.json()['id']
        return self._poll_audio_job(job_id)

    def _poll_audio_job(self, job_id: str) -> str:
        """Poll audio generation job until completion"""
        url = f'https://api.play.ai/api/v1/tts/{job_id}'
        while True:
            response = requests.get(url, headers=self.headers)
            if not response.ok:
                raise Exception(f"Status check failed: {response.text}")
                
            status = response.json().get('output', {}).get('status')
            if status == 'COMPLETED':
                return response.json()['output']['url']
            if status == 'FAILED':
                raise Exception("Audio generation failed")
                
            time.sleep(5)