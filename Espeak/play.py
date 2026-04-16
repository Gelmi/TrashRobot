import asyncio
import edge_tts
import subprocess

async def speak(text: str, voice: str = "en-US-JennyNeural"):
    communicate = edge_tts.Communicate(text, voice)
    proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
        stdin=subprocess.PIPE
    )
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            proc.stdin.write(chunk["data"])
    proc.stdin.close()
    proc.wait()

asyncio.run(speak("Hello! This plays instantly without saving any file."))
