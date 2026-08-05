"""Collect YouTube transcripts from football tactical analysis channels.

Uses yt-dlp (system install) for reliable subtitle extraction.
The youtube-transcript-api library is broken as of mid-2026 due to
YouTube's anti-scraping changes.
"""

import hashlib
import os
import re
import subprocess
import tempfile


# Curated tactical analysis videos from Tifo Football's "Tactics Explained" playlist
# and other channels. Each entry: (video_id, title, channel)
TACTICAL_VIDEOS = [
    # Tifo Football - tactical systems & concepts
    ("5QRWryN3okY", "What is Gegenpressing?", "Tifo Football"),
    ("t8jPSUQzyBE", "What is a False Nine?", "Tifo Football"),
    ("IoyjjlTipEE", "What is a Regista?", "Tifo Football"),
    ("cdGKolPn5Pk", "What is a Sweeper Keeper?", "Tifo Football"),
    ("kOFdKoGTwFA", "Enganche, Trequartista & Classic #10 Explained", "Tifo Football"),
    ("RNMeMa2OuI0", "Total Football Explained", "Tifo Football"),
    ("QNF91Aoe6eY", "The High Defensive Line", "Tifo Football"),
    ("ou44jxozOtc", "Catenaccio Explained", "Tifo Football"),
    ("BLMhylkO2eo", "Zonal & Man Marking Explained", "Tifo Football"),
    ("FvUxn0_7N_s", "Liverpool's Counter-Pressing Explained", "Tifo Football"),
    ("MlRXmjd95cM", "Three At The Back Explained", "Tifo Football"),
    ("sPVTxe5KbSQ", "What is a Raumdeuter?", "Tifo Football"),
    ("hGtq5FPpp08", "What is an Artificial Transition?", "Tifo Football"),
    ("YB_Gip6k_fA", "Zonal vs Man Marking", "Tifo Football"),
    ("_fTNwie0i8s", "The Return of the 4-4-2", "Tifo Football"),
    ("h9QpXX47Mo4", "Football Positions Are Evolving", "Tifo Football"),
    ("_8UjVeHSgOM", "How The Back Four Killed The Traditional Winger", "Tifo Football"),
    ("g0CceOu3sak", "Why Do Teams Play Out From The Back?", "Tifo Football"),
    ("dAbBlYQXl6M", "How Goal Kicks Are Changing Football", "Tifo Football"),
    ("cdzR9K7Ckjk", "Four Tactics No-One Uses Anymore", "Tifo Football"),
    ("Ru3KcF6OCNc", "What Actually is Rest Defence?", "Tifo Football"),
    ("29umGPRvhBA", "What's the Lavolpiana Build Up?", "Tifo Football"),
    # Tifo Football - team/manager tactical breakdowns
    ("lvkkYJ2YW-A", "Tifo's Guide to 4-3-3", "Tifo Football"),
    ("aH0JMFtpk0Y", "A Guide to 4-4-2", "Tifo Football"),
    ("hfNDVz1LdMI", "Tifo's Guide to 3-4-3", "Tifo Football"),
    ("wkhsGtK9uBI", "Why Teams Play With a Back Three", "Tifo Football"),
    ("KPmUmeEI3Qo", "Atalanta Tactics Explained", "Tifo Football"),
    ("3FbBFAz696k", "Tactics Explained: Roma", "Tifo Football"),
    ("vFFBqRsTZ-A", "Tactics Explained: Sassuolo", "Tifo Football"),
    ("lltJE4xDS_c", "Diego Simeone's Atletico Madrid Explained", "Tifo Football"),
    ("ZWJkzjouJgk", "Maurizio Sarri's Napoli Tactics Explained", "Tifo Football"),
    ("kl6qoK9W1es", "Arsenal Invincibles Tactics Explained", "Tifo Football"),
    ("3zK57GHRd9k", "Manchester United's 2007/08 Tactics Explained", "Tifo Football"),
    ("wDRVk4qvDX0", "Marcelo Bielsa Tactics Explained", "Tifo Football"),
    ("X1-9UHeBsjg", "Antonio Conte & Chelsea's 3-4-3 System", "Tifo Football"),
    ("MgrADw-zR9s", "Inter Milan's 2009/10 Treble Tactics Explained", "Tifo Football"),
    ("ilq5hMy1yZc", "Julian Nagelsmann's Hoffenheim Tactics Explained", "Tifo Football"),
    ("uLjkWniuBII", "Bayern Munich & Jupp Heynckes' Treble Tactics", "Tifo Football"),
    ("jUPsVSMOi4w", "Real Madrid & Why Casemiro is Crucial", "Tifo Football"),
    ("KyoTLvevy-8", "Arteta's Arsenal in Tactics", "Tifo Football"),
    ("OdK62JC7QdQ", "How Exactly Do Liverpool Play Football?", "Tifo Football"),
    ("hwpsWPseebM", "How Exactly Do Real Madrid Play Football?", "Tifo Football"),
    ("WmqLiEN53HE", "How Chelsea's Tactics Have Evolved", "Tifo Football"),
    ("NKHwUVYH4hU", "The Burnley Low Block Explained", "Tifo Football"),
    ("Z_hwwKcH48k", "Arsenal's 3-4-2-1 Explained", "Tifo Football"),
    ("H9CTWPrFnCg", "How Andrea Pirlo Sees Football", "Tifo Football"),
    ("-eI5PnMErPA", "Why Did Man City Spend So Much On Fullbacks?", "Tifo Football"),
    ("fIPFk2uNv3E", "How Liverpool's Front Three Became a Front Four", "Tifo Football"),
    ("HR0K-E2qszY", "Stefano Pioli's AC Milan Tactics Explained", "Tifo Football"),
    ("h747UjOjKwo", "West Brom & Set Pieces", "Tifo Football"),
]


def _parse_vtt(vtt_path):
    """Parse a VTT subtitle file into plain text, removing timestamps and duplicates.

    YouTube auto-generated VTT files have overlapping cues where each new cue
    repeats the previous line plus adds new words. We only keep lines from cues
    that start at a new position (the second line of each two-line cue block).
    """
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove VTT header and metadata
    content = re.sub(r"WEBVTT.*?\n\n", "", content, count=1, flags=re.DOTALL)

    # Remove all VTT formatting/timing tags like <00:00:05.520><c>
    content = re.sub(r"<[^>]+>", "", content)

    # Remove timestamp lines and position/alignment markers
    content = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->.*?\n", "", content)

    # Split into lines, strip, remove blanks
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    # Remove [Music] and similar markers
    lines = [line for line in lines if not re.match(r"^\[.*\]$", line)]

    # Deduplicate: YouTube auto-subs repeat lines across overlapping cue blocks
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    text = " ".join(deduped)

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _download_subtitle(video_id, tmp_dir):
    """Download English auto-generated subtitles for a video using yt-dlp."""
    output_template = os.path.join(tmp_dir, f"{video_id}")
    cmd = [
        "yt-dlp",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--skip-download",
        "--sub-format", "vtt",
        "--no-warnings",
        "-o", output_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    vtt_path = os.path.join(tmp_dir, f"{video_id}.en.vtt")
    if os.path.exists(vtt_path):
        return vtt_path
    return None


def collect_transcripts():
    """Download and parse English transcripts from curated tactical analysis videos."""
    documents = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for video_id, title, channel in TACTICAL_VIDEOS:
            try:
                vtt_path = _download_subtitle(video_id, tmp_dir)
                if vtt_path is None:
                    print(f"  Skipping '{title}': no subtitles available")
                    continue

                full_text = _parse_vtt(vtt_path)
            except subprocess.TimeoutExpired:
                print(f"  Skipping '{title}': download timed out")
                continue
            except Exception as e:
                print(f"  Skipping '{title}': {e}")
                continue

            if len(full_text.strip()) < 100:
                print(f"  Skipping '{title}': transcript too short")
                continue

            doc_id = hashlib.md5(f"yt-{video_id}".encode()).hexdigest()[:12]

            documents.append({
                "doc_id": doc_id,
                "source": "youtube",
                "title": title,
                "content": full_text,
                "metadata": {
                    "channel": channel,
                    "video_id": video_id,
                    "url": f"https://youtube.com/watch?v={video_id}",
                },
            })
            print(f"  Collected: {title} ({len(full_text)} chars)")

    return documents


if __name__ == "__main__":
    docs = collect_transcripts()
    print(f"\nTotal YouTube documents: {len(docs)}")
    if docs:
        print(f"Sample:\n{docs[0]['content'][:500]}")
