"""
Media Detection and Content Block Formatting for LLMFS

Detects binary media (images, PDFs, audio, video) from raw bytes using
magic byte signatures, and converts them into provider-specific content
blocks for multimodal LLM APIs.

Usage in the agent pipeline:
    1. Input arrives as raw bytes via 9P write
    2. detect_media(data) identifies media type (or returns None for text)
    3. For mixed input (e.g. text + image), parse_multipart_input() splits
    4. build_content_blocks() creates provider-agnostic content blocks
    5. Each provider's format_message() converts to API-specific format

Supported media types:
    - Images: PNG, JPEG, GIF, WebP, BMP, TIFF
    - Documents: PDF
    - Audio: WAV, MP3, OGG, FLAC, WEBM-audio
    - Video: MP4, WebM, MKV, AVI

Shell usage:
    cat image.png > $agent/input                    # pure image
    echo "describe this" | cat - image.png > $agent/input  # text + image
    cat img1.png img2.png > $agent/input            # multiple images
"""

import base64
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


# ─── Magic byte signatures ────────────────────────────────────────────
# Each entry: (magic_bytes, offset, media_type, mime_type)
# Ordered longest-match-first within same offset for correctness.

MAGIC_SIGNATURES: List[Tuple[bytes, int, str, str]] = [
    # Images
    (b'\x89PNG\r\n\x1a\n',  0, "image", "image/png"),
    (b'\xff\xd8\xff',        0, "image", "image/jpeg"),
    (b'GIF87a',              0, "image", "image/gif"),
    (b'GIF89a',              0, "image", "image/gif"),
    (b'RIFF',                0, "image", "image/webp"),   # RIFF....WEBP - refined below
    (b'BM',                  0, "image", "image/bmp"),
    (b'II\x2a\x00',         0, "image", "image/tiff"),
    (b'MM\x00\x2a',         0, "image", "image/tiff"),
    
    # PDF
    (b'%PDF',               0, "document", "application/pdf"),
    
    # Audio
    (b'ID3',                0, "audio", "audio/mpeg"),       # MP3 with ID3 tag
    (b'\xff\xfb',           0, "audio", "audio/mpeg"),       # MP3 sync word
    (b'\xff\xf3',           0, "audio", "audio/mpeg"),       # MP3 sync word
    (b'\xff\xf2',           0, "audio", "audio/mpeg"),       # MP3 sync word
    (b'OggS',               0, "audio", "audio/ogg"),
    (b'fLaC',               0, "audio", "audio/flac"),
    (b'RIFF',               0, "audio", "audio/wav"),        # RIFF....WAVE - refined below
    
    # Video (checked via container format)
    (b'\x1a\x45\xdf\xa3',  0, "video", "video/webm"),       # Matroska/WebM
    (b'\x00\x00\x00',       0, "video", "video/mp4"),        # MP4/MOV (ftyp box) - refined below
]


@dataclass
class MediaInfo:
    """Detected media information"""
    media_type: str     # "image", "document", "audio", "video"
    mime_type: str       # "image/png", "application/pdf", etc.
    data: bytes          # Raw binary data
    
    @property
    def base64_data(self) -> str:
        """Base64-encoded data string"""
        return base64.b64encode(self.data).decode('ascii')
    
    @property
    def data_uri(self) -> str:
        """Data URI for OpenAI-style APIs"""
        return f"data:{self.mime_type};base64,{self.base64_data}"


@dataclass
class ContentBlock:
    """
    Provider-agnostic content block.
    
    Either text or media. The agent stores these in Message.content_blocks
    and each provider formats them for its API.
    """
    type: str  # "text" or "media"
    
    # For text blocks
    text: Optional[str] = None
    
    # For media blocks
    media: Optional[MediaInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage (history, state snapshots)"""
        if self.type == "text":
            return {"type": "text", "text": self.text}
        else:
            return {
                "type": "media",
                "media_type": self.media.media_type,
                "mime_type": self.media.mime_type,
                "data_b64": self.media.base64_data,
            }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ContentBlock':
        """Deserialize from JSON"""
        if d["type"] == "text":
            return cls(type="text", text=d["text"])
        else:
            raw = base64.b64decode(d["data_b64"])
            media = MediaInfo(
                media_type=d["media_type"],
                mime_type=d["mime_type"],
                data=raw,
            )
            return cls(type="media", media=media)


def detect_media(data: bytes) -> Optional[MediaInfo]:
    """
    Detect media type from raw bytes using magic signatures.
    
    Returns MediaInfo if binary media is detected, None if it looks like text.
    """
    if len(data) < 4:
        return None
    
    for magic, offset, media_type, mime_type in MAGIC_SIGNATURES:
        if len(data) >= offset + len(magic) and data[offset:offset + len(magic)] == magic:
            # Refine RIFF-based formats (WebP vs WAV)
            if magic == b'RIFF' and len(data) >= 12:
                container = data[8:12]
                if container == b'WEBP':
                    return MediaInfo(media_type="image", mime_type="image/webp", data=data)
                elif container == b'WAVE':
                    return MediaInfo(media_type="audio", mime_type="audio/wav", data=data)
                else:
                    continue  # Unknown RIFF, skip
            
            # Refine MP4/MOV (look for 'ftyp' box)
            if magic == b'\x00\x00\x00' and len(data) >= 8:
                if data[4:8] == b'ftyp':
                    return MediaInfo(media_type="video", mime_type="video/mp4", data=data)
                else:
                    continue  # Not an MP4
            
            # Refine Matroska vs WebM (both use same magic)
            # For simplicity, treat all as video/webm
            
            return MediaInfo(media_type=media_type, mime_type=mime_type, data=data)
    
    return None


def _find_media_boundary(data: bytes) -> Optional[int]:
    """
    Find the byte offset where binary media starts in a mixed text+binary stream.
    
    Scans for known magic byte signatures using bytes.find() (C-speed).
    Only searches strong signatures (4+ bytes) to avoid false positives
    from short patterns (like MP3 sync words \xff\xfb) appearing inside
    binary data of another format.
    """
    # Strong signatures only — at least 4 bytes, low false positive rate
    STRONG_SIGNATURES = [
        (b'\x89PNG\r\n\x1a\n', "image",    "image/png"),
        (b'\xff\xd8\xff',       "image",    "image/jpeg"),     # 3 bytes but very distinctive
        (b'GIF87a',             "image",    "image/gif"),
        (b'GIF89a',             "image",    "image/gif"),
        (b'RIFF',               "image",    "image/webp"),     # refined below
        (b'%PDF',               "document", "application/pdf"),
        (b'OggS',               "audio",    "audio/ogg"),
        (b'fLaC',               "audio",    "audio/flac"),
        (b'ID3',                "audio",    "audio/mpeg"),
        (b'\x1a\x45\xdf\xa3',  "video",    "video/webm"),
        (b'BM',                 "image",    "image/bmp"),
        (b'II\x2a\x00',        "image",    "image/tiff"),
        (b'MM\x00\x2a',        "image",    "image/tiff"),
    ]
    # Note: MP3 sync words (\xff\xfb etc.) and MP4 (\x00\x00\x00) are
    # intentionally excluded from boundary scanning — too many false
    # positives inside binary data. They still work in detect_media()
    # for pure-media input (offset 0 check).
    
    best = None
    
    for magic, media_type, mime_type in STRONG_SIGNATURES:
        pos = data.find(magic)
        if pos == -1:
            continue
        
        # RIFF: must be WEBP or WAVE
        if magic == b'RIFF' and len(data) >= pos + 12:
            container = data[pos + 8:pos + 12]
            if container not in (b'WEBP', b'WAVE'):
                continue
            if container == b'WAVE':
                mime_type = "audio/wav"
                media_type = "audio"
        
        if best is None or pos < best:
            best = pos
    
    return best


def parse_input_data(data: bytes) -> List[ContentBlock]:
    """
    Parse raw input bytes into content blocks.
    
    Handles three cases:
    1. Pure text → single text block
    2. Pure binary media → single media block  
    3. Text followed by binary media (e.g. echo "prompt"; cat image.png)
    
    The shell pattern `{ echo "text"; cat file.bin; } > input` produces:
        b"text\\n" + <raw binary bytes>
    
    We find the first magic signature and split there.
    """
    if not data or not data.strip():
        return []
    
    # Case 1: entire input is a known media type (starts at byte 0)
    media = detect_media(data)
    if media is not None:
        return [ContentBlock(type="media", media=media)]
    
    # Case 2: look for media embedded after a text prefix
    boundary = _find_media_boundary(data)
    
    if boundary is not None and boundary > 0:
        # Split: text before boundary, media from boundary onward
        text_part = data[:boundary]
        media_part = data[boundary:]
        
        blocks = []
        
        # Decode the text prefix
        text = text_part.decode('utf-8', errors='replace').strip()
        if text:
            blocks.append(ContentBlock(type="text", text=text))
        
        # Detect the media portion
        media = detect_media(media_part)
        if media:
            blocks.append(ContentBlock(type="media", media=media))
        
        if blocks:
            return blocks
    
    # Case 3: try as pure text
    try:
        text = data.decode('utf-8')
        # Heuristic: if less than 5% non-printable chars, it's text
        non_printable = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if non_printable < len(text) * 0.05:
            return [ContentBlock(type="text", text=text.strip())]
    except UnicodeDecodeError:
        pass
    
    # Case 4: binary data we don't recognize — treat as text (best effort)
    text = data.decode('utf-8', errors='replace').strip()
    if text:
        return [ContentBlock(type="text", text=text)]
    
    return []


# ─── Provider-specific formatting ─────────────────────────────────────

def format_content_for_claude(blocks: List[ContentBlock]) -> Any:
    """
    Format content blocks for Anthropic Claude API.
    
    Claude format:
        {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}
        {"type": "text", "text": "..."}
    """
    if len(blocks) == 1 and blocks[0].type == "text":
        return blocks[0].text  # Simple string for text-only
    
    parts = []
    for block in blocks:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.media is not None:
            if block.media.media_type == "image":
                parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.media.mime_type,
                        "data": block.media.base64_data,
                    }
                })
            elif block.media.media_type == "document":
                parts.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": block.media.mime_type,
                        "data": block.media.base64_data,
                    }
                })
            else:
                # Audio/video: Claude doesn't natively support these yet,
                # add as text description placeholder
                parts.append({
                    "type": "text",
                    "text": f"[Attached {block.media.media_type}: {block.media.mime_type}]"
                })
    
    # If no text block exists, add a default prompt
    has_text = any(b.type == "text" for b in blocks)
    if not has_text:
        parts.append({"type": "text", "text": "What is this?"})
    
    return parts


def format_content_for_openai(blocks: List[ContentBlock]) -> Any:
    """
    Format content blocks for OpenAI-compatible APIs.
    (Also used by Groq, OpenRouter, Cerebras, Moonshot)
    
    OpenAI format:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        {"type": "text", "text": "..."}
    """
    if len(blocks) == 1 and blocks[0].type == "text":
        return blocks[0].text
    
    parts = []
    for block in blocks:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.media is not None:
            if block.media.media_type in ("image", "document"):
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": block.media.data_uri,
                    }
                })
            else:
                parts.append({
                    "type": "text",
                    "text": f"[Attached {block.media.media_type}: {block.media.mime_type}]"
                })
    
    has_text = any(b.type == "text" for b in blocks)
    if not has_text:
        parts.append({"type": "text", "text": "What is this?"})
    
    return parts


def format_content_for_gemini(blocks: List[ContentBlock]) -> list:
    """
    Format content blocks for Google Gemini API.
    
    Gemini format (parts list):
        {"text": "..."}
        {"inline_data": {"mime_type": "image/png", "data": "base64..."}}
    """
    parts = []
    for block in blocks:
        if block.type == "text":
            parts.append({"text": block.text})
        elif block.media is not None:
            parts.append({
                "inline_data": {
                    "mime_type": block.media.mime_type,
                    "data": block.media.base64_data,
                }
            })
    
    if not parts:
        parts.append({"text": "What is this?"})
    
    return parts


def estimate_media_tokens(media: MediaInfo) -> int:
    """
    Estimate token cost of a media attachment.
    
    Different providers count media tokens differently, but rough estimates:
    - Images: ~1000-2000 tokens depending on size (Claude uses ~1600 for medium)
    - PDFs: ~1500 per page (rough)
    - Audio/Video: varies widely
    
    This is a rough heuristic for context window management.
    """
    size_kb = len(media.data) / 1024
    
    if media.media_type == "image":
        # Claude: images cost between 170 (low-res) and 1600+ tokens
        # Use a conservative estimate based on data size
        if size_kb < 50:
            return 500
        elif size_kb < 500:
            return 1200
        else:
            return 2000
    elif media.media_type == "document":
        # PDFs: rough estimate per page, assume ~2KB per page
        pages = max(1, size_kb / 2)
        return int(pages * 1500)
    else:
        # Audio/video: very rough
        return int(size_kb * 2)