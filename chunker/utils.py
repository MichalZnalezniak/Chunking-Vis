import string


def chunk_by_words(text, chunk_size):
    words = text.split(" ")
    chunks = []
    current_chunk = ""

    for word in words:
        # +1 for space if current_chunk is not empty
        add_len = len(word) + (1 if current_chunk else 0)
        if len(current_chunk) + add_len <= chunk_size:
            current_chunk += (" " if current_chunk else "") + word
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = " " + word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunks_to_html(chunks):
    colors = ["#FFF3B0", "#FFD5C2", "#CCF2D0", "#D6E6F2", "#E5D9F2"]
    overlap_color = "#d3d3d3"  # gray

    html_parts = []
    base_index = 0
    for text, is_overlap in chunks:
        if is_overlap:
            color = overlap_color
        else:
            color = colors[base_index % len(colors)]
            if text.strip() != "":
                base_index += 1
        html_parts.append(f'<span style="white-space: pre-line; background-color:{color}">{text}</span>')
    return "".join(html_parts)


def join_elements(tokens):
    if not tokens:
        return ""

    result = tokens[0]
    for tok in tokens[1:]:
        if tok in string.punctuation:
            result += tok
        else:
            result += " " + tok
    return result
