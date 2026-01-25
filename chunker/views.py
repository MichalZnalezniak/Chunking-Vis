import math
import re

from django.shortcuts import render

from .forms import ChunkOverlap, ChunkSize, Separators, Splitter, TextToChunk
from .splitter import get_splitter
from .utils import chunks_to_html


def index(request):
    form_text_to_chunk = TextToChunk(request.POST or None)
    form_chunk_size = ChunkSize(request.POST or None)
    form_chunk_overlap = ChunkOverlap(request.POST or None, max_value=math.ceil(form_chunk_size.fields["chunk_size"].initial / 2) - 1)
    form_splitter = Splitter(request.POST or None)
    form_separators = Separators(request.POST or None)
    add_or_remove_snapshot = False

    form_data = {
        "text_to_chunk": form_text_to_chunk.fields["text_to_chunk"].initial,
        "chunk_size": form_chunk_size.fields["chunk_size"].initial,
        "chunk_overlap": form_chunk_overlap.fields["chunk_overlap"].initial,
        "splitter": form_splitter.fields["splitter"].initial,
        "separators": form_separators.fields["separators"].initial,
    }
    snapshots = request.session.get("snapshots", [])
    template_data = {}
    if request.method == "POST":
        if "chunk" in request.POST:
            if form_text_to_chunk.is_valid():
                form_data["text_to_chunk"] = form_text_to_chunk.cleaned_data["text_to_chunk"]
            if form_chunk_size.is_valid():
                form_data["chunk_size"] = form_chunk_size.cleaned_data["chunk_size"]
                form_chunk_overlap.set_max_value(math.ceil(form_data["chunk_size"] / 2) - 2)
            if form_chunk_overlap.is_valid():
                form_data["chunk_overlap"] = form_chunk_overlap.cleaned_data["chunk_overlap"]
            if form_splitter.is_valid():
                form_data["splitter"] = form_splitter.cleaned_data["splitter"]
            if form_separators.is_valid():
                separators = form_separators.cleaned_data["separators"]
                if separators:
                    separators = re.split(r",\s*", separators)
                    separators = [re.sub(r'[\'"]', "", separator) for separator in separators]
                    separators = [separator.encode().decode("unicode_escape") for separator in separators]
                form_data["separators"] = separators

            form_data["text_to_chunk"] = form_data["text_to_chunk"].replace("\r\n", "\n")
            request.session["form_data"] = form_data

        elif "snapshot-add" in request.POST or "snapshot-remove" in request.POST:
            add_or_remove_snapshot = True
            form_data = request.session.get("form_data")
            template_data = request.session.get("template_data")
            if form_data:
                form_text_to_chunk = TextToChunk(initial={"text_to_chunk": form_data["text_to_chunk"]})
                form_chunk_size = ChunkSize(initial={"chunk_size": form_data["chunk_size"]})
                form_chunk_overlap = ChunkOverlap(initial={"chunk_overlap": form_data["chunk_overlap"]})
                form_splitter = Splitter(initial={"splitter": form_data["splitter"]})
                form_separators = Separators(initial={"separators": form_data["separators"]})
                if "snapshot-add" in request.POST:
                    snapshots.append(
                        {
                            "html": template_data["chunked_text_html"],
                            "chunk_size": form_data["chunk_size"],
                            "chunk_overlap": form_data["chunk_overlap"],
                            "splitter": form_data["splitter"],
                            "separators": form_data["separators"],
                        }
                    )
                elif "snapshot-remove" in request.POST:
                    snapshots.pop()

                request.session["snapshots"] = snapshots
                request.session.modified = True
    if not add_or_remove_snapshot:
        chunked_text, chunks_number, characters_number, average_chunk_size = get_splitter(form_data["splitter"]).chunk(
            form_data["text_to_chunk"],
            form_data["chunk_size"],
            form_data["chunk_overlap"],
            form_data["separators"],
        )
        average_chunk_size = round(average_chunk_size, 2)
        chunked_text_html = chunks_to_html(chunked_text)
        request.session["chunked_text"] = chunked_text
        request.session["chunks_number"] = chunks_number
        request.session["characters_number"] = characters_number
        request.session["average_chunk_size"] = average_chunk_size
        request.session["chunked_text_html"] = chunked_text_html
    else:
        chunked_text = request.session["chunked_text_html"]
        chunks_number = request.session["chunks_number"]
        characters_number = request.session["characters_number"]
        average_chunk_size = request.session["average_chunk_size"]
        chunked_text_html = request.session["chunked_text_html"]

    template_data["chunks_number"] = chunks_number
    template_data["characters_number"] = characters_number
    template_data["chunked_text_html"] = chunked_text_html
    template_data["average_chunk_size"] = average_chunk_size
    request.session["template_data"] = template_data

    return render(
        request,
        "chunker/index.html",
        {
            "chunked_text_html": chunked_text_html,
            "characters_number": characters_number,
            "chunks_number": chunks_number,
            "average_chunk_size": average_chunk_size,
            "form_separators": form_separators,
            "form_splitter": form_splitter,
            "form_chunk_overlap": form_chunk_overlap,
            "form_chunk_size": form_chunk_size,
            "form_text_to_chunk": form_text_to_chunk,
            "snapshots": snapshots,
        },
    )
