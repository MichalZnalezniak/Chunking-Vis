from django import forms
from django.core.validators import MaxValueValidator


class TextToChunk(forms.Form):
    text_to_chunk = forms.CharField(
        label="TextToChunk",
        widget=forms.Textarea(
            attrs={
                "class": "my-input",
            }
        ),
        initial="""Marie went to the bakery in the morning. She bought a fresh loaf of bread, some croissants, and a small jar of homemade jam. The bakery smelled of warm bread and sweet pastries, and she greeted the baker with a cheerful smile. 

After that, she walked to the park to enjoy her breakfast on a bench under a large oak tree. The sun was shining brightly, and children were playing near the fountain, their laughter echoing across the park. Birds chirped from the branches above, and a gentle breeze carried the scent of blooming flowers. She unwrapped her croissants and poured herself a cup of coffee from her thermos, savoring each bite while watching the ducks swim in the pond. 
Later, she returned home and started preparing lunch, chopping fresh vegetables and seasoning a piece of chicken. The kitchen filled with delicious aromas, and she hummed a tune as she worked. After lunch, she cleaned up the dishes and sat by the window, reading a chapter of her favorite book while the afternoon light streamed in.""",
        max_length=4000,
    )


class ChunkSize(forms.Form):
    chunk_size = forms.IntegerField(
        label="Chunk Size",
        widget=forms.NumberInput(
            attrs={
                "class": "chunk_size",
            }
        ),
        initial=256,
        min_value=0,
        max_value=512,
    )


class ChunkOverlap(forms.Form):
    chunk_overlap = forms.IntegerField(
        label="Chunk Overlap",
        widget=forms.NumberInput(
            attrs={
                "class": "chunk_size_overlap",
            }
        ),
        error_messages={
            "max_value": "Please enter a number less than or equal to %(limit_value)s.",
            "required": "Please fill out the form",
        },
        initial=2,
        min_value=0,
        max_value=9,
    )

    def __init__(self, *args, max_value=None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_value is not None:
            # override max_value dynamically
            self.fields["chunk_overlap"].max_value = max_value
            validators = [v for v in self.fields["chunk_overlap"].validators if not isinstance(v, MaxValueValidator)]
            validators.append(MaxValueValidator(max_value))
            self.fields["chunk_overlap"].validators = validators
            self.fields["chunk_overlap"].widget.attrs["max"] = max_value

    def set_max_value(self, max_value):
        self.fields["chunk_overlap"].max_value = max_value
        new_validators = [v for v in self.fields["chunk_overlap"].validators if not isinstance(v, MaxValueValidator)]
        new_validators.append(MaxValueValidator(max_value))
        self.fields["chunk_overlap"].validators = new_validators
        self.fields["chunk_overlap"].widget.attrs["max"] = max_value


class Splitter(forms.Form):
    CHOICES = [
        ("Character Splitter", "Character Splitter"),
        ("Word Splitter", "Word Splitter"),
        ("Token Splitter", "Token Splitter"),
        ("Recursive Splitter", "Recursive Splitter"),
        ("Semantic Splitter", "Semantic Splitter"),
        ("Agentic Splitter", "Agentic Splitter"),
    ]

    splitter = forms.ChoiceField(
        choices=CHOICES,
        widget=forms.Select(attrs={"class": "my-dropdown"}),
        label="Splitter",
        initial="Character Splitter",
    )


class Separators(forms.Form):
    separators = forms.CharField(
        label="Separators",
        widget=forms.Textarea(
            attrs={
                "class": "my-input2",
            }
        ),
        initial="",
        max_length=100,
        required=False,
    )
