from meme_pipeline.stage_a.vehicle_extractor import extract_vehicle_candidates_from_captions


class DummyToken:
    def __init__(self, text: str, pos_: str, idx: int) -> None:
        self.text = text
        self.pos_ = pos_
        self.idx = idx
        self.lemma_ = text.lower()


class DummySpan:
    def __init__(self, text: str, start_char: int, end_char: int, root_text: str) -> None:
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.root = DummyToken(root_text, "NOUN", start_char)


class DummyDoc:
    def __init__(self, text: str) -> None:
        lowered = text.lower()
        self.noun_chunks = []
        if "a woman" in lowered:
            start = lowered.index("a woman")
            self.noun_chunks.append(DummySpan(text[start : start + len("A woman")], start, start + len("A woman"), "woman"))
        if "engagement ring" in lowered:
            start = lowered.index("engagement ring")
            span_text = text[start : start + len("engagement ring")]
            self.noun_chunks.append(DummySpan(span_text, start, start + len(span_text), "ring"))
        self._tokens = []
        if "engagement ring" in lowered:
            self._tokens.append(DummyToken("engagement", "ADJ", lowered.index("engagement")))
            self._tokens.append(DummyToken("ring", "NOUN", lowered.index("ring")))
        elif "woman" in lowered:
            self._tokens.append(DummyToken("woman", "NOUN", lowered.index("woman")))
        else:
            self._tokens.append(DummyToken("object", "NOUN", 0))

    def __iter__(self):
        return iter(self._tokens)


class DummyNLP:
    def __call__(self, text: str):
        return DummyDoc(text)


def test_vehicle_extractor_uses_img_captions():
    candidates = extract_vehicle_candidates_from_captions(
        [
            "A woman shows off her engagement ring which Thor approves of.",
            "A couple poses in the top left picture.",
        ],
        DummyNLP(),
    )
    assert candidates
    assert candidates[0]["caption_index"] == 0
    assert candidates[0]["normalized"] == "woman"
    assert any(candidate["normalized"].endswith("engagement ring") for candidate in candidates)
