from meme_pipeline.stage_a.vehicle_extractor import extract_vehicle_candidates


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
        self.text = text
        lowered = text.lower()
        self.noun_chunks = []
        if "confused cat" in lowered:
            start = lowered.index("a confused cat") if "a confused cat" in lowered else lowered.index("confused cat")
            surface = text[start : start + (14 if "a confused cat" in lowered else 12)]
            self.noun_chunks.append(DummySpan(surface, start, start + len(surface), "cat"))
        if "office worker" in lowered:
            start = lowered.index("office worker")
            surface = text[start : start + len("office worker")]
            self.noun_chunks.append(DummySpan(surface, start, start + len(surface), "worker"))
        self._tokens = []
        for word in lowered.split():
            clean = word.strip(".,")
            if clean in {"cat", "worker", "desk"}:
                self._tokens.append(DummyToken(clean, "NOUN", lowered.index(clean)))
            elif clean in {"confused", "office"}:
                self._tokens.append(DummyToken(clean, "ADJ", lowered.index(clean)))
            else:
                self._tokens.append(DummyToken(clean, "DET", lowered.index(clean)))

    def __iter__(self):
        return iter(self._tokens)


class DummyNLP:
    def __call__(self, text: str):
        return DummyDoc(text)


def test_vehicle_extractor_returns_noun_chunks():
    candidates = extract_vehicle_candidates("a confused cat near an office worker", DummyNLP())
    assert candidates
    assert candidates[0]["normalized"] == "confused cat"
    assert candidates[1]["head"] == "worker"
