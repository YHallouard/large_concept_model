from lcm_explo.domain.infrastructures.splitter import TextSplitter


class InMemorySplitter(TextSplitter):
    def __init__(self, memory: dict[str, list[str]]):
        self.memory = memory

    def split(self, text: str) -> list[str]:
        return self.memory[text]
