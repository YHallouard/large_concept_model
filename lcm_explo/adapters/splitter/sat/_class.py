from lcm_explo.domain.infrastructures.splitter import TextSplitter


class SaTSplitter(TextSplitter):
    """Sentence splitter backed by wtpsplit's SaT models."""

    def __init__(self, model_name: str = "sat-3l") -> None:
        from wtpsplit import SaT  # type: ignore[import-untyped]  # heavy import, deferred

        self._sat = SaT(model_name)

    def split(self, text: str) -> list[str]:
        return [sentence.strip() for sentence in self._sat.split(text) if sentence.strip()]
