from .mapper import SearchMapperTextBased


class CuhksysuTBPSMapper(SearchMapperTextBased):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
