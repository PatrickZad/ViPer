from .mapper import SearchMapperTextBased


class PrwTBPSMapper(SearchMapperTextBased):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
