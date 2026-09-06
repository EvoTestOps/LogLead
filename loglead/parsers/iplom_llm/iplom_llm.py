import logging

import polars as pl

logger = logging.getLogger(__name__)


class IPLoMLLMParser:
    """
    Wrapper for iplom-llm-parser (https://github.com/EvoTestOps/iplom-llm-parser)
    """

    def __init__(
        self,
        messages,
        config=None,
        config_path: str = "config.toml",
        client=None,
    ):
        try:
            from iplom_llm_parser import Config, LLMClient, load_config
        except ImportError as e:
            raise ImportError(
                "IPLoMLLMParser requires the 'iplom_llm' extra and Python 3.11+. "
                "Install with: pip install 'loglead[iplom_llm]'"
            ) from e

        if config is None:
            config = load_config(config_path)
        if not isinstance(config, Config):
            raise TypeError("config must be an iplom_llm_parser.Config instance")

        self.config = config
        self._owns_client = client is None
        self.client = client if client is not None else LLMClient(config.llm)

        self.messages_df = pl.DataFrame(
            {config.pipeline.content_col: messages}
        ).with_row_index("row_nr")

        self.df_log: pl.DataFrame | None = None
        self.stats = None

    def parse(self) -> "IPLoMLLMParser":
        from iplom_llm_parser import TemplatePipeline

        try:
            pipeline = TemplatePipeline(
                self.config.pipeline,
                self.config.iplom,
                self.client,
                self.messages_df,
            )
            df_res = pipeline.run()
            self.stats = pipeline.stats
        finally:
            if self._owns_client:
                self.client.close()

        self.df_log = df_res.select(
            pl.col("EventId").fill_null("e_null"),
            pl.col("EventTemplate"),
            pl.col("ParameterList"),
            pl.col("SlotTypes"),
        )
        return self
