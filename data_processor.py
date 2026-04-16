class DataPreprocessor:
    def __init__(self, flood_df, dfsi_df):
        self.flood_df = flood_df
        self.dfsi_df = dfsi_df

    def normalize_flood(self):
        max_level = self.flood_df["water_level"].max()
        self.flood_df["hazard_norm"] = self.flood_df["water_level"] / max_level
        return self.flood_df

    def merge_dfsi(self):
        # Merge on city column if available
        if "city" in self.flood_df.columns:
            self.flood_df = self.flood_df.merge(
                self.dfsi_df,
                on="city",
                how="left"
            )
        return self.flood_df