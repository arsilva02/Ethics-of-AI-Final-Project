"""Speeddating Dataset"""

from typing import List

import datasets

import pandas


VERSION = datasets.Version("1.0.0")
_BASE_FEATURE_NAMES = [
    "is_dater_male",
    "dater_age",
    "dated_age",
    "age_difference",
    "dater_race",
    "dated_race",
    "are_same_race",
    "same_race_importance_for_dater",
    "same_religion_importance_for_dater",
    "attractiveness_importance_for_dated",
    "sincerity_importance_for_dated",
    "intelligence_importance_for_dated",
    "humor_importance_for_dated",
    "ambition_importance_for_dated",
    "shared_interests_importance_for_dated",
    "attractiveness_score_of_dater_from_dated",
    "sincerity_score_of_dater_from_dated",
    "intelligence_score_of_dater_from_dated",
    "humor_score_of_dater_from_dated",
    "ambition_score_of_dater_from_dated",
    "shared_interests_score_of_dater_from_dated",
    "attractiveness_importance_for_dater",
    "sincerity_importance_for_dater",
    "intelligence_importance_for_dater",
    "humor_importance_for_dater",
    "ambition_importance_for_dater",
    "shared_interests_importance_for_dater",
    "self_reported_attractiveness_of_dater",
    "self_reported_sincerity_of_dater",
    "self_reported_intelligence_of_dater",
    "self_reported_humor_of_dater",
    "self_reported_ambition_of_dater",
    "reported_attractiveness_of_dated_from_dater",
    "reported_sincerity_of_dated_from_dater",
    "reported_intelligence_of_dated_from_dater",
    "reported_humor_of_dated_from_dater",
    "reported_ambition_of_dated_from_dater",
    "reported_shared_interests_of_dated_from_dater",
    "dater_interest_in_sports",
    "dater_interest_in_tvsports",
    "dater_interest_in_exercise",
    "dater_interest_in_dining",
    "dater_interest_in_museums",
    "dater_interest_in_art",
    "dater_interest_in_hiking",
    "dater_interest_in_gaming",
    "dater_interest_in_clubbing",
    "dater_interest_in_reading",
    "dater_interest_in_tv",
    "dater_interest_in_theater",
    "dater_interest_in_movies",
    "dater_interest_in_concerts",
    "dater_interest_in_music",
    "dater_interest_in_shopping",
    "dater_interest_in_yoga",
    "interests_correlation",
    "expected_satisfaction_of_dater",
    "expected_number_of_likes_of_dater_from_20_people",
    "expected_number_of_dates_for_dater",
    "dater_liked_dated",
    "probability_dated_wants_to_date",
    "already_met_before",
    "dater_wants_to_date",
    "dated_wants_to_date",
    "is_match"
]


DESCRIPTION = "Speed-dating dataset."
_HOMEPAGE = "https://www.openml.org/search?type=data&sort=nr_of_likes&status=active&id=40536"
_URLS = ("https://huggingface.co/datasets/mstz/speeddating/raw/main/speeddating.csv")
_CITATION = """"""

# Dataset info
urls_per_split = {
    "train": "https://huggingface.co/datasets/mstz/speeddating/raw/main/speeddating.csv",
}
features_types_per_config = {
    "dating": {
        "is_dater_male": datasets.Value("bool"),
        "dater_age": datasets.Value("int8"),
        "dated_age": datasets.Value("int8"),
        "age_difference": datasets.Value("int8"),
        "dater_race": datasets.Value("string"),
        "dated_race": datasets.Value("string"),
        "are_same_race": datasets.Value("bool"),
        "same_race_importance_for_dater": datasets.Value("float64"),
        "same_religion_importance_for_dater": datasets.Value("float64"),
        "attractiveness_importance_for_dated": datasets.Value("float64"),
        "sincerity_importance_for_dated": datasets.Value("float64"),
        "intelligence_importance_for_dated": datasets.Value("float64"),
        "humor_importance_for_dated": datasets.Value("float64"),
        "ambition_importance_for_dated": datasets.Value("float64"),
        "shared_interests_importance_for_dated": datasets.Value("float64"),
        "attractiveness_score_of_dater_from_dated": datasets.Value("float64"),
        "sincerity_score_of_dater_from_dated": datasets.Value("float64"),
        "intelligence_score_of_dater_from_dated": datasets.Value("float64"),
        "humor_score_of_dater_from_dated": datasets.Value("float64"),
        "ambition_score_of_dater_from_dated": datasets.Value("float64"),
        "shared_interests_score_of_dater_from_dated": datasets.Value("float64"),
        "attractiveness_importance_for_dater": datasets.Value("float64"),
        "sincerity_importance_for_dater": datasets.Value("float64"),
        "intelligence_importance_for_dater": datasets.Value("float64"),
        "humor_importance_for_dater": datasets.Value("float64"),
        "ambition_importance_for_dater": datasets.Value("float64"),
        "shared_interests_importance_for_dater": datasets.Value("float64"),
        "self_reported_attractiveness_of_dater": datasets.Value("float64"),
        "self_reported_sincerity_of_dater": datasets.Value("float64"),
        "self_reported_intelligence_of_dater": datasets.Value("float64"),
        "self_reported_humor_of_dater": datasets.Value("float64"),
        "self_reported_ambition_of_dater": datasets.Value("float64"),
        "reported_attractiveness_of_dated_from_dater": datasets.Value("float64"),
        "reported_sincerity_of_dated_from_dater": datasets.Value("float64"),
        "reported_intelligence_of_dated_from_dater": datasets.Value("float64"),
        "reported_humor_of_dated_from_dater": datasets.Value("float64"),
        "reported_ambition_of_dated_from_dater": datasets.Value("float64"),
        "reported_shared_interests_of_dated_from_dater": datasets.Value("float64"),
        "dater_interest_in_sports": datasets.Value("float64"),
        "dater_interest_in_tvsports": datasets.Value("float64"),
        "dater_interest_in_exercise": datasets.Value("float64"),
        "dater_interest_in_dining": datasets.Value("float64"),
        "dater_interest_in_museums": datasets.Value("float64"),
        "dater_interest_in_art": datasets.Value("float64"),
        "dater_interest_in_hiking": datasets.Value("float64"),
        "dater_interest_in_gaming": datasets.Value("float64"),
        "dater_interest_in_clubbing": datasets.Value("float64"),
        "dater_interest_in_reading": datasets.Value("float64"),
        "dater_interest_in_tv": datasets.Value("float64"),
        "dater_interest_in_theater": datasets.Value("float64"),
        "dater_interest_in_movies": datasets.Value("float64"),
        "dater_interest_in_concerts": datasets.Value("float64"),
        "dater_interest_in_music": datasets.Value("float64"),
        "dater_interest_in_shopping": datasets.Value("float64"),
        "dater_interest_in_yoga": datasets.Value("float64"),
        "interests_correlation": datasets.Value("float64"),
        "expected_satisfaction_of_dater": datasets.Value("float64"),
        "expected_number_of_likes_of_dater_from_20_people": datasets.Value("int8"),
        "expected_number_of_dates_for_dater": datasets.Value("int8"),
        "dater_liked_dated": datasets.Value("float64"),
        "probability_dated_wants_to_date": datasets.Value("float64"),
        "already_met_before": datasets.Value("bool"),
        "dater_wants_to_date": datasets.Value("bool"),
        "dated_wants_to_date": datasets.Value("bool"),
        "is_match": datasets.ClassLabel(num_classes=2, names=("no", "yes"))
    }
    
}
features_per_config = {k: datasets.Features(features_types_per_config[k]) for k in features_types_per_config}


class SpeeddatingConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(SpeeddatingConfig, self).__init__(version=VERSION, **kwargs)
        self.features = features_per_config[kwargs["name"]]


class Speeddating(datasets.GeneratorBasedBuilder):
    # dataset versions
    DEFAULT_CONFIG = "dating"
    BUILDER_CONFIGS = [
        SpeeddatingConfig(name="dating",
                          description="Binary classification."),
    ]


    def _info(self):
        info = datasets.DatasetInfo(description=DESCRIPTION, citation=_CITATION, homepage=_HOMEPAGE,
                                    features=features_per_config[self.config.name])

        return info
    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloads = dl_manager.download_and_extract(urls_per_split)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloads["train"]}),
        ]
    
    def _generate_examples(self, filepath: str):
        data = pandas.read_csv(filepath)
        data = self.preprocess(data, config=self.config.name)

        for row_id, row in data.iterrows():
            data_row = dict(row)

            yield row_id, data_row

    def preprocess(self, data: pandas.DataFrame, config: str = "dating") -> pandas.DataFrame:
        data.loc[data.race == "?", "race"] = "unknown"
        data.loc[data.race_o == "?", "race_o"] = "unknown"
        data.loc[data.race == "Asian/Pacific Islander/Asian-American", "race"] = "asian"
        data.loc[data.race_o == "Asian/Pacific Islander/Asian-American", "race_o"] = "asian"
        data.loc[data.race == "European/Caucasian-American", "race"] = "caucasian"
        data.loc[data.race_o == "European/Caucasian-American", "race_o"] = "caucasian"
        data.loc[data.race == "Other", "race"] = "other"
        data.loc[data.race_o == "Other", "race_o"] = "other"
        data.loc[data.race == "Latino/Hispanic American", "race"] = "hispanic"
        data.loc[data.race_o == "Latino/Hispanic American", "race_o"] = "hispanic"
        data.loc[data.race == "Black/African American", "race"] = "african-american"
        data.loc[data.race_o == "Black/African American", "race_o"] = "african-american"

        data = data.rename(columns={"gender": "is_dater_male"})
        data.loc[:, "is_dater_male"] = data.is_dater_male.apply(lambda x: 1 if x == "male" else 0)

        data.drop("has_null", axis="columns", inplace=True)
        data.drop("field", axis="columns", inplace=True)
        data.drop("wave", axis="columns", inplace=True)
        # data.drop("d_age", axis="columns", inplace=True)
        data.drop("d_d_age", axis="columns", inplace=True)
        data.drop("d_importance_same_race", axis="columns", inplace=True)
        data.drop("d_importance_same_religion", axis="columns", inplace=True)
        data.drop("d_pref_o_attractive", axis="columns", inplace=True)
        data.drop("d_pref_o_sincere", axis="columns", inplace=True)
        data.drop("d_pref_o_intelligence", axis="columns", inplace=True)
        data.drop("d_pref_o_funny", axis="columns", inplace=True)
        data.drop("d_pref_o_ambitious", axis="columns", inplace=True)
        data.drop("d_pref_o_shared_interests", axis="columns", inplace=True)
        data.drop("d_attractive_o", axis="columns", inplace=True)
        data.drop("d_sinsere_o", axis="columns", inplace=True)
        data.drop("d_intelligence_o", axis="columns", inplace=True)
        data.drop("d_funny_o", axis="columns", inplace=True)
        data.drop("d_ambitous_o", axis="columns", inplace=True)
        data.drop("d_shared_interests_o", axis="columns", inplace=True)
        data.drop("d_attractive_important", axis="columns", inplace=True)
        data.drop("d_sincere_important", axis="columns", inplace=True)
        data.drop("d_intellicence_important", axis="columns", inplace=True)
        data.drop("d_funny_important", axis="columns", inplace=True)
        data.drop("d_ambtition_important", axis="columns", inplace=True)
        data.drop("d_shared_interests_important", axis="columns", inplace=True)
        data.drop("d_attractive", axis="columns", inplace=True)
        data.drop("d_sincere", axis="columns", inplace=True)
        data.drop("d_intelligence", axis="columns", inplace=True)
        data.drop("d_funny", axis="columns", inplace=True)
        data.drop("d_ambition", axis="columns", inplace=True)
        data.drop("d_attractive_partner", axis="columns", inplace=True)
        data.drop("d_sincere_partner", axis="columns", inplace=True)
        data.drop("d_intelligence_partner", axis="columns", inplace=True)
        data.drop("d_funny_partner", axis="columns", inplace=True)
        data.drop("d_ambition_partner", axis="columns", inplace=True)
        data.drop("d_shared_interests_partner", axis="columns", inplace=True)
        data.drop("d_sports", axis="columns", inplace=True)
        data.drop("d_tvsports", axis="columns", inplace=True)
        data.drop("d_exercise", axis="columns", inplace=True)
        data.drop("d_dining", axis="columns", inplace=True)
        data.drop("d_museums", axis="columns", inplace=True)
        data.drop("d_art", axis="columns", inplace=True)
        data.drop("d_hiking", axis="columns", inplace=True)
        data.drop("d_gaming", axis="columns", inplace=True)
        data.drop("d_clubbing", axis="columns", inplace=True)
        data.drop("d_reading", axis="columns", inplace=True)
        data.drop("d_tv", axis="columns", inplace=True)
        data.drop("d_theater", axis="columns", inplace=True)
        data.drop("d_movies", axis="columns", inplace=True)
        data.drop("d_concerts", axis="columns", inplace=True)
        data.drop("d_music", axis="columns", inplace=True)
        data.drop("d_shopping", axis="columns", inplace=True)
        data.drop("d_yoga", axis="columns", inplace=True)
        data.drop("d_interests_correlate", axis="columns", inplace=True)
        data.drop("d_expected_happy_with_sd_people", axis="columns", inplace=True)
        data.drop("d_expected_num_interested_in_me", axis="columns", inplace=True)
        data.drop("d_expected_num_matches", axis="columns", inplace=True)
        data.drop("d_like", axis="columns", inplace=True)
        data.drop("d_guess_prob_liked", axis="columns", inplace=True)
        if "Unnamed: 123" in data.columns:
            data.drop("Unnamed: 123", axis="columns", inplace=True)

        data = data[data.age != "?"]
        data = data[data.age_o != "?"]
        data = data[data.importance_same_race != "?"]
        data = data[data.pref_o_attractive != "?"]
        data = data[data.pref_o_sincere != "?"]
        data = data[data.interests_correlate != "?"]
        data = data[data.pref_o_funny != "?"]
        data = data[data.pref_o_ambitious != "?"]
        data = data[data.pref_o_shared_interests != "?"]
        data = data[data.attractive_o != "?"]
        data = data[data.sinsere_o != "?"]
        data = data[data.intelligence_o != "?"]
        data = data[data.funny_o != "?"]
        data = data[data.ambitous_o != "?"]
        data = data[data.shared_interests_o != "?"]
        data = data[data.funny_important != "?"]
        data = data[data.ambtition_important != "?"]
        data = data[data.shared_interests_important != "?"]
        data = data[data.attractive != "?"]
        data = data[data.sincere != "?"]
        data = data[data.intelligence != "?"]
        data = data[data.funny != "?"]
        data = data[data.ambition != "?"]
        data = data[data.attractive_partner != "?"]
        data = data[data.sincere_partner != "?"]
        data = data[data.intelligence_partner != "?"]
        data = data[data.funny_partner != "?"]
        data = data[data.ambition_partner != "?"]
        data = data[data.shared_interests_partner != "?"]
        data = data[data.expected_num_interested_in_me != "?"]
        data = data[data.expected_num_matches != "?"]
        data = data[data.like != "?"]
        data = data[data.guess_prob_liked != "?"]
        data = data[data.met != "?"]

        data.columns = _BASE_FEATURE_NAMES
        data = data.astype({"is_dater_male": "bool", "are_same_race": "bool", "already_met_before": "bool",
                            "dater_wants_to_date": "bool", "dated_wants_to_date": "bool"})
        
        return data             
