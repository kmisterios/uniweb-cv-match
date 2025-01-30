import os
import re
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List

import nltk
import numpy as np
import pandas as pd
import simplemma
import yaml
from dotenv import load_dotenv
from loguru import logger
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from model import CvSelector
from utils.enums import Method, ModeInfo
from utils.request_api import OpenAIEmbedder, process_corpus

load_dotenv()
nltk.download("punkt_tab")


class MassCvSelector(CvSelector):
    def __init__(
        self, config: Dict, api_token: str, method: Method = Method.EMBEDDINGS
    ):
        CvSelector.__init__(
            self,
            config=config,
            api_token=api_token,
            method=method,
        )
        self.stemmer = SnowballStemmer(language="russian")
        self.days_filter_threshold = config["stage_1"]["days_filter_threshold"]
        self.top_n_init = config["stage_1"]["top_n_init"]
        self.close_dist_threshold = config["stage_1"]["close_dist_threshold"]

    def __normalize_text(self, text: str):
        text = text.lower().replace("\n", " ")
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = " ".join(sorted(text.split(" ")))
        return text

    def __minmax_scale(self, x: np.ndarray):
        return (x - x.min()) / (x.max() - x.min())

    def __tokenize_feat(self, text: str):
        norm_text = self.__normalize_text(text)
        words = word_tokenize(norm_text, language="russian")
        prep_words = []
        for word in words:
            word_lemm = simplemma.lemmatize(word, lang="ru")
            word_stemm = self.stemmer.stem(word_lemm)
            prep_words.append(word_stemm)
        return prep_words

    def __bm25_score(self, feature_list: List[str], query: str):
        tokenized_corpus = [self.__tokenize_feat(x) for x in feature_list]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self.__tokenize_feat(query)
        scores = np.abs(bm25.get_scores(tokenized_query))
        return self.__minmax_scale(scores)

    def __filter_1_stage(
        self, date: datetime.date, availability: str, days_thresh: int = 60
    ):
        if datetime.now().date() - timedelta(days=days_thresh) > date:
            return False
        if availability == "Нет":
            return False
        return True

    def __score_move(self, move: str, dist: float, dist_thresh: float = 50):
        if dist < dist_thresh:
            return 1.0
        value_array = np.array(["Невозможен", "Нет данных", "Возможен"])
        return float(np.where(value_array == move)[0][0]) / 2

    def __shorten_address(self, address: str):
        split_ = address.split(", ")
        split_ = split_[:3]
        result = []
        for item in split_:
            norm = True
            key_words = ["ул.", "улиц", "пер.", "переулок", "пл.", "площ"]
            for key in key_words:
                if key in item:
                    norm = False
                    break
            if norm:
                result.append(item)
        return ", ".join(result)

    def rank_first_stage(self, vacancy: Dict, df_relevant: pd.DataFrame):
        df_relevant[self.cluster_match_features[2]] = pd.to_datetime(
            df_relevant[self.cluster_match_features[2]]
        )
        df_relevant.dropna(subset=self.text_features[1], inplace=True)
        df_relevant[f"{self.text_features[1]}_sim"] = self.__bm25_score(
            df_relevant[self.text_features[1]].to_list(),
            query=self.__shorten_address(vacancy["Адрес"]),
        )
        df_relevant.loc[:, "filter"] = df_relevant[
            [self.cluster_match_features[2], "Доступность"]
        ].apply(
            lambda x: self.__filter_1_stage(
                x.iloc[0].date(), x.iloc[1], self.days_filter_threshold
            ),
            axis=1,
        )
        df_relevant_filtered = df_relevant[df_relevant["filter"]]
        df_relevant_filtered = df_relevant_filtered.sort_values(
            f"{self.text_features[1]}_sim", ascending=False
        )
        df_relevant_top_init = df_relevant_filtered.head(self.top_n_init)
        df_relevant_top_init.loc[:, f"{self.cluster_match_features[0]}_sim"] = (
            df_relevant_top_init[[self.cluster_match_features[0], "distance"]]
            .apply(
                lambda x: self.__score_move(
                    x.iloc[0], x.iloc[1], self.close_dist_threshold
                ),
                axis=1,
            )
            .values.copy()
        )
        df_relevant_top_init = df_relevant_top_init[
            df_relevant_top_init[f"{self.cluster_match_features[0]}_sim"] != 0.0
        ]
        df_relevant_top_init[f"{self.text_features[0]}_sim"] = self.__bm25_score(
            feature_list=df_relevant_top_init[self.text_features[0]].to_list(),
            query=vacancy[self.text_features[0]],
        )
        dist_values = df_relevant_top_init["distance"].values
        df_relevant_top_init[
            f"{self.cluster_match_features[1]}_sim"
        ] = 1 - self.__minmax_scale(dist_values)
        df_relevant_top_init[f"{self.cluster_match_features[2]}_sim"] = (
            1
            - df_relevant_top_init[self.cluster_match_features[2]].apply(
                lambda x: (datetime.now().date() - x.date()).days
            )
            / self.days_filter_threshold
        )
        features_rank = [self.text_features[0] + "_sim"] + [
            f"{feat}_sim" for feat in self.cluster_match_features
        ]
        df_relevant_top_init["sim_score_first"] = (
            df_relevant_top_init[features_rank].values * self.first_stage_weights
        ).sum(axis=-1) / self.first_stage_weights.sum()
        df_ranked = df_relevant_top_init.sort_values("sim_score_first", ascending=False)
        return df_ranked.head(self.top_n_first_stage)

    def __preprocess_cvs(self, df_relevant: pd.DataFrame):
        df_relevant["Описание"] = df_relevant["Описание"].fillna("Нет данных")
        new_cats = {}
        for cat in self.cats_find_cv:
            corpus = [
                "[SEP]".join(
                    [
                        df_relevant.iloc[i]["Описание"],
                        "cv",
                        cat,
                        self.category_desc.get(cat, "None"),
                        df_relevant.iloc[i]["Навыки"] if cat == "знания" else "None",
                    ]
                )
                for i in range(df_relevant.shape[0])
            ]
            new_cats[cat] = process_corpus(corpus=corpus, func=self.find_info)
            found_values = []
            for i in range(df_relevant.shape[0]):
                info_extracted = new_cats[cat][i]
                info_dict = self.postprocess_extracted_info(info_extracted, cat=cat)
                found_values.append(info_dict[cat])
            df_relevant[cat.capitalize()] = found_values

        return df_relevant

    def rank_second_stage(self, vacancy: Dict, df_relevant: pd.DataFrame):
        vacancy_prep = deepcopy(vacancy)
        vac_desc = vacancy_prep["Описание"]
        df_relevant = self.__preprocess_cvs(df_relevant=df_relevant)

        if self.method == Method.EMBEDDINGS:
            logger.info("Computing embeddings")
            for feat in self.feats_match:
                df_relevant[feat] = df_relevant[feat].fillna("Нет данных")
                embeddings = self.embedder.generate_embeddings(df_relevant, feat)
                embeddings_np = np.vstack(embeddings)
                embedding_vac = self.embedder.embed_corpus([vacancy_prep[feat]])
                embedding_vac_np = np.array(embedding_vac)
                cos_sims = cosine_similarity(embedding_vac_np, embeddings_np)[0]
                df_relevant[f"{feat}_sim"] = cos_sims
            for feat in self.feats_match_prompt:
                corpus = (
                    df_relevant[feat]
                    .apply(
                        lambda x: "[SEP]".join(
                            [f"{feat}:\n" + vacancy_prep[feat], f"{feat}:\n" + x]
                        )
                    )
                    .to_list()
                )
                sim_scores = process_corpus(
                    corpus=corpus,
                    func=self.match_prompt,
                    num_workers=self.request_num_workers,
                )
                df_relevant[f"{feat}_sim"] = sim_scores
        ## currently not working properly
        elif self.method == Method.PROMPT:
            logger.info("Computing scores with prompt")
            corpus_descs = (
                df_relevant["Full_description"]
                .apply(lambda x: "[SEP]".join([vac_desc, x]))
                .to_list()
            )
            sim_scores = process_corpus(
                corpus=corpus_descs,
                func=self.match_prompt,
                num_workers=self.request_num_workers,
            )
            df_relevant["Full_description_sim"] = sim_scores
        else:
            raise ValueError(
                f"Method doesn't exist: {self.method}, {self.method == Method.PROMPT}, {Method.PROMPT}"
            )

        nan_mask = self.vacancy_mask(vacancy_dict=vacancy_prep)
        weights = self.second_stage_weights * nan_mask
        sim_scores_names = [f"{feat}_sim" for feat in self.ranking_features]
        df_relevant["sim_score_second"] = (
            np.dot(df_relevant[sim_scores_names].values, weights) / weights.sum()
        )
        df_ranked = df_relevant.sort_values("sim_score_second", ascending=False)
        return df_ranked.head(self.top_n_second_stage), vacancy_prep, nan_mask


if __name__ == "__main__":
    config_path = "./config/config_mass.yaml"
    data_path = "./data_mass/Водитель белаза.csv"
    vac_path = "./data_mass/vacancies.csv"

    ## Load test data
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading test data")
    df_relevant = pd.read_csv(data_path)
    vacancy = pd.read_csv(vac_path).iloc[0].to_dict()

    ## Init model
    logger.info("Init model")
    config_model = config["model"]
    selector = MassCvSelector(config=config_model, api_token=os.getenv("OPENAI_TOKEN"))

    ## Make 1st stage ranking
    logger.info("1st stage ranking..")
    df_ranked_1st = selector.rank_first_stage(
        vacancy=vacancy, df_relevant=df_relevant.copy()
    )

    assert df_ranked_1st.shape[0] == config_model["stage_1"]["top_n"], "Wrong size"
    assert (
        df_ranked_1st.iloc[0]["id"] == "№ 2880355971"
    ), "First stage ranking is wrong!"
    logger.info("Finished successfully")

    # logger.info("2nd stage ranking..")
    df_ranked_2nd, vacancy_prep, nan_mask = selector.rank_second_stage(
        vacancy=vacancy, df_relevant=df_ranked_1st.copy()
    )

    assert df_ranked_2nd.shape[0] == config_model["stage_2"]["top_n"], "Wrong size"
    df_ranked_2nd.to_csv("./test_results.csv", index=False)
    assert (
        df_ranked_2nd.iloc[1]["id"] == "№ 2880355971"
    ), "Second stage ranking is wrong!"
    logger.info("Finished successfully")
