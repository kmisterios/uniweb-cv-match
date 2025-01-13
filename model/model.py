import json
import os
from json import JSONDecodeError
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from .request_api import OpenAIEmbedder, process_corpus

load_dotenv()


class CvSelector:
    def __init__(self, config: Dict, api_token: str):
        ## stage 1 config
        self.text_features = config["stage_1"]["text_features"]
        self.cluster_match_features = config["stage_1"]["cluster_match_features"]
        self.first_stage_weights = np.array(config["stage_1"]["weights"])
        self.top_n_first_stage = config["stage_1"]["top_n"]
        self.ranking_features_first_stage = None
        ## stage 2 config
        self.keys_vacancy = config["stage_2"]["keys_vacancy"]
        self.model_name = config["stage_2"]["model_name"]
        self.prompt_experience = config["stage_2"]["prompt_experience"]
        self.system_prompt_experience = config["stage_2"]["system_prompt_experience"]
        self.request_num_workers = config["stage_2"]["request_num_workers"]
        self.keys_cv = config["stage_2"]["keys_cv"]
        self.second_stage_weights = np.array(config["stage_2"]["weights"])
        self.top_n_second_stage = config["stage_2"]["top_n"]
        self.embedder = OpenAIEmbedder(
            api_key=api_token, model_name=config["stage_2"]["model_name_embed"]
        )
        self.api_token = api_token

    def __text_features_intersection(self, str_feats_ref: str, str_feats_match: str):
        set1 = set(str_feats_ref.lower().split(", "))
        set2 = set(str_feats_match.lower().split(", "))
        return len(set1.intersection(set2)) / len(set1)

    def __get_desc(self, vacancy: Dict, keys: List[str]):
        description_items = []
        for key in keys:
            description_items.append(f"{key}:\n{vacancy[key]}")
        return "\n\n".join(description_items)

    def __education_str(self, edu_data_str: str):
        result = ""
        try:
            edu_data = json.loads(edu_data_str)
        except JSONDecodeError:
            return "Нет данных"
        for i, data_item in enumerate(edu_data):
            for key in data_item:
                if data_item[key] is None or data_item[key] == "None":
                    data_item[key] = ""
            data_item_str = f"\n{i + 1}. {data_item['year']} - {data_item['name']}. {data_item['result']}, {data_item['organization']}"
            while not data_item_str[-1].isalpha():
                data_item_str = data_item_str[:-1]
                if len(data_item_str) == 0:
                    break
            if len(data_item_str) == 0:
                continue
            result += data_item_str.replace(" , ", " ") + "."
        if result == "":
            return "Нет данных"
        return result

    def __salary_str(self, salary_data_str: str):
        try:
            salary_data = json.loads(salary_data_str)
        except JSONDecodeError:
            return "Нет данных"
        return f"{salary_data['amount']} {salary_data['currency']}"

    def work_experience_summary(self, json_str: str):
        client = OpenAI(api_key=self.api_token)
        prompt = self.prompt_experience + json_str

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt_experience},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def rank_first_stage(self, vacancy: Dict, df_relevant: pd.DataFrame):
        ranking_features = self.cluster_match_features.copy()
        for feat in self.text_features:
            df_relevant[f"{feat}_sim"] = (
                df_relevant[feat]
                .fillna("None")
                .apply(lambda x: self.__text_features_intersection(vacancy[feat], x))
            )
            ranking_features.append(f"{feat}_sim")
        self.ranking_features_first_stage = ranking_features

        df_relevant["sim_score_first"] = (
            np.dot(df_relevant[ranking_features].values, self.first_stage_weights)
            / self.first_stage_weights.sum()
        )
        df_ranked = df_relevant.sort_values("sim_score_first", ascending=False)
        return df_ranked.head(self.top_n_first_stage)

    def rank_second_stage(self, vacancy: Dict, df_relevant: pd.DataFrame):
        vacancy["Профессиональная область"] = vacancy.pop("prof_field_full")
        vacancy["Должность категория"] = vacancy.pop("Должность_cat")
        vacancy["Должность подкатегория"] = vacancy.pop("Должность_subcat")
        vac_desc = self.__get_desc(vacancy=vacancy, keys=self.keys_vacancy)
        corpus_experience = df_relevant["Опыт"].fillna("").to_list()
        logger.info("Generating experience summaries")
        summaries_experience = process_corpus(
            corpus=corpus_experience,
            func=self.work_experience_summary,
            num_workers=self.request_num_workers,
        )
        df_relevant = df_relevant.rename(columns={"Опыт": "Опыт raw"})
        df_relevant["Опыт"] = summaries_experience
        df_relevant["Образование"] = df_relevant["Образование.Высшее"].apply(
            lambda x: self.__education_str(x)
        )
        df_relevant["Зарплатные ожидания"] = df_relevant["Зарплата"].apply(
            self.__salary_str
        )
        df_relevant = df_relevant.rename(
            columns={
                "prof_field_full": "Профессиональная область",
                "Должность_cat": "Должность категория",
                "Должность_subcat": "Должность подкатегория",
            }
        )
        descs = []
        for i in range(len(df_relevant)):
            cv_dict = df_relevant.iloc[i].to_dict()
            desc = self.__get_desc(cv_dict, keys=self.keys_cv)
            descs.append(desc)
        df_relevant["Full_description"] = descs
        logger.info("Computing descriptions embeddings")
        embeddings = self.embedder.generate_embeddings(df_relevant, "Full_description")
        embeddings_np = np.vstack(embeddings)
        embedding_vac = self.embedder.embed_corpus([vac_desc])
        embedding_vac_np = np.array(embedding_vac)
        cos_sims = cosine_similarity(embedding_vac_np, embeddings_np)[0]
        df_relevant["full_desc_emb_sim"] = cos_sims
        ranking_features = self.ranking_features_first_stage + ["full_desc_emb_sim"]
        df_relevant["sim_score_second"] = (
            np.dot(df_relevant[ranking_features].values, self.second_stage_weights)
            / self.second_stage_weights.sum()
        )
        df_ranked = df_relevant.sort_values("sim_score_second", ascending=False)
        return df_ranked.head(self.top_n_second_stage)


if __name__ == "__main__":
    config_path = "../config/config.yaml"
    data_path = "../../uniweb-demo/notebooks/data_jobs/test_cvs_subset.csv"
    vac_path = "../../uniweb-demo/notebooks/data_jobs/test_vacancy.csv"

    ## Load test data
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading test data")
    df_relevant = pd.read_csv(data_path)
    vacancy = pd.read_csv(vac_path, index_col="Unnamed: 0")["45"].to_dict()

    ## Init model
    logger.info("Init model")
    config_model = config["model"]
    selector = CvSelector(config=config_model, api_token=os.getenv("OPENAI_TOKEN"))

    ## Make 1st stage ranking
    logger.info("1st stage ranking..")
    df_ranked_1st = selector.rank_first_stage(
        vacancy=vacancy, df_relevant=df_relevant.copy()
    )

    assert df_ranked_1st.shape[0] == config_model["stage_1"]["top_n"], "Wrong size"
    assert (
        "Некула" in df_ranked_1st.iloc[0]["Описание"]
    ), "First stage ranking is wrong!"
    logger.info("Finished successfully")

    logger.info("2nd stage ranking..")
    df_ranked_2nd = selector.rank_second_stage(
        vacancy=vacancy, df_relevant=df_ranked_1st.copy()
    )

    assert df_ranked_2nd.shape[0] == config_model["stage_2"]["top_n"], "Wrong size"
    assert (
        "3500 EUR" in df_ranked_2nd.iloc[0]["Зарплатные ожидания"]
    ), "Second stage ranking is wrong!"
    logger.info("Finished successfully")
    df_ranked_2nd.to_csv("./test_results.csv", index=False)
