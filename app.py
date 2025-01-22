import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.hardcode_data import candidate_names, map_names
from utils.utils import *

if "computed" not in st.session_state:
    st.session_state["computed"] = False


st.title("Подбор кандидатов 💼")

vacancy_df = load_data(path="./data/vacancies.csv")
selector, config = load_model(config_path="./config/config.yaml")

vacancies = vacancy_df["Должность"].to_list()

features = deepcopy(config["model"]["stage_2"]["ranking_features"])
info_dict = {"Компонента": [], "Вес": []}
for feature, value in zip(features, config["model"]["stage_2"]["weights"]):
    info_dict["Компонента"].append(
        feature if feature not in map_names else map_names[feature]
    )
    info_dict["Вес"].append(round(value, 2))

df_info = pd.DataFrame(info_dict)
with st.sidebar:
    st.header("Информация ℹ️")
    st.write("Веса компонент скоринга.")
    st.markdown(df_info.to_html(escape=False), unsafe_allow_html=True)

option = st.selectbox(
    "Вакансии",
    vacancies,
    index=None,
    placeholder="Выберете вакансию...",
)

if option is not None:
    vacancy = vacancy_df[vacancy_df["Должность"] == option].iloc[0].to_dict()
    st.header(f"{option}", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        col1_container = st.container(border=True, height=100)
        col1_container.caption("Профессиональная область")
        col1_container.markdown(f"**{vacancy['prof_field_full']}**")
    with col2:
        col2_container = st.container(border=True, height=100)
        text = vacancy["Список навыков"]
        col2_container.caption("Список навыков")
        col2_container.markdown(f"**{text}**")
    container = st.container(border=True)
    container.caption("Описание")
    container.write(vacancy["Описание"])
    if st.button("Подобрать", type="primary"):
        df_cv = load_data(f"./data/{option}.csv")
        with st.status("Подбор кандидатов..."):
            if (
                not Path("./tmp_cvs.csv").exists()
                or config["general"]["mode"] == "prod"
            ):
                st.write(f"Первая фаза: анализ {df_cv.shape[0]} кандидатов..")
                df_ranked_1st = selector.rank_first_stage(
                    vacancy=vacancy, df_relevant=df_cv.copy()
                )
                st.write(f"Вторая фаза: анализ {df_ranked_1st.shape[0]} кандидатов..")
                df_ranked_2nd, vacancy_prep, nan_mask = selector.rank_second_stage(
                    vacancy=vacancy, df_relevant=df_ranked_1st.copy()
                )
                if config["general"]["mode"] != "prod":
                    df_ranked_2nd.to_csv("./tmp_cvs.csv", index=False)
                    with open("./tmp_vac.json", "w") as f:
                        json.dump(vacancy_prep, f, ensure_ascii=False)
                    np.save("./tmp_mask.npy", nan_mask)
            else:
                df_ranked_2nd = pd.read_csv("./tmp_cvs.csv")
                with open("./tmp_vac.json", "r") as f:
                    vacancy_prep = json.load(f)
                nan_mask = np.load("./tmp_mask.npy")

            data_cv = df2dict(df_ranked_2nd)
            st.session_state["computed"] = True
            st.write(f"Выбрано {df_ranked_2nd.shape[0]} лучших кандидатов.")
        if st.session_state["computed"]:
            nan_mask = np.delete(nan_mask, [1, 2, 5])
            st.subheader("Кандидаты", divider="blue")
            for key in data_cv:
                col1_results, col2_cv = st.columns(2)
                key_ = data_cv[key]["Должность"]
                if "(" in key and ")" not in key:
                    key_ += ")"
                key_ += f" ({round(data_cv[key]['sim_score_second'] * 100)}% match)"
                key_ = np.random.choice(candidate_names) + f" - {key_}"
                with st.expander(key_):
                    match_score_first = round(data_cv[key]["sim_score_first"] * 100)
                    accent_color = select_color(match_score_first)
                    st.markdown(
                        f"Первая фаза: :{accent_color}[{match_score_first}% match]"
                    )

                    match_score_second = round(data_cv[key]["sim_score_second"] * 100)
                    accent_color = select_color(match_score_second)
                    st.markdown(
                        f"Вторая фаза: :{accent_color}[{match_score_second}% match]"
                    )

                    match_score_full_desc = round(
                        data_cv[key]["Full_description_sim"] * 100
                    )
                    accent_color = select_color(match_score_second)
                    st.markdown(
                        f"Похожесть по полному описанию: :{accent_color}[{match_score_full_desc}% match]"
                    )

                    ranking_features = deepcopy(
                        config["model"]["stage_2"]["ranking_features"]
                    )
                    ranking_features.remove("Должность категория")
                    ranking_features.remove("Должность подкатегория")
                    ranking_features.remove("Full_description")
                    job_labels = [
                        "Должность_sim",
                        "Должность_cat_sim",
                        "Должность_subcat_sim",
                    ]
                    data_cv[key]["Должность_sim"] = (
                        sum([data_cv[key][job_label] for job_label in job_labels]) / 3
                    )
                    for i, feature in enumerate(ranking_features):
                        col_results_1, col_results_2, col_results_3 = st.columns(
                            [2, 1, 2], gap="small", vertical_alignment="center"
                        )
                        num_rows = (
                            max(len(data_cv[key][feature]), len(vacancy_prep[feature]))
                            / 20
                        )
                        container_height = round(num_rows * 30) + 60
                        with col_results_1:
                            if i == 0:
                                st.header("Кандидат")
                            container_cv = st.container(
                                border=True, height=container_height
                            )
                            feature_print = feature
                            if feature in map_names:
                                feature_print = map_names[feature]
                            container_cv.caption(feature_print)
                            formated_text = format_intersection(
                                vacancy_prep[feature],
                                data_cv[key][feature],
                            )
                            container_cv.markdown(formated_text.capitalize())

                        with col_results_2:
                            if i == 0:
                                st.header(" ")
                            container_score = st.container(
                                border=True, height=container_height
                            )
                            match_score = round(data_cv[key][f"{feature}_sim"] * 100)
                            flag_vac = False
                            flag_cv = False
                            if nan_mask[i] == 0:
                                match_score = 0
                                flag_vac = True
                            if data_cv[key][feature].lower() in [
                                "нет данных",
                                "нет информации",
                                "",
                                "none",
                                "не указано",
                                "не указана",
                            ]:
                                match_score = 0
                                flag_cv = True
                            if flag_vac * flag_cv:
                                match_score = 0
                            accent_color = select_color(match_score)
                            container_score.markdown(
                                "<br>" * int((num_rows // 2)), unsafe_allow_html=True
                            )
                            container_score.markdown(
                                f":{accent_color}[{match_score}%\nmatch]"
                            )

                        with col_results_3:
                            if i == 0:
                                st.header("Вакансия")
                            container_vac = st.container(
                                border=True, height=container_height
                            )
                            feature_print = feature
                            if feature in map_names:
                                feature_print = map_names[feature]
                            container_vac.caption(feature_print)
                            formated_text = format_intersection(
                                data_cv[key][feature], vacancy_prep[feature]
                            )
                            container_vac.markdown(formated_text.capitalize())
                        if feature == "Должность":
                            st.info(
                                "Указано среднее значение 3 скоров для компонент, связанных с Должностью.",
                                icon="ℹ️",
                            )
                        if i < len(ranking_features) - 1:
                            st.divider()
