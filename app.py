import pandas as pd
import streamlit as st

from utils.utils import *

if "computed" not in st.session_state:
    st.session_state["computed"] = False


st.title("Подбор кандидатов 💼")

method = st.select_slider(
    "Выберите метод",
    options=[
        "Метод 1",
        "Метод 2",
    ],
)

vacancy_df = load_data(path="./data/vacancies.csv")
selector = load_model(config_path="./config/config.yaml", method=method)

vacancies = vacancy_df["Должность"].to_list()


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
            st.write(f"Первая фаза: анализ {df_cv.shape[0]} кандидатов")
            df_ranked_1st = selector.rank_first_stage(
                vacancy=vacancy, df_relevant=df_cv.copy()
            )
            st.write(f"Вторая фаза: анализ {df_ranked_1st.shape[0]} кандидатов")
            df_ranked_2nd = selector.rank_second_stage(
                vacancy=vacancy, df_relevant=df_ranked_1st.copy()
            )
            data_cv = df2dict(df_ranked_2nd)
            st.session_state["computed"] = True
        if st.session_state["computed"]:
            st.subheader("Кандидаты", divider="blue")
            for key in data_cv:
                key_ = data_cv[key]["Должность"]
                if "(" in key and ")" not in key:
                    key_ += ")"
                key_ += f" ({round(data_cv[key]['sim_score_second'] * 100)}% match)"
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

                    col1_cv, col2_cv = st.columns(2)
                    with col1_cv:
                        col1_container_cv = st.container(border=True, height=150)
                        col1_container_cv.caption("Профессиональная область")
                        match_score = round(data_cv[key]["prof_field_full_sim"] * 100)
                        accent_color = select_color(match_score)
                        col1_container_cv.markdown(
                            f":{accent_color}[{match_score}% match]"
                        )
                        formated_text = format_intersection(
                            vacancy["Профессиональная область"],
                            data_cv[key]["Профессиональная область"],
                        )
                        col1_container_cv.markdown(formated_text)
                    with col2_cv:
                        col2_container_cv = st.container(border=True, height=150)
                        col2_container_cv.caption("Список навыков")
                        match_score = round(data_cv[key]["Список навыков_sim"] * 100)
                        accent_color = select_color(match_score)
                        col2_container_cv.markdown(
                            f":{accent_color}[{match_score}% match]"
                        )
                        formated_text = format_intersection(
                            vacancy["Список навыков"], data_cv[key]["Список навыков"]
                        )
                        col2_container_cv.markdown(formated_text)

                    container0 = st.container(border=True)
                    container0.caption("Зарплатные ожидания")
                    container0.write(data_cv[key]["Зарплатные ожидания"])
                    container1 = st.container(border=True)
                    container1.caption("Образование")
                    container1.write(data_cv[key]["Образование"])
                    container2 = st.container(border=True)
                    container2.caption("Опыт")
                    container2.write(data_cv[key]["Опыт"])
                    container3 = st.container(border=True)
                    container3.caption("Описание")
                    container3.write(data_cv[key]["Описание"])
