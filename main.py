import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# ----------------------------------- І. ПІДГОТОВКА ВХІДНИХ ДАНИХ ------------------------------
# 1.1. Парсінг файлу вхідних даних
input_dataframe = pd.read_excel('input_files/Pr15_sample_data.xlsx', parse_dates=['birth_date'])
print('------------------ вхідний масив ----------------------')
print(input_dataframe.head(5))
print('-------------------------------------------------------')

# 1.2. Аналіз структури вхідних даних
print('---------  пропущені значення стовпців (суми)  --------')
print(input_dataframe.isnull().sum())

# 1.3. Парсінг файлу та аналіз структури скорингових індикаторів
score_ind_input = pd.read_excel('input_files/Pr15_data_description.xlsx')
print('---------------  скорингові індикатори  ---------------')
print(score_ind_input.head(5))
print('-------------------------------------------------------')

# 1.4. Первинне формування скорингової таблиці
score_table_v1 = score_ind_input[(score_ind_input.Place_of_definition == 'Указывает заемщик')
                                 | (score_ind_input.Place_of_definition == 'параметры связанные с выданным продуктом')]
n_client_bank = score_table_v1['Place_of_definition'].size
score_table_v1.index = range(0, len(score_table_v1))
print('-------------  первинна скор. таблиця -----------------')
print(score_table_v1.head(5))
score_table_v1.to_csv('results/1_score_table_v1.csv')
print('-------------------------------------------------------')

# 1.5. Очищення даних
print('1.5.1. Аналіз перетину скорингових індикаторів та сегменту вхідних даних')
b = score_table_v1['Field_in_data']
if set(b).issubset(input_dataframe.columns):
    Flag_b = 'Flag_True'
else:
    Flag_b = 'Flag_False'
# ------ кількість співпадінь
n_columns = score_table_v1['Field_in_data'].size
j = 0
for i in range(0, n_columns):
    a = score_table_v1['Field_in_data'][i]
    if set([a]).issubset(input_dataframe.columns):
        j = j + 1
print('к-ть співпадінь: ', j)
Columns_Flag_True = np.zeros((j))
j = 0
for i in range(0, n_columns):
    a = score_table_v1['Field_in_data'][i]
    if set([a]).issubset(input_dataframe.columns):
        Flag = 'Flag_True'
        Columns_Flag_True[j] = i
        j = j + 1
    else:
        Flag = 'Flag_False'
print('Індекси співпадінь', Columns_Flag_True)

# 1.5.2. Формування DataFrame даних з урахуванням відсутніх індикаторів скорингової таблиці
score_table_wout_ind = score_table_v1.iloc[Columns_Flag_True]
score_table_wout_ind.index = range(0, len(score_table_wout_ind))
print('------------ DataFrame співпадінь -------------')
print(score_table_wout_ind.head(5))
score_table_wout_ind.to_csv('results/2_score_table_wout_ind.csv')
print('-----------------------------------------------')

# 1.5.3. Очищення скорингової таблиці від пропусків
b = score_table_wout_ind['Field_in_data']
d_data_client_bank = input_dataframe[b]
print('---- пропуски даних сегменту DataFrame --------')
print(d_data_client_bank.isnull().sum())
print('-----------------------------------------------')

# ------- СКОРИНГОВА КАРТА -------
# Очищення індикаторів скорингової таблиці
scoring_map = score_table_wout_ind.loc[(score_table_wout_ind['Field_in_data'] != 'fact_addr_start_date')]
scoring_map = scoring_map.loc[(scoring_map['Field_in_data'] != 'position_id')]
scoring_map = scoring_map.loc[(scoring_map['Field_in_data'] != 'employment_date')]
scoring_map = scoring_map.loc[(scoring_map['Field_in_data'] != 'has_prior_employment')]
scoring_map = scoring_map.loc[(scoring_map['Field_in_data'] != 'prior_employment_start_date')]
scoring_map = scoring_map.loc[(scoring_map['Field_in_data'] != 'prior_employment_end_date')]
scoring_map = scoring_map.loc[(scoring_map['Field_in_data'] != 'income_frequency_other')]
scoring_map.index = range(0, len(scoring_map))
scoring_map.to_csv('results/3_scoring_map.csv')

useless_columns = ['gender_id', 'birth_date', 'children_count_id', 'fact_addr_owner_type_id', 'fact_addr_start_date',
                   'employment_type_id', 'position_id', 'organization_type_id', 'organization_branch_id',
                   'empoyees_count_id', 'employment_date', 'seniority_years', 'has_prior_employment',
                   'prior_employment_start_date', 'prior_employment_end_date', 'income_frequency_other',
                   'income_source_id', 'other_loans_has_closed', 'other_loans_active', 'other_loans_about_current',
                   'other_loans_about_monthly', 'product_amount_from', 'product_amount_to',
                   'product_base_amount_limit', 'amount_limit']

# Очищення вхідних даних
d_clean = d_data_client_bank.drop(columns=useless_columns)
d_clean.index = range(0, len(d_clean))
d_clean.to_csv('results/4_d_clean.csv')
print(d_clean.isnull().sum())
print('---------- DataFrame вхідних даних - скорингова карта -----------')
print(d_clean.head(5))
print('----------------- DataFrame індикатори скорингу  ----------------')
print(scoring_map)
print('-----------------------------------------------------------------')

# ----------------------------------- ІІ. ФОРМУВАННЯ СКОРИНГОВОЇ МОДЕЛІ ------------------------------
# 2.1. Парсінг файлу індикаторів (критеріїв) скорингової карти
data_desc_minimax = pd.read_excel('input_files/data_desc_cleaning_minimax.xlsx')
df_desc_minimax = data_desc_minimax.loc[(data_desc_minimax['Minimax'] == 'min')
                                        | (data_desc_minimax['Minimax'] == 'max')]
df_desc_minimax.index = range(0, len(df_desc_minimax))
print('----------------- DataFrame df_desc_minimax  ----------------')
print(df_desc_minimax.head(5))
print('-----------------------------------------------------------------')
df_desc_minimax.to_csv('results/5_df_desc_minimax.csv')

# відбір даних за критеріями
d = df_desc_minimax['Field_in_data']
cols = d.values.tolist()
df_sample_minimax = d_clean[cols]
print('----------------- DataFrame df_sample_minimax  ----------------')
print(df_sample_minimax)
print('-----------------------------------------------------------------')
df_sample_minimax.to_csv('results/6_df_sample_minimax.csv')

# 2.2. Парсінг файлу індикаторів (критеріїв) скорингової карти
d_sample_min = df_sample_minimax[cols].min()
d_sample_max = df_sample_minimax[cols].max()
print('----------------- DataFrame: d_segment_sample_min  ----------------')
print(d_sample_min.head(5))
d_sample_min.to_csv('results/7_d_sample_min.csv')
print('----------------- DataFrame: d_segment_sample_max  ----------------')
print(d_sample_max.head(5))
d_sample_min.to_csv('results/8_d_sample_max.csv')

# 2.3.Нормування критеріїв
m = df_sample_minimax['loan_amount'].size
n = df_desc_minimax['Field_in_data'].size
d_minimax_norm = np.zeros((m, n))

delta_d = 0.3
for j in range(0, len(df_desc_minimax)):
    columns_d = df_desc_minimax['Minimax'][j]
    if columns_d == 'min':
        columns_m = df_desc_minimax['Field_in_data'][j]
        for i in range(0, len(df_sample_minimax)):
            max_max = d_sample_max[j] + (2 * delta_d)
            d_minimax_norm[i, j] = (delta_d + df_sample_minimax[columns_m][i]) / (max_max)
    else:
        for i in range(0, len(df_sample_minimax)):
            min_min = d_sample_max[j] + (2 * delta_d)
            d_minimax_norm[i, j] = (1 / (delta_d + df_sample_minimax[columns_m][i])) / (min_min)

print('----------------- Нормування критеріїв  ----------------')
print(d_minimax_norm)
np.savetxt('results/9_d_minimax_norm.txt', d_minimax_norm)
print('--------------------------------------------------------')


# 2.4.Інтегрована багатокритеріальна оцінка - SCOR
def Voronin(d_minimax_norm, n):
    Integro = np.zeros(499)
    Scor = np.zeros(499)
    for i in range(0, 499):
        Sum_Voronin = 0
        for j in range(0, n):
            Sum_Voronin = Sum_Voronin + ((1 - d_minimax_norm[i, j]) ** (-1))
        Integro[i] = Sum_Voronin
        Scor[i] = 1000
        np.savetxt('results/10_integro_scor.txt', Integro)
    plt.title('Integro_Scor до роботи з аномаліями')
    plt.plot(Integro)
    plt.show()
    return Integro

integro_score = Voronin(d_minimax_norm, n)
print('---------------- SCOR -------------------')
print(integro_score)

# ------------------------- РОБОТА З АНОМАЛІЯМИ ------------------------------
zscore = stats.zscore(integro_score)
d_w_score = df_sample_minimax.copy()
d_w_score['Score'] = integro_score
d_w_score['Zscore'] = zscore
d_w_score.to_csv('results/11_with_score.csv')

d_wout_anom = d_w_score[d_w_score['Zscore'] < 2]
integro_score = d_wout_anom['Score'].tolist()
plt.title('Integro_Scor після роботи з аномаліями')
plt.plot(integro_score)
plt.show()
d_wout_anom.to_csv('results/12_without_anomalies.csv')

print('--------------------------------------------------------')
print('Shape до роботи з аномаліями: ', d_w_score.shape)
print('Shape після роботи з аномаліями: ', d_wout_anom.shape)

# -------------------------- ЧИ ДАВАТИ КРЕДИТ? (таблиця)---------------------------------
d_wout_anom.loc[d_wout_anom['Score'] > d_wout_anom['Score'].mean(), 'Give_credit'] = 1
d_wout_anom.loc[d_wout_anom['Score'] < d_wout_anom['Score'].mean(), 'Give_credit'] = 0
d_wout_anom.to_csv('results/13_give_loan.csv')

# -------------------------- ЧИ ДАВАТИ КРЕДИТ? (візуалізація)---------------------------------
# кластеризація
sns.scatterplot(data=d_wout_anom, x='loan_days', y='loan_amount', hue='Give_credit',
                c=['blue', 'yellow'])
plt.title('Чи давати кредит?')
plt.show()
#видалення людей яким не дадуть кредит
d_give_loan = d_wout_anom[d_wout_anom['Score'] > d_wout_anom['Score'].mean()]
integro_score = d_give_loan['Score'].tolist()
plt.title('Integro_Score людей з одобреним кредитом')
plt.plot(integro_score)
plt.show()

# ---------------------------ЧИ ПОВЕРНЕ КРЕДИТ? (таблиця, діаграма) -----------------------------------
d_return_loan = d_give_loan[d_give_loan['Score'] > d_give_loan['Score'].mean()]
d_return_loan.loc[d_give_loan['Score'] > d_return_loan['Score'].mean(), 'Return_credit'] = 1
d_return_loan.loc[d_give_loan['Score'] < d_return_loan['Score'].mean(), 'Return_credit'] = 0
d_return_loan.to_csv('results/14_return_loan.csv')

sns.scatterplot(data=d_return_loan, x='loan_days', y='loan_amount', hue='Return_credit',
                c=['blue', 'yellow'])
plt.title('Чи поверне кредит?')
plt.show()

