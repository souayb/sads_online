import json
import pathlib
from itertools import count
from collections import Counter
from turtle import width
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os 
import pickle 
from datetime import datetime
# from pickle import dump as pkl_dump, load as pkl_load
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn import mixture
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from batch_preprocess import Preprocessing
import pandas as pd
from matplotlib.colors import ListedColormap
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt
import json_logging, logging, sys
from pdb import set_trace

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

from shift_detect import pettitt_test

# """
# pip install streamlit-aggrid
# """


import base64
# caching.clear_cache()
st.set_page_config(layout="wide") # setting the display in the 

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        button[data-baseweb="tab"] {font-size: 26px;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

SMALL_SIZE = 3
MEDIUM_SIZE =3
BIGGER_SIZE = 3
# plt.rcParams['figure.figsize'] = (5, 10)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE, dpi=600)  # fontsize of the figure title
plt.style.context('bmh')
new_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> SADS: Shop-floor Anomaly Detection Service: Offine mode </p> </h2></center>'

st.markdown(new_title, unsafe_allow_html=True)

@st.cache(suppress_st_warning=True)
# @st.experimental_memo(suppress_st_warning=True)
def data_reader(dataPath:str) -> pd.DataFrame :
    df = pd.read_csv(dataPath, decimal=',')
    prepro = Preprocessing()
    data = prepro.preprocess(df)
    # data = data[['BarCode', 'Face', 'Cell', 'Point', 'Group' , 'Output Joules' , 'Charge (v)', 'Residue (v)', 'Force L N','Force L N_1', 'Y/M/D hh:mm:ss']]
    data.rename(columns={'BarCode':'Barcode', 'Output Joules': 'Joules', 'Charge (v)':'Charge', 'Residue (v)':'Residue','Force L N':'Force_N', 'Force L N_1':'Force_N_1', 'Y/M/D hh:mm:ss': 'Datetime'}, inplace=True)
    data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']] = data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].apply(np.float32)
    data[['Face', 'Cell', 'Point']] = data[['Face', 'Cell', 'Point']].values.astype( int )
    JOULES = data['Joules'].values 
    return data[['Barcode', 'anomaly', 'Face', 'Cell', 'Point','Face_Cell_Point','Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'ts']]           

def train_model(data, model_type:str='ifor'):
    data = data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1',]].values
    # set_trace()
    # scaler_new = MinMaxScaler()
    # data = scaler_new.fit_transform(data)
    if model_type =='ifor':
        ifor = Pipeline([
            ('scaler',  StandardScaler()),
            ('clf', IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.01),random_state=0))])
        ifor.fit(data)
        # cluster = ifor.predict(data)
        return ifor
    elif model_type == 'gmm':
        gmm = Pipeline([
            ('scaler',  StandardScaler()),
            ('clf', mixture.GaussianMixture(n_components=2, covariance_type="full"))
        ])
        gmm.fit(data)
        return gmm

    elif model_type == 'bgmm':
        bgmm = Pipeline([
            ('scaler',  StandardScaler()),
            ('clf', mixture.BayesianGaussianMixture(n_components=2, covariance_type="full"))
        ])
        bgmm.fit(data)
        return bgmm


    elif model_type =='lof':
        lof = Pipeline([
            ('scaler',  StandardScaler()),
            ('clf', LocalOutlierFactor(n_neighbors=2, novelty=True))
        ])
        lof.fit(data)
        return lof
    
    elif model_type =='svm':
        svm = Pipeline([
            ('scaler',  StandardScaler()),
            ('clf', OneClassSVM(gamma='auto'))
        ])
        svm.fit(data)
        return svm
    


# ########################## PREDICTION FORM #######################################
# SADA_settings = st.sidebar.form("SADS")
# SADA_settings.title("SADS settings")

SADS_CONFIG_FILE = 'sads_config.json'
SADS_CONFIG = {}
JOULES = []
SHIFT_DETECTED = False
SHIFT_RESULT = []
RESULT_CHANGED = False

# Max_battery_pack = SADA_settings.slider("Max battery pack", min_value=1, max_value=20, step=1)
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_logger(save:bool=True):
    """
    Generic utility function to get logger object with fixed configurations
    :return:
    logger object
    """
    SADS_CONFIG['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    SADS_CONFIG['drift_detected'] = SHIFT_DETECTED
    SADS_CONFIG['Joules'] = JOULES
    SADS_CONFIG['drift_result'] = SHIFT_RESULT
    SADS_CONFIG['result_change'] =  RESULT_CHANGED
    if save:
        with open(SADS_CONFIG_FILE, 'w') as outfile:
            json.dump(SADS_CONFIG, outfile)
    else:
        with open(SADS_CONFIG_FILE) as infile:
            return json.load(infile)

with st.sidebar.container():
    st.title("SADS settings input") 
    training_type =     st.radio(
            "Apply on: 👇",
            ["Pack", "Whole"],
            disabled=False,
            horizontal= True,
        )
    with st.form('setting'):
        with st.expander('Model setting input'):
            # with st.form("Models"):
            st.subheader("SADS models")
            check_left, check_right = st.columns(2)
            model_ifor = check_left.checkbox('Isolation forest', value=True )
            model_lof = check_left.checkbox('Local Outlier Factor', value=False)
            model_repeat = check_left.checkbox('Repeat', value=False)
            model_gmm = check_right.checkbox('Gaussian Mixture', value=False)
            model_bgmm = check_right.checkbox('Bayesian gaussian Mixture', value=False)
            model_svm = check_right.checkbox('One Class SVM', value=False)
            # train_ = st.form_submit_button("Apply")

        # st.subheader("Table control input")
        with st.expander("Show Table control"):
        # with st.form('my_form'): 
            st.subheader("Table setting")
            sample_size = st.number_input("rows", min_value=10, value=30)
            grid_height = st.number_input("Grid height", min_value=200, max_value=800, value=300)

            return_mode = st.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
            return_mode_value = DataReturnMode.__members__[return_mode]

            # update_mode = st.selectbox("Update Mode", list(GridUpdateMode.__members__), index=len(GridUpdateMode.__members__)-1)
            # update_mode_value = GridUpdateMode.__members__[update_mode]

            #enterprise modules
            enable_enterprise_modules = st.checkbox("Enable Enterprise Modules")
            if enable_enterprise_modules:
                enable_sidebar =st.checkbox("Enable grid sidebar", value=False)
            else:
                enable_sidebar = False
            #features
            fit_columns_on_grid_load = st.checkbox("Fit Grid Columns on Load")

            enable_selection=st.checkbox("Enable row selection", value=True)

            if enable_selection:
                
                # st.sidebar.subheader("Selection options")
                selection_mode = st.radio("Selection Mode", ['single','multiple'], index=1)

                use_checkbox = st.checkbox("Use check box for selection", value=True)
                if use_checkbox:
                    groupSelectsChildren = st.checkbox("Group checkbox select children", value=True)
                    groupSelectsFiltered = st.checkbox("Group checkbox includes filtered", value=True)

                if ((selection_mode == 'multiple') & (not use_checkbox)):
                    rowMultiSelectWithClick = st.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
                    if not rowMultiSelectWithClick:
                        suppressRowDeselection = st.checkbox("Suppress deselection (while holding CTRL)", value=False)
                    else:
                        suppressRowDeselection=False
                st.text("___")

            enable_pagination = st.checkbox("Enable pagination", value=False)
            if enable_pagination:
                st.subheader("Pagination options")
                paginationAutoSize = st.checkbox("Auto pagination size", value=True)
                if not paginationAutoSize:
                    paginationPageSize = st.number_input("Page size", value=5, min_value=0, max_value=sample_size)
                st.text("___")

        with st.expander('Chart plot setting'):
            st.subheader("Plot setting")
            chart_left, chart_right = st.columns(2)
            show_joules = chart_left.checkbox('Joules', value=True)
            show_force_n = chart_left.checkbox('Force right', value=False)
            show_force_n_1 = chart_right.checkbox('Force left', value=False)
            show_residue = chart_right.checkbox('Residue', value=False)
            show_charge = chart_right.checkbox('Charge', value=False)


        submitted = st.form_submit_button('Apply')
# with open(SADS_CONFIG_FILE, "w") as outfile:
#     json.dump(SADS_CONFIG, outfile)
# stop_bt, rerun_bt, show_bt = SADA_settings.columns((1,1,2))
# SADS_submit = show_bt.form_submit_button("Predict") 
# stop_submit = stop_bt.form_submit_button("Stop") 
# rerun_submit = rerun_bt.form_submit_button("Rerun") 




# if rerun_submit:
#     st.experimental_rerun()
# if stop_submit:
#     st.stop()
#     # st.success('Thank you for inputting a name.')

# label_choice = SADA_settings.radio(
#         "Choose the labeling strategy",
#         ("Repeat labelling", "Isolation forest"))


# model_choice = SADA_settings.selectbox(
#         "Choose the model",
#         ("Random Forest", "XGboost", "Thresholding"))


# SADS_info = SADA_settings.expander("See explanation")
# SADS_info.markdown("""
# - Max length of the scatter: Maximum length of the plot 
# - Stop: Stop the simulation
# - 
# - Choose the model :
#     - ***Repeat***: Model based on the repeated labeling method
#         - Labeling strategy provided by the WAM technik
#     - ***Iforest***: Model base on Isolation forest labeling method 
#         - Unsupervised labeling mechanisme for anomalies. A clustering based method 

#         """)


# #
# # progress_bar = st.empty()
# # plot1, plot2 = st.columns((2, 1))
# #
# # title_main = st.empty()
# Main = st.empty()
# day_left, time_right = Main.columns(2)
# good_weld = day_left.empty()
# bad_weld = time_right.empty()
# # weld = st.empty()
# # anomaly_plot = plot1.empty()
# # forecast_plot = st.empty()
# # py_chart = plot2.empty()






uploaded_files = st.file_uploader("Choose a CSV file" )



# fig = make_subplots(rows=5, cols=1)


if uploaded_files is not None:

    if pathlib.Path ( uploaded_files.name ).suffix not in ['.csv', '.txt']:
        st.error ( "the file need to be in one the follwing format ['.csv', '.txt'] not {}".format (
            pathlib.Path ( uploaded_files.name ).suffix ) )
        raise Exception ( 'please upload the right file ' )

    with st.spinner('Wait for preprocess and model training'):
        st.info('Preporcesssing started ')
        data = data_reader(uploaded_files)
        new_joule = data['Joules'].values 
        st.success('Preprocessing complete !')
        if not os.path.exists(SADS_CONFIG_FILE):
            JOULES = new_joule.tolist()
            get_logger(save=True)
            IF = pickle.load(open('model.pkl', 'rb'))
        else : 
            SADS_CONFIG = get_logger(save=False)
            # SHIFT_RESULT = SADS_CONFIG['drift_result']
            # set_trace()
            # testing drift
            # set_trace()
            to_test = np.hstack([np.array(SADS_CONFIG['Joules'][:500]), new_joule[:500]])
            test_resutl = pettitt_test(to_test, alpha=0.8)
            if test_resutl.cp >= 500 and test_resutl.cp <= 502: 
                st.write("DRIFT FOUND NEED THE RETRAIN THE MODEL")
                JOULES = new_joule.tolist()
                SHIFT_DETECTED = True
                get_logger(save=True)
                if training_type=='Whole':
                    with st.spinner('Training...: This may take some time'):
                        IF = train_model(data=data)
                        pickle.dump(IF, open('model.pkl', 'wb'))
                        st.success('Training completed !')
                    
            else : 
                # JOULES = new_joule.tolist()
                # get_logger(save=True)
                st.write(" NO DRIFT FOUND")
                IF = pickle.load(open('model.pkl', 'rb'))

#     data_tab.subheader ( "The data frame representation " )
#     data_tab.dataframe( data )

#     face_1 = np.ones ( shape=(14, 16) ) * 0.
#     face_2 = np.ones ( shape=(14, 16) ) * 0.

#     face_1_maske = np.ones ( shape=(14, 16) )
#     face_2_maske = np.ones ( shape=(14, 16) )

#     face_1_repeat = np.zeros ( shape=(14, 16) )
#     face_2_repeat = np.zeros ( shape=(14, 16) )

#     face_1_repeat_mask = np.ones ( shape=(14, 16) )
#     face_2_repeat_mask = np.ones ( shape=(14, 16) )

#     colorscale = [[0.0, 'rgb(169,169,169)'],
#                   [0.5, 'rgb(0, 255, 0)'],
#                   [1.0, 'rgb(255, 0, 0)']]
#     time_plot_1_1 = 0
#     time_plot_1_2 = 0
#     time_plot_2_1 = 0
#     time_plot_2_2 = 0
#     plot_count_1 = 0
#     plot_count_1 = 0
#     plot_count_2 = 0
#     if options:
#         selected_bar = options[0]
#         time_count = next ( counter )
#         pack = data[data['BarCode']== selected_bar]
#         for col in ["Face", "Cell", "Point"]:
#             pack[col] = pack[col].str.slice ( start=0, stop=-2 )
#         for it, (index , cell )  in enumerate(pack.iterrows()):
#             location_info = cell[['Face', 'Cell', 'Point']].values.astype ( int )
#             print('in the localtin', location_info)
#             if not it:
#                 old_location = location_info
#                 if location_info[0] == 1:  # face1
#                     if location_info[-1] == 1:
#                         face_1[plot_count_1, time_plot_1_1] = location_info[1]
#                         face_1_maske[plot_count_1, time_plot_1_1] = False
#                         time_plot_1_1 += 1
#                     else:

#                         face_1[plot_count_1 + 1, time_plot_1_2] = location_info[1]
#                         face_1_maske[plot_count_1 + 1, time_plot_1_2] = False
#                         time_plot_1_2 += 1
#                     if time_plot_1_2 == 16:
#                         plot_count_1 += 2
#                         time_plot_1_1 = 0
#                         time_plot_1_2 = 0

#                 else:
#                     if location_info[-1] == 1:
#                         face_2[plot_count_2, time_plot_2_1] = location_info[1]
#                         face_2_maske[plot_count_2, time_plot_2_1] = False
#                         time_plot_2_1 += 1
#                     else:
#                         face_2[plot_count_2 + 1, time_plot_2_2] = location_info[1]
#                         face_2_maske[plot_count_2 + 1, time_plot_2_2] = False
#                         time_plot_2_2 += 1
#                     if time_plot_2_2 == 16:
#                         plot_count_2 += 2
#                         time_plot_2_2 = 0
#                         time_plot_2_1 = 0
#                 if plot_count_1 == 14:
#                     plot_count_1 = 0
#                     face_1 = np.ones ( shape=(14, 16) ) * 0.
#                     face_1_maske = np.ones ( shape=(14, 16) )
#                     face_1_repeat = np.zeros ( shape=(14, 16) )
#                     face_1_repeat_mask = np.ones ( shape=(14, 16) )
#                 if plot_count_2 == 14:
#                     plot_count_2 = 0

#                     face_2 = np.ones ( shape=(14, 16) ) * 0.
#                     face_2_maske = np.ones ( shape=(14, 16) )
#                     face_2_repeat = np.zeros ( shape=(14, 16) )
#                     face_2_repeat_mask = np.ones ( shape=(14, 16) )


#             elif (old_location != location_info[0]).any () and it > 0:

#                 old_location = location_info
#                 if location_info[0] == 1:  # face1
#                     if location_info[-1] == 1:
#                         print("shshshshs", face_1.shape, plot_count_1, time_plot_1_1 )
#                         face_1[plot_count_1, time_plot_1_1] = location_info[1]
#                         face_1_maske[plot_count_1, time_plot_1_1] = False
#                         time_plot_1_1 += 1
#                     else:

#                         face_1[plot_count_1 + 1, time_plot_1_2] = location_info[1]
#                         face_1_maske[plot_count_1 + 1, time_plot_1_2] = False

#                         time_plot_1_2 += 1
#                     if time_plot_1_2 == 15:
#                         plot_count_1 += 2
#                         time_plot_1_1 = 0
#                         time_plot_1_2 = 0

#                 else:
#                     if location_info[-1] == 1:
#                         face_2[plot_count_2, time_plot_2_1] = location_info[1]
#                         face_2_maske[plot_count_2, time_plot_2_1] = False
#                         time_plot_2_1 += 1
#                     else:
#                         face_2[plot_count_2 + 1, time_plot_2_2] = location_info[1]
#                         face_2_maske[plot_count_2 + 1, time_plot_2_2] = False
#                         time_plot_2_2 += 1
#                     if time_plot_2_2 == 15:
#                         plot_count_2 += 2
#                         time_plot_2_2 = 0
#                         time_plot_2_1 = 0
#                 if plot_count_1 == 14:
#                     plot_count_1 = 0
#                     face_1 = np.ones ( shape=(14, 16) ) * 0.
#                     face_1_maske = np.ones ( shape=(14, 16) )
#                     face_1_repeat = np.zeros ( shape=(14, 16) )
#                     face_1_repeat_mask = np.ones ( shape=(14, 16) )
#                 if plot_count_2 == 14:
#                     plot_count_2 = 0

#                     face_2 = np.ones ( shape=(14, 16) ) * 0.
#                     face_2_maske = np.ones ( shape=(14, 16) )
#                     face_2_repeat = np.zeros ( shape=(14, 16) )
#                     face_2_repeat_mask = np.ones ( shape=(14, 16) )

#             else:

#                 old_location = location_info
#                 if location_info[0] == 1:  # face1
#                     if location_info[-1] == 1:
#                         face_1_repeat[plot_count_1, time_plot_1_1] = location_info[1]
#                         face_1_repeat_mask[plot_count_1, time_plot_1_1] = False
#                     else:

#                         face_1_repeat[plot_count_1 + 1, time_plot_1_2] = location_info[1]
#                         face_1_repeat_mask[plot_count_1 + 1, time_plot_1_2] = False
#                 else:
#                     if location_info[-1] == 1:
#                         face_2_repeat[plot_count_2, time_plot_2_1] = location_info[1]
#                         face_2_repeat_mask[plot_count_2, time_plot_2_1] = False
#                     else:
#                         face_2_repeat[plot_count_2 + 1, time_plot_2_2] = location_info[1]
#                         face_2_repeat_mask[plot_count_2 + 1, time_plot_2_2] = False


#         face_1_cp_mask = face_1_maske.reshape ( 14, 16 )
#         face_2_cp_mask = face_2_maske.reshape ( 14, 16 )

#         fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
#         fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
#         print(face_1)
#         sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
#                       linewidths=0.8, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_maske, \
#                       yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
#                                    'cell_7', ''], annot=True, cbar_kws={
#                 'pad': .001,
#                 'ticks': [0, 1],
#                 "shrink": 0.01
#             },
#                       )
#         sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
#                       linewidths=0.8, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_maske, \
#                       yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
#                                    'cell_7', ''], annot=True, )
#         sns.heatmap ( face_1_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
#                       linewidths=0.8, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_repeat_mask, \
#                       yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
#                                    'cell_7', ''], annot=True, )
#         face_ax_1[1].set_title ( "Reapeted face 1" )
#         sns.heatmap ( face_2_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
#                       linewidths=0.8, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_repeat_mask, \
#                       yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
#                                    'cell_7', ''], annot=True, )
#         face_ax_2[1].set_title ( "Reapeted face 2" )
#         plot_tab, data_tab, model_tab
#         good_weld.pyplot ( fig_pack_1, use_container_width=True )
#         bad_weld.pyplot ( fig_pack_2, use_container_width=True )

# import streamlit as st


# init_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    init_options = data['Barcode'].unique().tolist()
    if 'options' not in st.session_state:
        st.session_state.options = init_options
    if 'default' not in st.session_state:
        st.session_state.default = []
    # print('initial option', st.session_state.options)

    ms = st.multiselect(
        label='Pick a Barcodef',
        options=st.session_state.options,
        default=st.session_state.default
    )
    DDDF = st.empty()
    Main = st.empty()
    # day_left, time_right = Main.columns(2)
    pack_view, table_view, chart_view = st.tabs(["Battery Pack", "🗃Table", "📈 Charts"])
        # Example controlers
    
    if ms:
        # print('we are in ms', ms)
        if ms in st.session_state.options:
            st.session_state.options.remove(ms[-1])
            st.session_state.default = ms[-1]
            st.experimental_rerun()
        pack_data = data[data['Barcode']== ms[-1]]

        ## TRAINING THE MODEL
        if SHIFT_DETECTED:
            if training_type == 'Pack':
                if model_ifor:
                    ifor = train_model(pack_data, model_type='ifor')
                    ifor_cluster = ifor.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    pack_data['ifor_anomaly']  =pack_data['ifor_anomaly'].astype(bool)

                if model_gmm :
                    gmm = train_model(pack_data, model_type='gmm')
                    gmm_cluster = gmm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['gmm_anomaly']  =  gmm_cluster 
                    pack_data['gmm_anomaly']  =  pack_data['gmm_anomaly'].astype(bool)

                if model_bgmm :
                    bgmm = train_model(pack_data, model_type='bgmm')
                    bgmm_cluster = bgmm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['bgmm_anomaly']  =  bgmm_cluster 
                    pack_data['bgmm_anomaly']  =  pack_data['bgmm_anomaly'].astype(bool)

                if model_lof:
                    lof = train_model(pack_data, model_type='lof')
                    lof_cluster = lof.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['lof_anomaly']  =  lof_cluster 
                    pack_data['lof_anomaly']  =  pack_data['lof_anomaly'].astype(bool)

                if model_svm:
                    svm = train_model(pack_data, model_type='svm')
                    svm_cluster = svm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['svm_anomaly']  =  svm_cluster 
                    pack_data['svm_anomaly']  =  pack_data['svm_anomaly'].astype(bool)

            else :
                if model_ifor:
                    ifor = train_model(data, model_type='ifor')
                    ifor_cluster = ifor.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    data['ifor_anomaly']  =data['ifor_anomaly'].astype(bool)

                if model_gmm :
                    gmm = train_model(data, model_type='gmm')
                    gmm_cluster = gmm.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['gmm_anomaly']  =  gmm_cluster 
                    data['gmm_anomaly']  =  data['gmm_anomaly'].astype(bool)

                if model_bgmm :
                    bgmm = train_model(data, model_type='bgmm')
                    bgmm_cluster = bgmm.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['bgmm_anomaly']  =  bgmm_cluster 
                    data['bgmm_anomaly']  =  data['bgmm_anomaly'].astype(bool)

                if model_lof:
                    lof = train_model(data, model_type='lof')
                    lof_cluster = lof.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['lof_anomaly']  =  lof_cluster 
                    data['lof_anomaly']  =  data['lof_anomaly'].astype(bool)

                if model_svm:
                    svm = train_model(data, model_type='svm')
                    svm_cluster = svm.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['svm_anomaly']  =  lof_cluster 
                    data['svm_anomaly']  =  data['svm_anomaly'].astype(bool)

        else:

            if training_type == 'Pack':
                if model_ifor:
                    ifor = train_model(pack_data, model_type='ifor')
                    ifor_cluster = ifor.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    pack_data['ifor_anomaly']  =pack_data['ifor_anomaly'].astype(bool)

                if model_gmm :
                    gmm = train_model(pack_data, model_type='gmm')
                    gmm_cluster = gmm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['gmm_anomaly']  =  gmm_cluster 
                    pack_data['gmm_anomaly']  =  pack_data['gmm_anomaly'].astype(bool)

                if model_bgmm :
                    bgmm = train_model(pack_data, model_type='bgmm')
                    bgmm_cluster = bgmm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['bgmm_anomaly']  =  bgmm_cluster 
                    pack_data['bgmm_anomaly']  =  pack_data['bgmm_anomaly'].astype(bool)

                if model_lof:
                    lof = train_model(pack_data, model_type='lof')
                    lof_cluster = lof.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['lof_anomaly']  =  lof_cluster 
                    pack_data['lof_anomaly']  =  pack_data['lof_anomaly'].astype(bool)

                if model_svm:
                    svm = train_model(pack_data, model_type='svm')
                    svm_cluster = svm.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['svm_anomaly']  =  svm_cluster 
                    pack_data['svm_anomaly']  =  pack_data['svm_anomaly'].astype(bool)


        with table_view:        
            gb = GridOptionsBuilder.from_dataframe(pack_data)

            cellsytle_jscode = JsCode("""
            function(params) {
                if (params.value == 0) {
                    
                    return {
                        'color': 'white',
                        'backgroundColor': 'darkred'
                    }
                } else {
                    return {
                        'color': 'black',
                        'backgroundColor': 'white'
                    }
                }
            };
            """)
            gb.configure_column("ifor_anomaly", cellStyle=cellsytle_jscode)

            if enable_sidebar:
                gb.configure_side_bar()

            if enable_selection:
                gb.configure_selection(selection_mode)
                if use_checkbox:
                    gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
                if ((selection_mode == 'multiple') & (not use_checkbox)):
                    gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)

            if enable_pagination:
                if paginationAutoSize:
                    gb.configure_pagination(paginationAutoPageSize=True)
                else:
                    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

            gb.configure_grid_options(domLayout='normal')
            gridOptions = gb.build()

            #Display the grid``
            print(f" mss {ms[-1]} -- {type(ms[-1])}")
            st.header(f"Table view : -- {ms[-1]}") 
            st.markdown("""
                This is the table view of the battery pack filtered using the Barcode
            """)

            grid_response = AgGrid(
                pack_data, 
                gridOptions=gridOptions,
                height=grid_height, 
                width='100%',
                data_return_mode=return_mode_value, 
                # update_mode=update_mode_value,
                update_mode=GridUpdateMode.MANUAL,
                fit_columns_on_grid_load=fit_columns_on_grid_load,
                allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                enable_enterprise_modules=enable_enterprise_modules
                )
            # grid_table = AgGrid(face_1_df_1 , height=500,  gridOptions=gridoptions,
            #                                 update_mode=GridUpdateMode.SELECTION_CHANGED, allow_unsafe_jscode=True)
            #             selected_row = grid_table["selected_rows"]
            #             st.table(selected_row)
            # df = grid_response['data']
            # selected = grid_response['selected_rows']

        with chart_view :
            if model_ifor:
                with st.expander("ISOLATION FOREST"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['ifor_anomaly'] = pack_data['ifor_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='ifor_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='ifor_anomaly' , names='ifor_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='ifor_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='ifor_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='ifor_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='ifor_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )

            if model_gmm:
                with st.expander("GAUSSIAN MIXTURE"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['gmm_anomaly'] = pack_data['gmm_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='gmm_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='gmm_anomaly' , names='gmm_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='gmm_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='gmm_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='gmm_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='gmm_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )


            if model_bgmm:
                with st.expander("BAYESIAN GAUSSIAN MIXTURE"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['bgmm_anomaly'] = pack_data['bgmm_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='bgmm_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='bgmm_anomaly' , names='bgmm_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='bgmm_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='bgmm_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='bgmm_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='bgmm_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )


            if model_lof:
                with st.expander("LOCAL OUTLIER FACTOR"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['lof_anomaly'] = pack_data['lof_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='lof_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='lof_anomaly' , names='lof_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='lof_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='lof_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='lof_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='lof_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )

            if model_svm:
                with st.expander("One-Class SVM"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['svm_anomaly'] = pack_data['svm_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='svm_anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='svm_anomaly' , names='svm_anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='svm_anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='svm_anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='svm_anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='svm_anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )
            

            if model_repeat:
                with st.expander("REPEAT FROM MACHINE"):
                    plot_st, pi_st = st.columns((3,1))
                    if model_ifor:
                        pack_data['anomaly'] = pack_data['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                        fig = px.scatter ( pack_data, y='Joules', color='anomaly', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        

                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='anomaly' , names='anomaly', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'}) 
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='anomaly', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:
                        
                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='anomaly', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='anomaly', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_c, use_container_width=True )
                    

                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='anomaly', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' ) 
                        plot_st.plotly_chart ( fig_r, use_container_width=True )
            

            # fig.append_trace ( go.Scatter ( y=pack_data['Charge'], x=pack_data.index, mode='lines', name='Force_N',
            #                                 marker = { 'colorscale': colorscale, 'size': 10 } ), row=1, col=1, )
            # fig.update_layout (plot_bgcolor='rgb(131, 193, 212)' ) 
            # st.plotly_chart ( fig, use_container_width=True )
            # if model_gmm :
            #     gmm = train_model(data, model_type='gmm')
            #     gmm_cluster = gmm.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
            #     data['gmm_anomaly']  =  gmm_cluster 
            #     data['gmm_anomaly']  =  data['gmm_anomaly'].astype(bool)

            # if model_bgmm :
            #     bgmm = train_model(data, model_type='bgmm')
            #     bgmm_cluster = bgmm.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
            #     data['bgmm_anomaly']  =  bgmm_cluster 
            #     data['bgmm_anomaly']  =  data['bgmm_anomaly'].astype(bool)

            # if model_lof:
            #     lof = train_model(data, model_type='lof')
            #     lof_cluster = lof.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
            #     data['anomaly']  =  lof_cluster 
                #     data['lof_anomaly']  =  data['lof_anomaly'].astype(bool)
        with pack_view :
                 
       

            pack_data_non_dup = pack_data[~pack_data.duplicated(subset=['Barcode', 'Face', 'Cell', 'Point'], keep= 'last')]
            pack_data_dup = pack_data[pack_data.duplicated(subset=['Barcode',  'Face', 'Cell', 'Point'], keep= 'last')]

            face_1 = np.ones ( shape=(14, 16) ) * 0.
            face_2 = np.ones ( shape=(14, 16) ) * 0.

            face_1_maske = np.ones ( shape=(14, 16) )
            face_2_maske = np.ones ( shape=(14, 16) )

            face_1_repeat = np.zeros ( shape=(14, 16) )
            face_2_repeat = np.zeros ( shape=(14, 16) )

            face_1_repeat_mask = np.ones ( shape=(14, 16) )
            face_2_repeat_mask = np.ones ( shape=(14, 16) )

            colorscale = [[0.0, 'rgb(169,169,169)'],
                        [0.5, 'rgb(0, 255, 0)'],
                        [1.0, 'rgb(255, 0, 0)']]
            time_plot_1_1 = 0
            time_plot_1_2 = 0
            time_plot_2_1 = 0
            time_plot_2_2 = 0
            plot_count_1 = 0
            plot_count_1 = 0
            plot_count_2 = 0
            # pack = data[data['Barcode']== ms[0]]

            if model_ifor:
                with st.expander("ISOLATION FOREST"):
                    pack_face1, pack_face2 = st.columns(2)

                    pack_data_non_dup['ifor_anomaly'] = pack_data_non_dup['ifor_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df = pack_data_non_dup[pack_data_non_dup['Face']==2]

                    face_1_df_1_val = face_1_df_1['ifor_anomaly'].values
                    face_1_df_1_val = face_1_df_1_val.reshape(-1, 16)

                    face_1_df_2_val = face_1_df_2['ifor_anomaly'].values
                    face_1_df_2_val = face_1_df_2_val.reshape(-1, 16)

                    face_2_df_1_val = face_2_df_1['ifor_anomaly'].values
                    face_2_df_1_val = face_2_df_1_val.reshape(-1, 16)

                    face_2_df_2_val = face_2_df_2['ifor_anomaly'].values
                    face_2_df_2_val = face_2_df_2_val.reshape(-1, 16)


                    fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    face_1[0::2,:] = face_1_df_1_val
                    face_1_maske[0::2,:] = False
                    face_1[1::2,:] = face_1_df_2_val
                    face_1_maske[1::2,:] = False

                    face_2[0::2,:] = face_2_df_1_val
                    face_2_maske[0::2,:] = False
                    face_2[1::2,:] = face_2_df_2_val
                    face_2_maske[1::2,:] = False


                    sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[0].set_title ( "Face 1" )
                        # cbar_kws={
                        #     'pad': .001,
                        #     'ticks': [0, 1],
                        #     "shrink": 0.01
                        # },
                                
                    sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[0].set_title ( "Face 2" )
                    sns.heatmap ( face_1_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[1].set_title ( "Reapeted face 1" )
                    sns.heatmap ( face_2_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[1].set_title ( "Reapeted face 2" )
                    pack_face1.pyplot ( fig_pack_1, use_container_width=True )
                    pack_face2.pyplot ( fig_pack_2, use_container_width=True )
                    st.table(pack_data_dup)

            if model_gmm:
                with st.expander("GAUSSIAN MIXTURE"):
                    pack_face1, pack_face2 = st.columns(2)
                    pack_data_non_dup['gmm_anomaly'] = pack_data_non_dup['gmm_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]# & (pack_data_non_dup['anomaly']==False)]
                    face_2_df = pack_data_non_dup[pack_data_non_dup['Face']==2]

                    face_1_df_1_val = face_1_df_1['gmm_anomaly'].values
                    face_1_df_1_val = face_1_df_1_val.reshape(-1, 16)

                    face_1_df_2_val = face_1_df_2['gmm_anomaly'].values
                    face_1_df_2_val = face_1_df_2_val.reshape(-1, 16)

                    face_2_df_1_val = face_2_df_1['gmm_anomaly'].values
                    face_2_df_1_val = face_2_df_1_val.reshape(-1, 16)

                    face_2_df_2_val = face_2_df_2['gmm_anomaly'].values
                    face_2_df_2_val = face_2_df_2_val.reshape(-1, 16)


                    fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    face_1[0::2,:] = face_1_df_1_val
                    face_1_maske[0::2,:] = False
                    face_1[1::2,:] = face_1_df_2_val
                    face_1_maske[1::2,:] = False

                    face_2[0::2,:] = face_2_df_1_val
                    face_2_maske[0::2,:] = False
                    face_2[1::2,:] = face_2_df_2_val
                    face_2_maske[1::2,:] = False


                    sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[0].set_title ( "Face 1" )
                        # cbar_kws={
                        #     'pad': .001,
                        #     'ticks': [0, 1],
                        #     "shrink": 0.01
                        # },
                                
                    sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_maske, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[0].set_title ( "Face 2" )
                    sns.heatmap ( face_1_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_1[1].set_title ( "Reapeted face 1" )
                    sns.heatmap ( face_2_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.8, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_repeat_mask, \
                                yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
                                            'cell_7', ''], annot=True, )
                    face_ax_2[1].set_title ( "Reapeted face 2" )
                    pack_face1.pyplot ( fig_pack_1, use_container_width=True )
                    pack_face2.pyplot ( fig_pack_2, use_container_width=True )


    #         face_1 =  np.ones(shape=(14,16))*0.
    #         face_2 =  np.ones(shape=(14,16))*0.

    #         face_1_maske = np.ones ( shape=(14,16) )
    #         face_2_maske = np.ones ( shape=(14,16) )

    #         face_1_repeat = np.zeros(shape=(14,16))
    #         face_2_repeat = np.zeros(shape=(14,16))

    #         face_1_repeat_mask =  np.ones(shape=(14,16))
    #         face_2_repeat_mask = np.ones(shape = (14,16))

    #         colorscale=[[0.0, 'rgb(169,169,169)'],
    #                 [0.5, 'rgb(0, 255, 0)'],
    #                 [1.0, 'rgb(255, 0, 0)']]
    #         time_plot_1_1 = 0
    #         time_plot_1_2 = 0
    #         time_plot_2_1 = 0
    #         time_plot_2_2 = 0
    #         plot_count_1 = 0
    #         plot_count_1 = 0
    #         plot_count_2 = 0
    #         # set_trace()
    #         pack_data['ifor_anomaly'] = pack_data['ifor_anomaly'].apply ( lambda x: False if x == 'Normal' else  True )

    #         pack_data_non_dup = pack_data[~pack_data.duplicated(subset=['Barcode', 'Face', 'Cell', 'Point'], keep= 'last')]
    #         pack_data_dup = pack_data[pack_data.duplicated(subset=['Barcode',  'Face', 'Cell', 'Point'], keep= 'last')]
    #         for time_count, (index , cell )  in enumerate(pack_data_non_dup.iterrows()):
    #         # if not time_count:
    #             if cell.Face == 1:  # face1
    #                 if cell.Point == 1:
    #                     face_1[plot_count_1, time_plot_1_1] = cell.ifor_anomaly
    #                     face_1_maske[plot_count_1, time_plot_1_1] = False
    #                     time_plot_1_1 += 1
    #                 else:

    #                     face_1[plot_count_1 + 1, time_plot_1_2] = cell.ifor_anomaly
    #                     face_1_maske[plot_count_1 + 1, time_plot_1_2] = False
    #                     time_plot_1_2 += 1
    #                 if time_plot_1_2 == 16:
    #                     plot_count_1 += 2
    #                     time_plot_1_1 = 0
    #                     time_plot_1_2 = 0

    #             else:
    #                 if cell.Point == 1:
    #                     face_2[plot_count_2, time_plot_2_1] = cell.ifor_anomaly
    #                     face_2_maske[plot_count_2, time_plot_2_1] = False
    #                     time_plot_2_1 += 1
    #                 else:
    #                     face_2[plot_count_2 + 1, time_plot_2_2] = cell.ifor_anomaly
    #                     face_2_maske[plot_count_2 + 1, time_plot_2_2] = False
    #                     time_plot_2_2 += 1
    #                 if time_plot_2_2 == 16:
    #                     plot_count_2 += 2
    #                     time_plot_2_2 = 0
    #                     time_plot_2_1 = 0
    #             if plot_count_1 == 14:
    #                 plot_count_1 = 0
    #                 face_1 = np.ones ( shape=(14, 16) ) * 0.
    #                 face_1_maske = np.ones ( shape=(14, 16) )
    #                 face_1_repeat = np.zeros ( shape=(14, 16) )
    #                 face_1_repeat_mask = np.ones ( shape=(14, 16) )
    #             if plot_count_2 == 14:
    #                 plot_count_2 = 0

    #                 face_2 = np.ones ( shape=(14, 16) ) * 0.
    #                 face_2_maske = np.ones ( shape=(14, 16) )
    #                 face_2_repeat = np.zeros ( shape=(14, 16) )
    #                 face_2_repeat_mask = np.ones ( shape=(14, 16) )

        
    #         fig_pack_1, face_ax_1 = plt.subplots (nrows=2, ncols=1, figsize=(5, 5) )
    #         fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1,  figsize=(5, 5) )

    #         sns.heatmap(face_1, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_1[0], cbar=False, mask=face_1_maske,\
    #                     yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],cbar_kws={
    #                 'pad': .001,
    #                 'ticks': [0, 1],
    #                 "shrink": 0.01
    #             },
    # )
    #         sns.heatmap(face_2, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_2[0], cbar=False, mask=face_2_maske,\
    #                     yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
    #         # sns.heatmap(face_1_repeat, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_1[1], cbar=False, mask=face_1_repeat_mask,\
    #         #             yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
    #         # face_ax_1[1].set_title("Reapeted face 1")
    #         # sns.heatmap(face_2_repeat, cmap=ListedColormap(['green', 'red']),  vmin=0, vmax=1,linecolor='lightgray', linewidths=0.8,square=True, ax = face_ax_2[1], cbar=False, mask=face_2_repeat_mask,\
    #         #             yticklabels=['cell_1','', 'cell_2', '','cell_3', '', 'cell_4', '','cell_5', '','cell_6', '','cell_7', ''],)
    #         # face_ax_2[1].set_title ( "Reapeted face 2" )

    #         st.pyplot(fig_pack_1,use_container_width=True)
    #         st.pyplot(fig_pack_2,use_container_width=True)





            # elif (old_location != location_info[0]).any() and time_count > 0:

            #     old_location = location_info
            #     if location_info[0][0] == 1:  # face1
            #         if location_info[0][-1] == 1:
            #             face_1[plot_count_1, time_plot_1_1] = cell.ior_anomaly
            #             face_1_maske[plot_count_1, time_plot_1_1] = False
            #             time_plot_1_1 += 1
            #         else:

            #             face_1[plot_count_1 + 1, time_plot_1_2] = cell.ior_anomaly
            #             face_1_maske[plot_count_1 + 1, time_plot_1_2] = False
            #             time_plot_1_2 += 1
            #         if time_plot_1_2 == 16:
            #             plot_count_1 += 2
            #             time_plot_1_1 = 0
            #             time_plot_1_2 = 0

            #     else:
            #         if location_info[0][-1] == 1:
            #             face_2[plot_count_2, time_plot_2_1] = cell.ior_anomaly
            #             face_2_maske[plot_count_2, time_plot_2_1] = False
            #             time_plot_2_1 += 1
            #         else:
            #             face_2[plot_count_2 + 1, time_plot_2_2] = cell.ior_anomaly
            #             face_2_maske[plot_count_2 + 1, time_plot_2_2] = False
            #             time_plot_2_2 += 1
            #         if time_plot_2_2 == 16:
            #             plot_count_2 += 2
            #             time_plot_2_2 = 0
            #             time_plot_2_1 = 0
            #     if plot_count_1 == 14:
            #         plot_count_1 = 0
            #         face_1 = np.ones ( shape=(14, 16) ) * 0.
            #         face_1_maske = np.ones ( shape=(14, 16) )
            #         face_1_repeat = np.zeros ( shape=(14, 16) )
            #         face_1_repeat_mask = np.ones ( shape=(14, 16) )
            #     if plot_count_2 == 14:
            #         plot_count_2 = 0

            #         face_2 = np.ones ( shape=(14, 16) ) * 0.
            #         face_2_maske = np.ones ( shape=(14, 16) )
            #         face_2_repeat = np.zeros ( shape=(14, 16) )
            #         face_2_repeat_mask = np.ones ( shape=(14, 16) )



            # if not pack_data_dup.empty:
            #     if location_info[0][0] == 1:  # face1
            #         if location_info[0][-1] == 1:
            #             face_1_repeat[plot_count_1, time_plot_1_1] = pred
            #             face_1_repeat_mask[plot_count_1, time_plot_1_1] = False
            #         else:

            #             face_1_repeat[plot_count_1 + 1, time_plot_1_2] = pred
            #             face_1_repeat_mask[plot_count_1 + 1, time_plot_1_2] = False
            #     else:
            #         if location_info[0][-1] == 1:
            #             face_2_repeat[plot_count_2, time_plot_2_1] = pred
            #             face_2_repeat_mask[plot_count_2, time_plot_2_1] = False
            #         else:
            #             face_2_repeat[plot_count_2 + 1, time_plot_2_2] = pred
            #             face_2_repeat_mask[plot_count_2 + 1, time_plot_2_2] = False


        # day_left, time_right = pack_view.columns(2)
        # good_weld = day_left.empty()
        # bad_weld = time_right.empty()
    
        # # print(F' Other option {data.columns}' )

        # face_1 = np.ones ( shape=(14, 16) ) * 0.
        # face_2 = np.ones ( shape=(14, 16) ) * 0.

        # face_1_maske = np.ones ( shape=(14, 16) )
        # face_2_maske = np.ones ( shape=(14, 16) )

        # face_1_repeat = np.zeros ( shape=(14, 16) )
        # face_2_repeat = np.zeros ( shape=(14, 16) )

        # face_1_repeat_mask = np.ones ( shape=(14, 16) )
        # face_2_repeat_mask = np.ones ( shape=(14, 16) )

        # colorscale = [[0.0, 'rgb(169,169,169)'],
        #             [0.5, 'rgb(0, 255, 0)'],
        #             [1.0, 'rgb(255, 0, 0)']]
        # time_plot_1_1 = 0
        # time_plot_1_2 = 0
        # time_plot_2_1 = 0
        # time_plot_2_2 = 0
        # plot_count_1 = 0
        # plot_count_1 = 0
        # plot_count_2 = 0
        # pack = data[data['Barcode']== ms[0]]

        # face_1_df_1 = pack[(pack['Face']==1) & (pack['Point']==1)]# & (pack['anomaly']==False)]
        # face_1_df_2 = pack[(pack['Face']==1) & (pack['Point']==2)]# & (pack['anomaly']==False)]
        # face_2_df_1 = pack[(pack['Face']==2) & (pack['Point']==1)]# & (pack['anomaly']==False)]
        # face_2_df_2 = pack[(pack['Face']==2) & (pack['Point']==2)]# & (pack['anomaly']==False)]
        # face_2_df = pack[pack['Face']==2]

        # face_1_df_1_val = face_1_df_1['Cell'].values
        # face_1_df_1_val = face_1_df_1_val.reshape(-1, 16)

        # face_1_df_2_val = face_1_df_2['Cell'].values
        # face_1_df_2_val = face_1_df_2_val.reshape(-1, 16)

        # face_2_df_1_val = face_2_df_1['Cell'].values
        # face_2_df_1_val = face_2_df_1_val.reshape(-1, 16)

        # face_2_df_2_val = face_2_df_2['Cell'].values
        # face_2_df_2_val = face_2_df_2_val.reshape(-1, 16)

        # with st.expander("Show Based on Repeat"):
    
        #     gd = GridOptionsBuilder.from_dataframe(face_1_df_1.drop(['Face', 'Cell', 'Point'], axis=1))
        #     gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        #     gridoptions = gd.build()
        #     # gridoptions['getRowStyle'] = jscode

        #     cellsytle_jscode = JsCode("""
        #     function(params) {
        #         if (params.value == False) {
        #             return {
        #                 'color': 'white',
        #                 'backgroundColor': 'darkred'
        #             }
        #         } else {
        #             return {
        #                 'color': 'black',
        #                 'backgroundColor': 'white'
        #             }
        #         }
        #     };
        #     """)
        #     gd.configure_column("anomaly", cellStyle=cellsytle_jscode)

        #     grid_table = AgGrid(face_1_df_1 , height=500,  gridOptions=gridoptions,
        #                         update_mode=GridUpdateMode.SELECTION_CHANGED, allow_unsafe_jscode=True)
        #     selected_row = grid_table["selected_rows"]
        #     st.table(selected_row)
        #     st.write('## Selected')
            
           

        # test_df_1.dataframe(face_1_df_1) 
        # test_df_1.dataframe(face_1_df_2) 
        # test_df_2.dataframe(face_2_df_1)
        # test_df_2.dataframe(face_2_df_2)
        # fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
        # fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
        # face_1[0::2,:] = face_1_df_1_val
        # face_1_maske[0::2,:] = False
        # face_1[1::2,:] = face_1_df_2_val
        # face_1_maske[1::2,:] = False


        
        # for it, (index , cell )  in enumerate(face_1_df.iterrows()):
        #     location_info = cell[['Face', 'Cell', 'Point']] .values
        #     time.sleep(0.5)
        #     print(F'in the localtin {location_info} -- {time_plot_1_1}')
        #     print('raw number', max(location_info[1]%17 -1,0))
      

        #     # plot_count_1, time_plot_1_1 = location_info[1]
        #     # print('location', location_info[1]//16 + plot_count_1)
        #     face_1[location_info[1]//16  + location_info[-1]-1+plot_count_1, max(location_info[1]%17 -1,0)] = location_info[1]
        #     face_1_maske[location_info[1]//16 + location_info[-1]-1+plot_count_1, max(location_info[1]%17 -1,0)] = False
        #     time_plot_1_1 += 1
        #     # del fig_pack_1, face_ax_1
           
            
        #     # print(face_1)
        #     sns.heatmap( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
        #                 linewidths=0.8, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_maske, \
        #                 yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
        #                             'cell_7', ''], annot=True, cbar_kws={
        #             'pad': .001,
        #             'ticks': [0, 1],
        #             "shrink": 0.01
        #         },)
        #     good_weld.pyplot ( fig_pack_1, use_container_width=True )

        #     if  time_plot_1_1 == 34:
        #         plot_count_1 += 1
        #         time_plot_1_1 = 0
        #         time_plot_1_2 = 0
        #         time_plot_1_1 += 1
            # else:
            #     if location_info[-1] == 1:
            #         face_2[plot_count_2, time_plot_2_1] = location_info[1]
            #         face_2_maske[plot_count_2, time_plot_2_1] = False
            #         time_plot_2_1 += 1
            #     else:
            #         face_2[plot_count_2 + 1, time_plot_2_2] = location_info[1]
            #         face_2_maske[plot_count_2 + 1, time_plot_2_2] = False
            #         time_plot_2_2 += 1
            #     if time_plot_2_2 == 16:
            #         plot_count_2 += 2
            #         time_plot_2_2 = 0
            #         time_plot_2_1 = 0
            
            # if plot_count_1+1 == 13:
            #     plot_count_1 = 0
            #     face_1 = np.ones ( shape=(14, 16) ) * 0.
            #     face_1_maske = np.ones ( shape=(14, 16) )
            #     face_1_repeat = np.zeros ( shape=(14, 16) )
            #     face_1_repeat_mask = np.ones ( shape=(14, 16) )
            # if plot_count_2 == 14:
            #     plot_count_2 = 0

            #     face_2 = np.ones ( shape=(14, 16) ) * 0.
            #     face_2_maske = np.ones ( shape=(14, 16) )
            #     face_2_repeat = np.zeros ( shape=(14, 16) )
            #     face_2_repeat_mask = np.ones ( shape=(14, 16) )


            # elif (old_location != location_info[0]).any () and it > 0:

            #     old_location = location_info
            #     if location_info[0] == 1:  # face1
            #         if location_info[-1] == 1:
            #             print("shshshshs", face_1.shape, plot_count_1, time_plot_1_1 )
            #             face_1[plot_count_1, time_plot_1_1] = location_info[1]
            #             face_1_maske[plot_count_1, time_plot_1_1] = False
            #             time_plot_1_1 += 1
            #         else:

            #             face_1[plot_count_1 + 1, time_plot_1_2] = location_info[1]
            #             face_1_maske[plot_count_1 + 1, time_plot_1_2] = False

            #             time_plot_1_2 += 1
            #         if time_plot_1_2 == 15:
            #             plot_count_1 += 2
            #             time_plot_1_1 = 0
            #             time_plot_1_2 = 0

            #     else:
            #         if location_info[-1] == 1:
            #             face_2[plot_count_2, time_plot_2_1] = location_info[1]
            #             face_2_maske[plot_count_2, time_plot_2_1] = False
            #             time_plot_2_1 += 1
            #         else:
            #             face_2[plot_count_2 + 1, time_plot_2_2] = location_info[1]
            #             face_2_maske[plot_count_2 + 1, time_plot_2_2] = False
            #             time_plot_2_2 += 1
            #         if time_plot_2_2 == 15:
            #             plot_count_2 += 2
            #             time_plot_2_2 = 0
            #             time_plot_2_1 = 0
            #     if plot_count_1 == 14:
            #         plot_count_1 = 0
            #         face_1 = np.ones ( shape=(14, 16) ) * 0.
            #         face_1_maske = np.ones ( shape=(14, 16) )
            #         face_1_repeat = np.zeros ( shape=(14, 16) )
            #         face_1_repeat_mask = np.ones ( shape=(14, 16) )
            #     if plot_count_2 == 14:
            #         plot_count_2 = 0

            #         face_2 = np.ones ( shape=(14, 16) ) * 0.
            #         face_2_maske = np.ones ( shape=(14, 16) )
            #         face_2_repeat = np.zeros ( shape=(14, 16) )
            #         face_2_repeat_mask = np.ones ( shape=(14, 16) )

            # else:

            #     old_location = location_info
            #     if location_info[0] == 1:  # face1
            #         if location_info[-1] == 1:
            #             face_1_repeat[plot_count_1, time_plot_1_1] = location_info[1]
            #             face_1_repeat_mask[plot_count_1, time_plot_1_1] = False
            #         else:

            #             face_1_repeat[plot_count_1 + 1, time_plot_1_2] = location_info[1]
            #             face_1_repeat_mask[plot_count_1 + 1, time_plot_1_2] = False
            #     else:
            #         if location_info[-1] == 1:
            #             face_2_repeat[plot_count_2, time_plot_2_1] = location_info[1]
            #             face_2_repeat_mask[plot_count_2, time_plot_2_1] = False
            #         else:
            #             face_2_repeat[plot_count_2 + 1, time_plot_2_2] = location_info[1]
            #             face_2_repeat_mask[plot_count_2 + 1, time_plot_2_2] = False

        # fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
        # fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
        # # print(face_1)
        # sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
        #             linewidths=0.8, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_maske, \
        #             yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
        #                         'cell_7', ''], annot=True, cbar_kws={
        #         'pad': .001,
        #         'ticks': [0, 1],
        #         "shrink": 0.01
        #     },
        #             )
        # sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
        #             linewidths=0.8, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_maske, \
        #             yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
        #                         'cell_7', ''], annot=True, )
        # sns.heatmap ( face_1_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
        #             linewidths=0.8, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_repeat_mask, \
        #             yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
        #                         'cell_7', ''], annot=True, )
        # face_ax_1[1].set_title ( "Reapeted face 1" )
        # sns.heatmap ( face_2_repeat, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
        #             linewidths=0.8, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_repeat_mask, \
        #             yticklabels=['cell_1', '', 'cell_2', '', 'cell_3', '', 'cell_4', '', 'cell_5', '', 'cell_6', '',
        #                         'cell_7', ''], annot=True, )
        # face_ax_2[1].set_title ( "Reapeted face 2" )
        # good_weld.pyplot ( fig_pack_1, use_container_width=True )
        # bad_weld.pyplot ( fig_pack_2, use_container_width=True )


    # st.write('##### Valid Selection')
    # st.write(str(ms))