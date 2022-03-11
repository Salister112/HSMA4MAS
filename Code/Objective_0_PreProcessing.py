#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:39:09 2022

@author: mattheweves
"""

"""
Pre-processing file for the project. 
Reads in a service data set (csv) and ONS population dataset
Returns x2 pre-processed datasets.
One for unsupervised HEA Objective 1a
One for supervised DNA Objective 1b

#git location: https://github.com/Salister112/HSMA4MAS

CURRENT KNOWN BUG TO FIX: LINE 1180
CHECK IF NOTEBOOK WORKS / FALLS OVER AT SAME PLACE
CODE IS THE SAME...

"""

#Load standard modules numpy and pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import missingno as msno

#library 
#import textwrap as twp

#Import Machine Learning modules
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler


#Import spreadsheet manipulation modules
#import openpyxl
#from openpyxl.styles import Alignment, Font, Color, colors, PatternFill, Border

#Import file and directory navigation modules
import os
#import glob


#Branding Colours
#Taken from https://www.england.nhs.uk/nhsidentity/identity-guidelines/colours/

#Level 1 - Major Colours (prominent use, with plenty of nhs_blue specifically)
nhs_dark_blue = "#003087"
nhs_blue = "#005EB8"
nhs_bright_blue = "#0072CE"
nhs_light_blue = "#41B6E6"
nhs_aqua_blue = "#00A9CE"

#Level 2 - Neutrals
nhs_mid_grey = "#768692" #pale neutral, plentiful use
nhs_pale_grey = "#E8EDEE" #pale neutral, plentiful use
nhs_black = "#231f20" #moderate use
nhs_dark_grey = "#425563" #moderate use
nhs_white = "#FFFFFF" #moderate use

#Level 3 - NHS Support Greens
nhs_dark_green = "#006747"
nhs_green = "#009639"
nhs_light_green = "#78BE20"
nhs_aqua_green = "#00A499"

#Level 4 - NHS highlights (minimal use)
nhs_purple = "#330072"
nhs_dark_pink = "#7C2855"
nhs_pink = "#AE2573"
nhs_dark_red = "#8A1538"
nhs_emergency_services_red = "#DA291C"
nhs_orange = "#ED8B00"
nhs_warm_yellow = "#FFB81C"
nhs_yellow = "#FAE100"


"""
#Locate current working directory
#print the current working directory (which may differ to the file location)
cwd = os.getcwd()
print(cwd)

#Tell the program where it is itself located in the directory and retrieve that file path
fileLocation = os.path.abspath(os.path.dirname(__file__))

#Change the current working directory to the program's own directory
#This means the program can locate files (csv / xlsx etc.) within that location
#Or subfolders within that directory

os.chdir(fileLocation)
print("New working directory:")
cwd = os.getcwd()
print(cwd)
"""

#-----------------------------------------------------------
#Define Functions

#Identify any duplicate rows and allow user to decide whether to remove
def identify_duplicate_rows(df):
    #Could use stringtobool in this function - this recognises commonly used yes/no responses, converts to 1 or 0
    """
    >> from distutils.util import strtobool
    >> strtobool('yes')
    """
    
    original_df_count = df.count()
    original_df_num_rows = int(original_df_count[0])
    print(f'\nOriginal number of rows: {original_df_num_rows}')

    df_dedup = df.drop_duplicates()
    df_dedup_count = df_dedup.count()
    df_dedup_num_rows = int(df_dedup_count[0])
    print(f'\nNumber of unique rows if duplicates removed: {df_dedup_num_rows}')

    num_duplicates = original_df_num_rows - df_dedup_num_rows
    #print(f'Number of duplicates removed: {num_duplicates}')

    df_to_use = pd.DataFrame()

    #Remove Duplicates - User Decision if duplicates detected
    if num_duplicates != 0:
        print(f'\nWARNING: This data set has {num_duplicates} duplicate rows:')
        
        df_dup_rows = df.duplicated()
        print(df[df_dup_rows])

        print('\nWould you like to remove these duplicate rows?')
        while True:
            try:
                remove_dups = int((input('\nPress 1 to remove or 2 to retain >>>> ')))
                if remove_dups == 1:
                    df_to_use = df_to_use.append(df_dedup)
                    break
                elif remove_dups == 2:
                    df_to_use = df_to_use.append(df)
                    break
                else:
                    print('That is not a valid selection.')
            except:
                print('That is not a valid selection.')
    else:
        print('This data set has no duplicate rows.')
    
    return df_to_use

#-----------------------------------------------------

#Allow user to define what column heading / data field name is a unique identifier for patients
#such as customer ID, patient ID etc. 
#Assign to variable name of "patient_id", which is then used throughout the rest of the program
#Attempts to control for typos / name variations etc. in data sets for different use cases

def user_field_selection(field_list, field_name):
    dict_of_fields = {}
    key_counter = 1
    for value in field_list:
        dict_of_fields[key_counter] = value
        key_counter += 1

    print("\nThe data fields present in the data set are listed below")
    for key_number in dict_of_fields:
        print(str(key_number) + " : " + str(dict_of_fields[key_number]))

    print(f"\nPlease enter the number that represents the {field_name} field in the above list:\n")

    while True:
        try:
            patient_id_num = int(input('Make a selection: '))
            break

        except ValueError:
            print('Invalid Input. Please enter a number.')

    print(f"\nGreat. The entered number was: {patient_id_num}")
    print(f"\nThe {field_name} field that the program will use is: {dict_of_fields[patient_id_num]}")

    return dict_of_fields[patient_id_num]

#-----------------------------------------------------
def data_quality_summary(df, field_list):
    present_values = []
    missing_values = []
    percent_missing = []

    list_of_lists = [present_values, missing_values, percent_missing]
    list_names = ["Present", "Missing", "% Missing"]

    for col_name in field_list:
        col_missing = df[col_name].isna().sum()
        missing_values.append(col_missing)

        col_present = df[col_name].count()
        present_values.append(col_present)

        try:
            percent = (col_missing / (col_missing + col_present)) * 100
            percent_missing.append(round(percent,1))
        except:
            percent_missing.append(round(0.0,1))

    data_quality_summary_df = pd.DataFrame(list_of_lists, columns = field_list, index = list_names)

    #create new lists, representing counts of present / missing data by 
    #variable, and percent missing
    #To enable summary table beneath chart (subsequent cell)

    present = data_quality_summary_df.loc["Present"].values.tolist()
    missing = data_quality_summary_df.loc["Missing"].values.tolist()
    missing_pct = data_quality_summary_df.loc["% Missing"].values.tolist()
    
    return list_names, present, missing, missing_pct, data_quality_summary_df

#-----------------------------------------------------
#consider adding second axis plot for % missing
#This could improve interpretation (where missing count small, its invisible on the chart...)

def present_missing_stacked_bar(fields_list, present, missing, present_color, missing_color, title):
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(fields_list, present, width, label="Present Values", color=present_color)
    ax.bar(data_fields, missing, width, bottom=present, label="Missing Values", color=missing_color)

    ax.set_ylabel("Value count")
    ax.set_title(f"Missing Values Summary: {title}")
    ax.tick_params(axis='x', labelrotation = 90)
    ax.legend(loc="lower right")

    plt.show()
#-----------------------------------------------------

#https://www.pythonpool.com/matplotlib-table/
#data_quality_summary_df.loc[data_quality_summary_df['column_name'] == some_value]

def present_missing_stacked_bar_with_table(
    list_names, 
    fields_list, 
    present, 
    missing, 
    missing_pct,
    present_color,
    missing_color,
    pct_color,
    title):

    plt.rcParams['figure.figsize'] = [10, 5]
    data = [present, missing, missing_pct]

    columns = fields_list
    rows = list_names #present, missing, % missing 

    colors = [present_color, missing_color, pct_color]

    n_rows = len(data) #3

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []

    for row in range(n_rows):
        #y_offset_edit = y_offset + data[row]
        cell_text.append([f'{x:.1f}' for x in data[row]])
        if "%" in rows[row]:
            continue
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Count of rows in the data set used")

    #plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title(f"Missing Data Summary: {title}.")

    plt.show()
#-----------------------------------------------------
#Function to visually check if missing at random or not
#Definte function to visualise matrix of missing values
#Missing data appears as white space in the chart
#Development idea: set fig size for the this (?and possibly, all) chart(s) to ensure consistency. Esp. if creating a report at the end?

def dq_matrix(df):
    matrix = msno.matrix(df_with_imd_unique_pts)
    return matrix

#-----------------------------------------------------
#The missingno correlation heatmap measures nullity correlation: 
#how strongly the presence or absence of one variable affects the presence of another: 
#(source: https://github.com/ResidentMario/missingno)

def heatmap(df):
    print("WARNING: Results in the below chart that show '<1' or '>-1' suggest the presence of a correlation that is close to being exactingly negative or positive (but is still not quite perfectly so) and cases requiring special attention.")
    heatmap = msno.heatmap(df_with_imd_unique_pts)
    return heatmap

#df_with_imd_unique_pts.head()
#collisions = pd.read_csv("https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv")
#msno.heatmap(collisions)

#-----------------------------------------------------
def missing_values_field_names(fields_list, missing):
    """
    dict datafields (keys) and missing series (values)
    iterate over values in datafields dict, check if >0
    if so, create new dict of the subset where there are missing values (misisng == >0)
    then, iterate over that, and use each key/value pair, plug into impute function
    """

    keys = fields_list
    values = missing
    #dict_fieldname_missing_values = dict(zip(keys, values))
    #dict_fieldname_missing_values

    #Create dictionary of field names (keys) and county of missing values (values)
    iterator_fieldname_missing_values = zip(keys, values)
    dict_fieldname_missing_values = dict(iterator_fieldname_missing_values)

    #Use this dictionary to create a list, consisting of the subset of field names, where there are missing values
    #This will be used later in the program to identify which fields require missing values to be accounted for in some way
    missing_values_fields = []
    fields_none_missing = []

    for field in data_fields:
        if dict_fieldname_missing_values[field] >0:
            missing_values_fields.append(field)
        else:
            fields_none_missing.append(field)

    return dict_fieldname_missing_values, missing_values_fields 


#-----------------------------------------------------
#Function adjusted from the titanic example by Mike Allen
def impute_missing_with_median(series):
    """
    Replace missing values in a Pandas series with median,
    Returns a comppleted series, and a series showing which values are imputed
    """
    
    median = series.median(skipna = True) #ignore missing values in calculation
    missing = series.isna()
    
    # Copy the series to avoid change to the original series.
    series_copy = series.copy()
    series_copy[missing] = median
    
    return series_copy, missing
#-----------------------------------------------------
def group_fields_by_data_type(df, fields_list, dict_fieldname_missing_values):
    """
    Identify which fields both have missing values and are an appropriate data type to have their median calculated.
    This doesn't consider whether an int/float field is contextually appropriate to have a median calculated.
    E.g. if appt_id is an incremental numeric field, it will be logically identified and assigned to int_float_fields
    but, it is contextually meaningless to give a median appt_id. 
    Therefore, the output of this step needs some form of user validation / approval to proceed.
    This is to be handled in the next cell.
    """
    #df_with_imd_unique_pts.dtypes
    data_type_dict = dict(df.dtypes)

    int_float_fields = []
    str_fields = []
    datetime_fields = []

#original - poss error in df_with_imd_unique_pts
    """
    #identify which fields both have missing values and are int or float type. Append to list.
    for field in fields_list:
        if dict_fieldname_missing_values[field] > 0:
            if df_with_imd_unique_pts[field].dtype == "float" or df_with_imd_unique_pts[field].dtype == "int":
                int_float_fields.append(field)
            elif df_with_imd_unique_pts[field].dtype == "<M8[ns]":
                datetime_fields.append(field)
            else:
                str_fields.append(field)
    return int_float_fields, str_fields, datetime_fields, data_type_dict
    """

#experimental code attempt to correct error 

    #identify which fields both have missing values and are int or float type. Append to list.
    for field in fields_list:
        if dict_fieldname_missing_values[field] > 0:
            if df[field].dtype == "float" or df[field].dtype == "int":
                int_float_fields.append(field)
            elif df[field].dtype == "<M8[ns]":
                datetime_fields.append(field)
            else:
                str_fields.append(field)
    return int_float_fields, str_fields, datetime_fields, data_type_dict


#-----------------------------------------------------
"""
User confirmation these are all logically / contextually relevant 
(e.g. numeric patient ID could have median calculated but this doesnt make sense and
should be removed before the process to control for missing values takes place)

This function requires two input parameters: 
1. a field_list, and 
2. a data type as a string ("int" or "string")
"""


def identify_fields_to_impute(fields_list, data_type):
    #tell python which variables from the global scope to update
    global int_float_fields_keep
    global int_float_fields_drop
    global str_fields_keep
    global str_fields_drop
    global datetime_fields_keep
    global datetime_fields_drop
    
    #Create a dictionary of incremental numbers starting from 1 (keys) and 
    #field names identified in the last step as being int/float data type and having missing values (values)   
    dict_of_missing_fields = {}
    key_counter = 1
    for value in fields_list:
        dict_of_missing_fields[key_counter] = value
        key_counter += 1

    #Print to screen a numbered 'list' to the user, listing the fields that have been identified as int/float and missing values
    if data_type == "int":
        data_type_descriptor = "numeric (integer/decimal)"
    elif data_type == "str":
        data_type_descriptor = "string (text)"
    else:
        data_type_descriptor = "date time type"
        
    print(f"\nThe data fields with missing data that look to be {data_type_descriptor} are listed below")
    
    for key_number in dict_of_missing_fields:
        print(str(key_number) + " : " + str(dict_of_missing_fields[key_number]))

    #Get user confirmation that it is contextually appropriate to proceed with these fields for missing value control
    print("\nDo you want to impute missing data for all of these fields?")
    print("\nPlease enter 'y' to continue with all fields in scope, or 'n' to choose which fields to drop")

    while True:
        try:
            selection = input("Please make a selection (y or n): ").lower()
            if selection == "y" or selection == "n":
                break
            print("Invalid selection")
        except ValueError:
            print("\nThat is not a valid selection. Please try again (y or n).")   

    #loop to choose which fields to drop from the missing value process
    if selection == "n":
        while True:
            print("\nThe data fields currently in scope are listed below:")
            for key_number in dict_of_missing_fields:
                print(f"{key_number} : {dict_of_missing_fields[key_number]}")
            print("0 : To continue with the current selection")
        
            while True:
                try:
                    drop_selection = int(input("\nPlease enter the number of the field to remove, or 0 to continue with the current selection:"))
                    if drop_selection in range(key_counter):
                        break
                    else:
                        print("\nThat is not a correct selection.")
                except ValueError:
                    print("Value Error - try again.")

            if drop_selection == 0:
                break
            else:
                if data_type == "int":
                    int_float_fields_drop.append(dict_of_missing_fields[drop_selection]) #create new list of fields user does not want to control for missing values
                    del dict_of_missing_fields[drop_selection]
                elif data_type == "str":
                    str_fields_drop.append(dict_of_missing_fields[drop_selection]) #create new list of fields user does not want to control for missing values
                    del dict_of_missing_fields[drop_selection]
                else:
                    datetime_fields_drop.append(dict_of_missing_fields[drop_selection]) #create new list of fields user does not want to control for missing values
                    del dict_of_missing_fields[drop_selection]

        else:
            print("Please enter a valid selection.")
    
    if data_type == "int":
        int_float_fields_keep = fields_list
        for item in int_float_fields_drop:
            int_float_fields_keep.remove(item)
    elif data_type == "str":
        str_fields_keep = fields_list
        for item in str_fields_drop:
            str_fields_keep.remove(item)
    else:
        datetime_fields_keep = fields_list
        for item in datetime_fields_drop:
            datetime_fields_keep.remove(item)

    print()
    print(f"\nGreat. The following {data_type_descriptor} fields will be used when controlling for missing data:")
    
    if data_type == "int": 
        for item in int_float_fields_keep:
            print(item)
    elif data_type == "str":
        for item in str_fields_keep:
            print(item)   
    else:
        for item in datetime_fields_keep:
            print(item)   

            
#Dev note:
#Need to decide whether to remove df rows based on missing values in int_float_fields_drop and str_fields_drop lists or
#exclude the df columns these fields relate to entirely, 
#as ML can't handle Nulls - DECISION NEEDED -> ME/SW think remove column :)


#-----------------------------------------------------

def remove_inappropriate_appt_statuses(df, df_field, field_name):    
    print(f"\nThe {field_name} field that the program will use is: {df_field}")
    print("\nThe program requires a binary outcome measure (e.g. attended or did not attend).")
        
    ls = list(set(df[df_field]))
    
    unique_options_attend_status = [item for item in ls if type(item) != float]
    
    if len(unique_options_attend_status) == 2:
        print(f"{df_field} is a binary measure - perfect!")
    elif len(unique_options_attend_status) > 2:
        print(f"\n{df_field} has too many possible appointment statuses.")
        print("\nThe unique appointment statuses are listed below.")
        print("One by one, please enter the number for each option that represents a staus other than attended and did not attend (e.g. patient cancelled, etc.)")
        print("This will remove rows for those appointments from the data set to be used in the analysis.")
        print("Once the list only includes 2 appointment statuses, enter 0 to continue.")
              
        dict_of_unique_options = {}
        unique_key_counter = 1
        for option in unique_options_attend_status:
            dict_of_unique_options[unique_key_counter] = option
            unique_key_counter += 1    

        appt_status_to_drop = []
        appt_status_to_keep = []
        
        appt_status_to_drop.append("nan")
        
        #loop to choose which fields to drop from the missing value process
        while True:
            print("\nThe appointment statuses currently in scope are listed below:")
            for key_number in dict_of_unique_options:
                print(f"{key_number} : {dict_of_unique_options[key_number]}")
            print("0 : To continue with the current selection")

            while True:
                try:
                    drop_selection = int(input("\nPlease enter the number of the appointment status to remove, or 0 to continue with the current selection:"))
                    if drop_selection in range(unique_key_counter):
                        break
                    else:
                        print("\nThat is not a correct selection.")
                except ValueError:
                    print("Value Error - try again.")

            if drop_selection != 0:
                appt_status_to_drop.append(dict_of_unique_options[drop_selection]) #create new list of appt statuses user wants to exclude
                del dict_of_unique_options[drop_selection]
            elif drop_selection == 0:
                break

#                else:
#                    appt_status_to_keep.append(dict_of_unique_options[drop_selection]) #create new list of appt statuses user wants to include
#                    del dict_of_unique_options[drop_selection]

        else:
            print("Please enter a valid selection.")
    
    for key in dict_of_unique_options:
        appt_status_to_keep.append(dict_of_unique_options[key])
    
#    else:
#        continue #WARNING possible point of failure if there is only one appointment status in the column, not handled currently
          
    return appt_status_to_drop, appt_status_to_keep

#print(patient_id) #test print statement, delete




#-----------------------------------------------------
#Consider how to change the terminology used here, in case the term "missing" is used in the actual data set passed
#Taken from titanic - 01. pre-processing example:
def impute_missing_with_missing_label(series):
    """Replace missing values in a Pandas series with the text 'missing'"""

    missing = series.isna()    
    
    # Copy the series to avoid change to the original series.
    series_copy = series.copy()
    series_copy[missing] = 'missing'
    
    return series_copy, missing
#-----------------------------------------------------
def impute_missing(field_list, data_type):
    
    #Tell the function to update variables that exist in the global space, and not locally within the function
    global dict_ints_imputed
    global dict_ints_imputed_missing
    global dict_str_imputed
    global dict_str_imputed_missing
    
    if data_type == "int":
        for field in field_list:
            series, missing = impute_missing_with_median(df_with_imd_unique_pts[field])
            dict_ints_imputed[field] = series
            dict_ints_imputed_missing[field] = missing

    else:
        for field in field_list:
            series, missing = impute_missing_with_missing_label(df_with_imd_unique_pts[field])
            dict_str_imputed[field] = series
            dict_str_imputed_missing[field] = missing
#-----------------------------------------------------
#duplicate function for impute missing objective 1b - error using original for 2nd time
#testing new solution with this approach
def impute_missing_1b(df, field_list, data_type):
    
    #Tell the function to update variables that exist in the global space, and not locally within the function
    global dict_ints_imputed_1b
    global dict_ints_imputed_missing_1b
    global dict_str_imputed_1b
    global dict_str_imputed_missing_1b
    
    if data_type == "int":
        for field in field_list:
            series, missing = impute_missing_with_median(df[field])
            dict_ints_imputed_1b[field] = series
            dict_ints_imputed_missing_1b[field] = missing

    else:
        for field in field_list:
            series, missing = impute_missing_with_missing_label(df[field])
            dict_str_imputed_1b[field] = series
            dict_str_imputed_missing_1b[field] = missing
#-----------------------------------------------------

#identify the fields that are labels and not numbers (as the encoding will be done on label data only)
def fields_to_encode_or_drop(df):
    final_processed_data_fields = list(df.columns)
    fields_to_not_encode = []
    fields_to_encode = []
    fields_to_remove = [] #unsure if this is needed? had thought so for bools, but now removed those via alternative method

    for field in final_processed_data_fields:
        if field != patient_id and field != appt_id:
            if df[field].dtype == "float64" or df[field].dtype == "int" or df[field].dtype == "<M8[ns]":
                fields_to_not_encode.append(field)
            #elif df[field].dtype == "<M8[ns]":
            #    fields_to_remove.append(field)
            else: 
                fields_to_encode.append(field)
    
    return fields_to_not_encode, fields_to_encode, fields_to_remove
#-----------------------------------------------------

#create function to encode all relevant fields for a given data frame and list of fields to encode, passed into the function as parameters
def encoding_dataframe(dataframe, field_list_to_encode, field_list_to_not_encode):

    unencoded_df = pd.DataFrame()
    for field in field_list_to_not_encode:
        unencoded_df = pd.concat([unencoded_df, dataframe[field]], axis=1)
    
    #Create blank dataframe that will be populated with encoded series
    encoded_dataframe = pd.DataFrame()
    
    #iterate through all fields in the dataframe passed to the function
    for field in field_list_to_encode:       
        
        #encode the current field in the loop and create a dataframe for the current fields encoding 
        field_encoded = pd.get_dummies(dataframe[field], prefix="encoded")
        
        #concatenate the current fields encoding with the newly create dataframe above
        encoded_dataframe = pd.concat([encoded_dataframe, field_encoded], axis=1)

    combined_df = pd.concat([unencoded_df, encoded_dataframe], axis=1)
        
    return combined_df


#-----------------------------------------------------
#Function to remove bool series from final produced df
def remove_booleans_from_df(df):
    final_processed_df_bools_removed = pd.DataFrame()
    
    for column in df:
        if df[column].dtype != 'bool':
            final_processed_df_bools_removed[column] = df[column]
    return final_processed_df_bools_removed
#-----------------------------------------------------

#PROGRAM STARTS
#Read in the data set / create DataFrame

df = pd.read_csv('dummy_data_set_with_date_01.csv')
#print(df.head())

#-----------------------------------------------------
#Identify duplicate rows using pre-determined function
df_to_use = identify_duplicate_rows(df)

#-----------------------------------------------------
#Library called requests (pip install needed) - commonly used for http request
#requests.get(url) - this could get the data from a file on a given website, 
#store this in dataframe (or save it)

#Current non-API approach referencing downloaded full LSOA to IMD decile csv 
# from https://opendatacommunities.org/

#fine for pilot / project, beyond that would need to be scalable
df_lsoa_imd_decile = pd.read_csv('imd2019lsoa_decile.csv') 
df_lsoa_imd_decile = df_lsoa_imd_decile[['FeatureCode', 'Value']]

#-----------------------------------------------------
#merge IMD decile value into service data - result is new dataframe
df_with_imd = df_to_use.merge(df_lsoa_imd_decile, left_on='LSOA', right_on='FeatureCode', how='left').drop(columns='FeatureCode', axis=1)
df_with_imd.columns = df_with_imd.columns.str.lower()
df_with_imd = df_with_imd.rename(columns={"value": "IMD_decile", "attendance_status": "attend_status"})

#Identify the fields (columns) present in the data set
data_fields = list(df_with_imd.columns)

#Test print for user to visually check the joining of the datasets has worked
#print(df_with_imd.head())
#-----------------------------------------------------
#use function for user to assign the appropriate field name to patient_id variable (for later use)
patient_id = user_field_selection(data_fields, "unique patient ID")

#-----------------------------------------------------
#use function for user to assign the appropriate field name to appointment date_time variable (for later use)
appt_date_time = user_field_selection(data_fields, "Appointment Date and Time")

#-----------------------------------------------------
#convert the appt date and time field to a pandas datetime data type, using user identified field
df_with_imd[appt_date_time] = pd.to_datetime(df_with_imd[appt_date_time])

#Sort the data frame in ascending order
df_with_imd.sort_values(by = appt_date_time, inplace=True)

#test statement - delete - checking if appt_date_time changed to datetime type - it has 
#print(df_with_imd.dtypes)
#-----------------------------------------------------
#use function for user to assign the appropriate field name to appointment id to the appt_id variable (for later use)
appt_id = user_field_selection(data_fields, "appointment ID")

#-----------------------------------------------------
#use function for user to assign the appropriate field name to attendance status to the attend_status variable (for later use)
attend_status = user_field_selection(data_fields, "attendance status")

#-----------------------------------------------------
#Creating Processed Data for Objective 1A (HEA)
    
count_row_before = df_with_imd.shape[0]

#Create DataFrame for Objective 1A (HEA / profile):
#keep="last" as df earlier sorted ascending
#Assumption is because specific demographics can change, the latest known contact should be the most accurate

df_with_imd_unique_pts = df_with_imd.drop_duplicates(subset=patient_id, keep="last") 
count_row_after = df_with_imd_unique_pts.shape[0]
#df_with_imd_unique_pts.shape

print(f"Total rows with duplicate patients: {count_row_before}")
print(f"\nTotal rows with unique patients only: {count_row_after}")



#-----------------------------------------------------
#Data quality description section
list_names, present_values, missing_values, missing_pct, data_quality_summary_df = data_quality_summary(df_with_imd_unique_pts, data_fields)

#-----------------------------------------------------
#Test print statement for above function - can be used in final report
#print(data_quality_summary_df)

#-----------------------------------------------------
#Produce a stacked bar chart
#consider how to save chart to disk, and later use as an asset in a 
#compiled report
present_missing_stacked_bar(data_fields, present_values, missing_values, nhs_blue, nhs_dark_red, "Unique Patients")

#-----------------------------------------------------
#Call function to show data quality stacked bar chart with frequency counts table underneath
present_missing_stacked_bar_with_table(
    list_names, 
    data_fields, 
    present_values, 
    missing_values, 
    missing_pct,
    nhs_blue,
    nhs_dark_red,
    nhs_pale_grey,
    "Unique Patients")

#-----------------------------------------------------
#Call the matrix function to display DQ matrix
#Consider how to save to file, for use as an asset in a to be compiled report
dq_matrix(df_with_imd_unique_pts)

#-----------------------------------------------------
#Call the heatmap function - since adding date/time/datetime fields, 
#sorting ascending, and retaining latest known contact from the patient, 
#the correlation has changed for ethnicity and gender, but, the values in 
#those 2 fields (ethnicity and gender) havent been altered - group discussion 
#needed!
#My view: not sure this is adding value if we are now not worrying about 
#various means to control for missing data?
#Consider how to save to file, for use as an asset in a to be compiled report
heatmap(df_with_imd_unique_pts)

#-----------------------------------------------------
#Objective 1 begins

#Create dictionary of field names (keys) to count of missing values (values)
dict_fieldname_missing_values, missing_values_fields = missing_values_field_names(data_fields, missing_values)
#test print:
#print(dict_fieldname_missing_values)

#-----------------------------------------------------
#Call the function to create list of field names with missing values of 
#int/float type and string type respectively
int_float_fields, str_fields, datetime_fields, data_type_dict = group_fields_by_data_type(df_with_imd_unique_pts, data_fields, dict_fieldname_missing_values)

#-----------------------------------------------------
#List to populate with the fields the user decides to remove from / exclude from median (or other) missing values imputing
int_float_fields_keep = [] #fields to impute missing values for
int_float_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
str_fields_keep = [] #fields to impute missing values for
str_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
datetime_fields_keep = []
datetime_fields_drop = []

#-----------------------------------------------------
#CHECK REQUIRED - NEEDS TO CREATE NULLS / MISSING DATETIMES IN THE DUMMY 
#DATA SET TO CHECK THIS CELL FUNCTIONS AS EXPECTED - STILL TO DO..!

#allow user to decide whether to control for missing appt date times or exclude rows where this is empty
#Call function for user to confirm integer fields to retain - where there are integer fields with missing values only
if len(datetime_fields) > 0:
    identify_fields_to_impute(datetime_fields, "datetime")

if len(int_float_fields) > 0:
    identify_fields_to_impute(int_float_fields, "int")
    
if len(str_fields) > 0:
    identify_fields_to_impute(str_fields, "str")
    
#-----------------------------------------------------
#Global dicts for use when impute_missing function is called
dict_ints_imputed = {}
dict_ints_imputed_missing = {}
dict_str_imputed = {}
dict_str_imputed_missing = {}

#-----------------------------------------------------
#Call the function to impute missing int/float fields using the median
impute_missing(int_float_fields_keep, "int")

#Call the function to impute missing string fields with the word "missing"
impute_missing(str_fields_keep, "string")

#DECISION NEEDED: do we call impute_missing function for missing appointment 
#date times???

#-----------------------------------------------------
#Create a final df consisting of all the imputed fields, the fields for which 
#no imputing needed, and excluding all dropped fields
final_processed_df = pd.DataFrame()

data_fields_copy = data_fields.copy()

for field in data_fields:
    if field in int_float_fields_drop:
        data_fields_copy.remove(field)
    elif field in int_float_fields_keep:
        data_fields_copy.remove(field)
    elif field in str_fields_drop:
        data_fields_copy.remove(field)
    elif field in str_fields_keep:
        data_fields_copy.remove(field)

for field in data_fields_copy:
    final_processed_df[field] = df_with_imd_unique_pts[field]

#-----------------------------------------------------
#Test print - delete in final
#print(final_processed_df.info())

#-----------------------------------------------------
#For all fields with missing data that the user chose to keep/impute values for
#work through each included field, and where the missing row count for that field
#is >0, add a new series to the df for the series including imputed values
#and a boolean series indicating which was an imputed value

#The final processed df:
#1. includes fields that originally had no missing data
#2. excludes fields that had missing data, but the user decided to drop / not impute
#3. includes fields that originally had missing data, and the user imputed data for missing values

if len(int_float_fields_keep) > 0:
    for field in int_float_fields_keep:
        #final_processed_df[field] = (df_with_imd_unique_pts[field]) #Exclue original field with nulls
        final_processed_df[field+'_with_median'] = dict_ints_imputed[field]
        final_processed_df[field+'_Null_replaced'] = dict_ints_imputed_missing[field]
        
        #if field in list(final_processed_df.columns.values):
        #    final_processed_df = final_processed_df.drop(field, 1)
        
if len(str_fields_keep) > 0:
    for field in str_fields_keep:
        #final_processed_df[field] = (df_with_imd_unique_pts[field]) #Exclue original field with nulls
        final_processed_df[field+'_with_missing'] = dict_str_imputed[field]
        final_processed_df[field+'_Null_replaced'] = dict_str_imputed_missing[field]
        
        #if field in list(final_processed_df.columns.values):
        #    final_processed_df = final_processed_df.drop(field, 1)

#Test print statement - delete in final
#print(final_processed_df.info())

#-----------------------------------------------------
#Check if this df has nulls replaced as expected - delete when confirmed
#age known to have had missing values subsequently imputed in our dummy data
#replace the field name in square brackets as required, when using live data

#print(final_processed_df[final_processed_df['age_Null_replaced'] == True])

#-----------------------------------------------------
#call function to remove bool fields from the DataFrame
final_processed_df_bools_removed = remove_booleans_from_df(final_processed_df)

#-----------------------------------------------------
#check if the remove_booleans_from_df has worked as expected by filtering
#the df to the same index numbers known to have had age replaced above
#these index numbers are true for the dummy data set used to produce the code.
#replace with those identified in the above test print statement when using 
#live data

#Test print - delete 
#final_processed_df_bools_removed.loc[[117, 81, 125]]

#-----------------------------------------------------
#call function to create list of fields to omit from imputing function and 
#list of fields to inc in impute function

fields_to_not_encode, fields_to_encode, fields_to_remove = fields_to_encode_or_drop(final_processed_df_bools_removed)

#-----------------------------------------------------
#call the encoding_dataframe function
encoded_dataframe = encoding_dataframe(final_processed_df_bools_removed, fields_to_encode, fields_to_not_encode)

#Test print statements - delete in final
#print(encoded_dataframe.dtypes)
#Test to view list of final field names
#list(encoded_dataframe.reset_index(drop=True).columns)

#-----------------------------------------------------
#All done, save pre-processed file
os.makedirs('processed_data', exist_ok=True)  
encoded_dataframe.to_csv('processed_data/processed_data_1A_HEA_new.csv', index=False)  
print("File saved.")

#-----------------------------------------------------
#Creating Processed Data for Objective 1B: DNA Profiling

#-----------------------------------------------------
#removing duplicates based on user-instructed appt_id for objective 1B
#keep="last" as df earlier sorted ascending - consistent with approach to 1A
#appt ID (dup patients fine to inc in obj 2) but have all rows removed that 
#have an attendance status that isnt attended or DNA

original_df = df_with_imd.copy()

df_with_imd_unique_appts_1b = original_df.drop_duplicates(subset=appt_id, keep="last")
count_row_after_unique_appt_id = df_with_imd_unique_appts_1b.shape[0]

count_row_before = df_with_imd.shape[0]

print(f"Total rows with duplicate appt IDs: {count_row_before}")
print(f"\nTotal rows with unique appt IDs only: {count_row_after_unique_appt_id}")

#-----------------------------------------------------
#RELEVANT TO 1B ONLY - DNA PROFILING
#once tested and working for 1B processed data set production, remove the 
#doc stringed out function in function section
appt_status_to_drop, appt_status_to_keep = remove_inappropriate_appt_statuses(df_with_imd_unique_appts_1b, attend_status, "attendance status")  #field_list, df, field_name

#Test print statements - delete when final
#print(appt_status_to_drop) 
#print(appt_status_to_keep)

#-----------------------------------------------------
#RELEVANT TO 1B ONLY - DNA PROFILING
#Code to remove rows where appointment statuses are those user indicated to drop 
#this is on a test copied df for now. once confirmed working, repoint to 
#df_with_imd_unique_appts
#Delete Rows by Checking Conditions: any row in attend_status that is equal 
#to values in appt_status_to_drop to be removed from the df

df_with_imd_unique_appts_1b_dna_attend = df_with_imd_unique_appts_1b.copy()

for status in appt_status_to_drop:
    indexNames = df_with_imd_unique_appts_1b_dna_attend[df_with_imd_unique_appts_1b_dna_attend[attend_status] == status ].index
    df_with_imd_unique_appts_1b_dna_attend.drop(indexNames , inplace=True)

print(f"Original number of unique appointments: {df_with_imd_unique_appts_1b.shape[0]}")
print(f"Number of unique appointments with a status of attended or DNA inc Nulls: {df_with_imd_unique_appts_1b_dna_attend.shape[0]}")

#DECISION NEEDED: do we also dropna's from the attendance status column at 
#this point?

#df_with_imd_unique_appts_1b_dna_attend = df_with_imd_unique_appts_1b_dna_attend.dropna(subset=[attend_status])#
#print(f"Number of unique appointments with a status of attended or DNA exc. Nulls: {df_with_imd_unique_appts_1b_dna_attend.shape[0]}")

#Following 2 lines would remove null values in appt status field - GROUP DECISION NEEDED
#df_with_imd_unique_appts_1b_dna_attend = df_with_imd_unique_appts_1b_dna_attend.dropna(subset=['attend_status'])
#print(f"Net number of unique attended or DNA appts with Nulls removed: {df_with_imd_unique_appts_1b_dna_attend.shape[0]}")

#-----------------------------------------------------
list_names_1b, present_values_1b, missing_values_1b, missing_pct_1b, data_quality_summary_df_1b = data_quality_summary(df_with_imd_unique_appts_1b_dna_attend, data_fields)

#-----------------------------------------------------
#Test print statement for above function - can be used in final report
#print(data_quality_summary_df_1b)

#-----------------------------------------------------
#create dictionary of field names (keys) to missing value counts (values)
dict_fieldname_missing_values_1b, missing_values_fields_1b = missing_values_field_names(data_fields, missing_values_1b)

#-----------------------------------------------------
#Call the function to create list of field names with missing values of int/float type and string type respectively
int_float_fields_1b, str_fields_1b, datetime_fields_1b, data_type_dict_1b = group_fields_by_data_type(df_with_imd_unique_appts_1b_dna_attend, data_fields, dict_fieldname_missing_values_1b)

#-----------------------------------------------------
#Test print statements - can be deleted once complete.
#print(int_float_fields_1b)
#print(str_fields_1b) 
#print(datetime_fields_1b)
#print(data_type_dict_1b)

#-----------------------------------------------------
#List to populate with the fields the user decides to remove from / exclude from median (or other) missing values imputing
int_float_fields_keep = [] #fields to impute missing values for
int_float_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
str_fields_keep = [] #fields to impute missing values for
str_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
datetime_fields_keep = [] #fields to impute missing values for
datetime_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm

#-----------------------------------------------------
#CHECK REQUIRED - NEEDS TO CREATE NULLS / MISSING DATETIMES IN THE DUMMY DATA SET TO CHECK 
#THIS CELL FUNCTIONS AS EXPECTED - STILL TO DO..!

#allow user to decide whether to control for missing appt date times or 
#exclude rows where this is empty
if len(datetime_fields_1b) > 0:
    identify_fields_to_impute(datetime_fields_1b, "datetime")
    
#test print only
#print(datetime_fields_1b)

#-----------------------------------------------------
#Call function for user to confirm integer fields to retain - where there are 
#integer fields with missing values only
if len(int_float_fields_1b) > 0:
    identify_fields_to_impute(int_float_fields_1b, "int")

#-----------------------------------------------------   
#Call function for user to confirm string fields to retain - where there are 
#string fields with missing values only
if len(str_fields_1b) > 0:
    identify_fields_to_impute(str_fields_1b, "str")

#-----------------------------------------------------
#Global dicts for use when impute_missing function is called
dict_ints_imputed_1b = {}
dict_ints_imputed_missing_1b = {}
dict_str_imputed_1b = {}
dict_str_imputed_missing_1b = {}

#test print statements to inform logic - delete once working
#print(data_type_dict_1b.keys())
#print(data_type_dict_1b.values())
#print(dict_fieldname_missing_values_1b.values())
#print(data_fields)

#-----------------------------------------------------
#Call function to impute missing values using median for int/float
impute_missing_1b(df_with_imd_unique_appts_1b_dna_attend, int_float_fields_keep, "int")

#-----------------------------------------------------
#Call function to impute missing vaues for string fields using text "missing"
impute_missing_1b(df_with_imd_unique_appts_1b_dna_attend, str_fields_keep, "str")

#DECISION NEEDED: Do we also impute missing appt date/times or remove entirely?

#-----------------------------------------------------
#Create a final df consisting of all the imputed fields, the fields for 
#which no imputing needed, and excluding all dropped fields

final_processed_df_1b = pd.DataFrame()

data_fields_copy = data_fields.copy()

for field in data_fields_copy:
    if field in int_float_fields_drop:
        data_fields_copy.remove(field)
    elif field in int_float_fields_keep:
        data_fields_copy.remove(field)
    elif field in str_fields_drop:
        data_fields_copy.remove(field)
    elif field in str_fields_keep:
        data_fields_copy.remove(field)

for field in data_fields_copy:
    final_processed_df_1b[field] = df_with_imd_unique_appts_1b_dna_attend[field]

#Test print
#print(final_processed_df_1b.head())

#-----------------------------------------------------
#For all fields with missing data that the user chose to keep/impute values for
#work through each included field, and where the missing row count for that field
#is >0, add a new series to the df for the series including imputed values
#and a boolean series indicating which was an imputed value

#The final processed df:
#1. includes fields that originally had no missing data
#2. excludes fields that had missing data, but the user decided to drop / not impute
#3. includes fields that originally had missing data, and the user imputed data for missing values

if len(int_float_fields_keep) > 0:
    for field in int_float_fields_keep:
        #final_processed_df[field] = (df_with_imd_unique_pts[field]) #Exclue original field with nulls
        final_processed_df_1b[field+'_with_median'] = dict_ints_imputed_1b[field]
        final_processed_df_1b[field+'_Null_replaced'] = dict_ints_imputed_missing_1b[field]
        
        #Couldn't locate why gender field remained, but other string replaced fields removed. 
        #This for loop looks to solve the issue - for each field user says to keep, bring back the imputed values
        #Then, if that field is still in the df (for reasons not yet identified!) remove it
        #and only keep the new field with imputed values for blanks
        
        if field in list(final_processed_df_1b.columns.values):
            final_processed_df_1b = final_processed_df_1b.drop(field, 1)

if len(str_fields_keep) > 0:
    for field in str_fields_keep:
        #final_processed_df[field] = (df_with_imd_unique_pts[field]) #Exclue original field with nulls
        final_processed_df_1b[field+'_with_missing'] = dict_str_imputed_1b[field]
        final_processed_df_1b[field+'_Null_replaced'] = dict_str_imputed_missing_1b[field]
        
        if field in list(final_processed_df_1b.columns.values):
            final_processed_df_1b = final_processed_df_1b.drop(field, 1)
#-----------------------------------------------------
#Check if this df has nulls replaced as expected - delete when confirmed
#final_processed_df_1b[final_processed_df_1b['age_Null_replaced'] == True]

#-----------------------------------------------------
#Call function to remove bool series from df to encode
final_processed_df_bools_removed_1b = remove_booleans_from_df(final_processed_df_1b)

#Need to think of approach to make this generalisable, as attend statuses may vary
final_df_attended_only_obj_2 = final_processed_df_bools_removed_1b.loc[final_processed_df_bools_removed_1b['attend_status_with_missing']=='Attended']

os.makedirs('processed_data', exist_ok=True)  
final_df_attended_only_obj_2.to_csv('processed_data/processed_data_2_carbon_emissions.csv', index=False)  
print("File saved.")
#encoded_dataframe.to_csv('./data/processed_data.csv', index=False)


#Test print - delete 
#final_processed_df_bools_removed_1b.loc[[69, 68, 42, 29, 67, 81]]

#-----------------------------------------------------
#call function to create list of fields to omit from imputing function and 
#list of fields to inc in impute function
fields_to_not_encode_1b, fields_to_encode_1b, fields_to_remove_1b = fields_to_encode_or_drop(final_processed_df_bools_removed_1b)

#test print statements - remove in final edit
#print(fields_to_encode_1b)
#print()
#print(fields_to_not_encode_1b)
#print()
#print(fields_to_remove_1b)

#-----------------------------------------------------
#Test print statement - delete
#print(final_processed_df_bools_removed_1b.info())

#-----------------------------------------------------
#call the encoding_dataframe function
encoded_dataframe_1b = encoding_dataframe(final_processed_df_bools_removed_1b, fields_to_encode_1b, fields_to_not_encode_1b)

#Test print - delete in final
#print(encoded_dataframe_1b.dtypes)

#-----------------------------------------------------
#Save the file
os.makedirs('processed_data', exist_ok=True)  
encoded_dataframe_1b.to_csv('processed_data/processed_data_1B_DNA_Profile.csv', index=False)  
print("File saved.")
#encoded_dataframe.to_csv('./data/processed_data.csv', index=False)
