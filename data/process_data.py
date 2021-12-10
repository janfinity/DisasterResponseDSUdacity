
#import all relevant libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load messages data as well as the categories data

    Arguments:
        messages_filepath : path leading to the messages csv-file
        categories_filepath : path leading to the categories csv-file

    Output:
        df : combined data of messages and categories
    """
    #read messages csv
    messages = pd.read_csv(messages_filepath)
    #read categories csv
    categories = pd.read_csv(categories_filepath)
    #combine both dataframes in a new dataframe
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    Fuction to wrangle and clean the data in the dataframe

    Arguments:
        df : combined data of messages and categories

    Output:
        df : wrangled and cleaned dataframe of the combined data
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories
    category_colnames = [r[:-2] for r in row]
    #rename columns
    categories.columns = category_colnames

    #Convert values to numbers of 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    #Drop column 'original' and 'child_alone'
    df.drop(['original', 'child_alone'], axis=1, inplace=True)

    #Drop rows containing other values than 0 or 1
    df = df[df['related'] != 2]

    return df

def save_data(df, database_filename):
    """
    Function to save the data to a SQLite database

    Arguments:
        df : Combined and cleaned data
        database_filename : path to SQLite database
    """

    engine = create_engine('sqlite:///data/'+ database_filename)
    #create variabl to convert filepath into tablename
    table_name = database_filename.replace('.db','')
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
