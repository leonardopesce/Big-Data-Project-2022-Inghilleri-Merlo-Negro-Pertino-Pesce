####################################################################################################
#                    ___                                                                           #
#                  //   \\                                                                         #
#                 ||=====||                                                                        #
#                  \\___//                                                                         #
#                   ./O           System and Methods for Big and Unstructured Data - Group 2       #
#               ___/ //|\\                                                                         #
#              / o    /}                           Riccardo Inghilleri                             #
#             (       /                               Manuela Merlo                                #
#             \      /                                 Matteo Negro                                #
#             |     (                                 Paolo Pertino                                #
#             |      \                               Leonardo Pesce                                #
#             )       \                                                                            #
#            /         \                                                                           #
#          /            )                                                                          #    
#        /              |                                                                          #
#      //             / /                                                                          #
#    /       ___(    ,| \                                                                          #
#  /       /    \     |  \                                                                         #
# (      /  /   /\     \  \                                                                        #
# \\   /___ _-_//'|     |  |                                                                       #
#  \\_______-/     \     \  \                                                                      #
#                   \-_-_-_-_-                                                                     #
####################################################################################################
import pandas as pd
import numpy as np
import csv
import json
import jsonlines
from datetime import datetime
import glob
import random
import requests
import names
import lorem
import time
import tqdm
import pyarrow

from random_object_id import generate
from os import system, name
from os.path import isfile
from ast import literal_eval
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType, IntegerType, DateType
from pyspark.sql.functions import collect_set, col, expr, explode, array_contains, flatten, concat, sum, size
from pyspark import pandas as ps
import pyarrow as pa
import pyarrow.parquet as pq

# Set to a default number for having the same results in the group.
random.seed(1234)

####################################################################################################
#                                             PATHS                                                #
####################################################################################################

BOOK_PATH = "output_book.csv"
INCOLLECTION_PATH = "output_incollection.csv"
INPROCEEDINGS_PATH = "output_inproceedings.csv"
PROCEEDINGS_PATH = "output_proceedings.csv"
ARTICLE_PATH = "output_article.csv"
HAS_CITATIONS_PATH = "output_cite_has_citation.csv"
CITE_PATH = "output_cite.csv"
CITE_RELATIONSHIP = "output_cite_relationship.csv"
AUTHORS_DATA_PATH = "authors.csv"
AUTHORS_NODE_PATH = "output_author.csv"
AUTHORS_NODE_EXTENDED_PATH = "output_author_extended.csv"
BELONG_TO_PATH = "output_belong_relationship.csv"
PHD_THESIS_PATH = "output_phdthesis.csv"
PUBLISHED_IN_PATH = "output_journal_published_in.csv"
PART_OF_PATH = "output_series_is_part_of.csv"
PHD_THESIS_HEADER_PATH = "output_phdthesis_header.csv"
INPROCEEDINGS_HEADER_PATH = "output_inproceedings_header.csv"
ARTICLE_HEADER_PATH = "output_article_header.csv"
BOOK_HEADER_PATH = "output_book_header.csv"
PROCEEDINGS_HEADER_PATH = "output_proceedings_header.csv"
WWW_PATH = "output_www.csv"
INCOLLECTION_PATH = "output_incollection.csv"
AUTHORED_BY_PATH = "output_author_authored_by.csv"
AUTHORED_BY_FINAL_PATH = "output_author_authored_by_new.csv"
PUBLISHED_BY_PATH = "output_publisher_published_by.csv"
EDITED_BY_PATH = "output_editor_edited_by.csv"
PUBLISHER_PATH = "output_publisher.csv"
PAPERS_PATH_JSONL = "papers.jsonl"
REDUCED_PAPERS_PATH_JSONL = "reduced_papers.jsonl"
PAPERS_PATH_CSV = "papers.csv"
PAPERS_TEXT_MERGED_PATH = "papers_text.json"
PAPERS_FINAL_PATH = "papers_full_info.json"
ARTICLE_FINAL_PATH = "output_article_new.csv"
BOOK_FINAL_PATH = "output_book_new.csv"
PROCEEDINGS_FINAL_PATH = "output_proceedings_new.csv"
PHD_THESIS_FINAL_PATH = "output_phdthesis_new.csv"
INPROCEEDINGS_FINAL_PATH = "output_inproceedings_new.csv"
ARTICLE_FINAL_HEADER_PATH = "output_article_header_new.csv"
BOOK_FINAL_HEADER_PATH = "output_book_header_new.csv"
PROCEEDINGS_FINAL_HEADER_PATH = "output_proceedings_header_new.csv"
PHD_THESIS_FINAL_HEADER_PATH = "output_phdthesis_header_new.csv"
INPROCEEDINGS_FINAL_HEADER_PATH = "output_inproceedings_header_new.csv"
AUTHORED_BY_FINAL_PATH = "output_author_authored_by_new.csv"
PUBLISHED_BY_FINAL_PATH = "output_publisher_published_by_new.csv"
EDITED_BY_FINAL_PATH = "output_editor_edited_by_new.csv"
JOURNAL_PATH = "output_journal.csv"
ARTICLE_FINAL_PATH_EXTENDED = "output_article_final_extended_new.csv"
PARQUET_ARTICLE = "df_article.parquet"
PARQUET_JOURNAL = "df_journal.parquet"
PARQUET_AUTHOR = "df_author.parquet"

####################################################################################################
#                                         PARSER FUNCTIONS                                         # 
####################################################################################################
#                                              Neo4J                                               #
####################################################################################################

def authors_additional_information() -> pd.DataFrame:
    """Given the csv file containing the authors from dblp and the one coming from Semantic Scholar, it extends the data contained in the first dataset
        with the all the additional information contained in the second one.

    Returns:
        pd.Dataframe: a dataframe containing the information about the authors merged together.
    """
    author_data = pd.read_csv(AUTHORS_DATA_PATH, sep="\t", engine='python')
    author_node = pd.read_csv(AUTHORS_NODE_PATH, sep=";")

    # Dropping authors which doesn't have an alias on DBLP (we are only looking at dblp entries).
    author_data = author_data[author_data['externalids.DBLP'].notna()]
    author_data.drop(['authorid', 'externalids', 'aliases', 'updated'], axis=1, inplace=True)

    # Each author in DBLP can be recognized in different ways. In the dataset we used they were stored in an array. We explode that array, replicating 
    # for each possible alias the whole information of the author.
    author_data['externalids.DBLP'] = author_data['externalids.DBLP'].apply(literal_eval)
    n_author_data = author_data.explode('externalids.DBLP')

    # Merging the two dataframes on the DBLP name attribute
    author_node.rename(columns={'author:string' : 'key'}, inplace=True)
    n_author_data.rename(columns={'externalids.DBLP' : 'key'}, inplace=True)
    merged_frame = pd.merge(author_node, n_author_data, how='left', on = 'key')

    # We ony want a single entry for each author (there might be duplicates)
    merged_frame.drop(["key"], axis=1, inplace=True)
    merged_frame.drop_duplicates([':ID'], inplace=True)
    author_node.rename(columns={'key' : 'name'}, inplace=True)

    # Since some of the authors might be got filtered from the initial frame, then we concatenate it to the merged one and filter out the duplicates again
    # (the duplicates will be in this case all the authors of the concatenated frame which have extended info in the merged one)
    final_frame = pd.concat([merged_frame, author_node])
    final_frame.drop_duplicates([':ID'], keep='first', inplace=True)

    return final_frame

def reduce_frame(path: str, first_column_label: str, first_column_index : int, second_column_label : str, second_column_index : int) -> pd.DataFrame:
    """Reduces the columns of a Dataset represented by a Pandas dataframe by keeping only the 2 columns related to the 2 indexes passed as parameters, by giving them the respective labels.

    Args:
        path (str): the path of the csv file containing the data.
        first_column_label (str): the label to assign to the first column extracted.
        first_column_index (int): the index of the column of the dataframe representing the dataset to extract.
        second_column_label (str): the label to assign to the second column extracted.
        second_column_index (int): the index of the column of the dataframe representing the dataset to extract.

    Returns:
        pd.DataFrame: the dataset with only the two columns chosen with their respective new labels.
    """
    frame = pd.read_csv(path, sep = ";", low_memory=False)
    reduced_frame = pd.DataFrame({f"{first_column_label}" : frame.iloc[:, first_column_index], f"{second_column_label}" : frame.iloc[:, second_column_index]})
    return reduced_frame

def merge_cite_nodes() -> pd.DataFrame:
    """Elaborate the content of the files related to the CITE relationship.

    Returns:
        pd.DataFrame: elaborated dataframe corresponding to the CITE relationship
    """
    citation_frame = pd.read_csv(CITE_PATH, sep=";")
    has_citation_frame = pd.read_csv(HAS_CITATIONS_PATH, sep=";")

    citation_frame.rename(columns={':ID': ':END_ID'}, inplace=True)

    # The merged_frame will contain a pair (START_ID, key) in which the START_ID is the ID of the article which is citing another article, whereas key is the DBLP key
    # of the cited article.
    merged_frame = pd.merge(has_citation_frame, citation_frame)
    merged_frame.rename(columns={'cite:string': 'key'}, inplace=True)

    # Dropping citation references to a document with key ... - They're incosistent
    for index, row in merged_frame.iterrows():
        if row['key'] == '...':
            merged_frame.drop([index], inplace=True)

    merged_frame = pd.DataFrame(
        {':START_ID': merged_frame.iloc[:, 0], 'key': merged_frame.iloc[:, 2]})

    return merged_frame

def concat_all_publication_frames(reduced_article_frame : pd.DataFrame, reduced_book_frame : pd.DataFrame, reduced_incollection_frame : pd.DataFrame, reduced_inproceedings_frame : pd.DataFrame, reduced_proceedings_frame : pd.DataFrame) -> pd.DataFrame:
    """Given the dataframes related to the data of all the publications, returns a single dataset which is a collection of all the previously mentioned ones.

    Args:
        reduced_article_frame (pd.DataFrame): reduced dataframe containing the article ID and DBLP key
        reduced_book_frame (pd.DataFrame): reduced dataframe containing the book ID and DBLP key
        reduced_incollection_frame (pd.DataFrame): reduced dataframe containing the incollection ID and DBLP key
        reduced_inproceedings_frame (pd.DataFrame): reduced dataframe containing the inproceedings ID and DBLP key
        reduced_proceedings_frame (pd.DataFrame): reduced dataframe containing the proceedings ID and DBLP key

    Returns:
        pd.DataFrame: a merged dataframe containing all the dataset passed as arguments
    """
    
    return pd.concat([reduced_article_frame, reduced_book_frame, reduced_incollection_frame, reduced_inproceedings_frame, reduced_proceedings_frame])

def compile_relationship(frame_1 : pd.DataFrame, frame_2 : pd.DataFrame) -> pd.DataFrame:
    """Given two datasets which has a relationship between their elements, it returns a dataframe containing the START_ID and END_ID of all the elements involved.

    Args:
        frame_1 (pd.DataFrame): left-handside dataset of the relationship. 
        frame_2 (pd.DataFrame): right-handside dataset of the relationship.

    Returns:
        pd.DataFrame: a dataframe representing the relationship.
    """
    merged_frame = pd.merge(frame_1, frame_2)
    merged_frame = pd.DataFrame({':START_ID': merged_frame.iloc[:, 0], ':END_ID' : merged_frame.iloc[:, 2]})
    return merged_frame

def make_belong_relationship() -> pd.DataFrame:
    """Given all the inproceedings and all the proceedings it create a dataset which represents the BELONG_TO relationship (i.e. an inproceeding belongs to a certain proceeding).

    Returns:
        pd.DataFrame: a dataframe corresponding to the BELONG_TO relationship.
    """
    inproceedings_frame = pd.read_csv(INPROCEEDINGS_PATH, sep=";", header=None, low_memory=False)
    proceedings_frame = pd.read_csv(PROCEEDINGS_PATH, sep = ";", header=None, low_memory=False)

    # The inproceeding contain information which should be in the relationship between an inproceeding and a proceeding (such as the number, the pages and the volume).
    reduced_inproceedings_frame = pd.DataFrame({":START_ID" : inproceedings_frame.iloc[:, 0], "key" : inproceedings_frame.iloc[:, 8], "number" : inproceedings_frame.iloc[:, 18], "pages" : inproceedings_frame.iloc[:, 19], "volume" : inproceedings_frame.iloc[:, 27]})
    reduced_proceedings_frame = pd.DataFrame({":END_ID" : proceedings_frame.iloc[:, 0], "key" : proceedings_frame.iloc[:, 14]})
    
    # Belongs to frame with properties (number, pages, volume) creation
    belongs_to_frame = pd.merge(reduced_inproceedings_frame, reduced_proceedings_frame, on='key')
    belongs_to_frame.drop(["key"], inplace=True, axis=1)

    return belongs_to_frame

def update_published_in_relationship():
    """Given the published in raw relationship it add to it the number, pages and volume properties (initially contained in the article). Finally it saves a new csv file
    representing the updated relationship. 
    """
    published_in_frame = pd.read_csv(PUBLISHED_IN_PATH, sep=";", low_memory=False)
    article_frame = pd.read_csv(ARTICLE_PATH, sep=";", low_memory=False, header=None)

    # The article frame contains information which should be in the relationship between it and the journal it is published in.
    reduced_published_in_frame = pd.DataFrame({":START_ID" : published_in_frame.iloc[:, 0], ":END_ID" : published_in_frame.iloc[:, 1]})
    reduced_article_frame = pd.DataFrame({":START_ID" : article_frame.iloc[:, 0], "number" : article_frame.iloc[:, 22], "pages" : article_frame.iloc[:, 23], "volume" : article_frame.iloc[:, 33]})
    new_published_in_frame = pd.merge(reduced_published_in_frame, reduced_article_frame, on=":START_ID")

    new_published_in_frame.to_csv(PUBLISHED_IN_PATH, sep=";", index=False)

def update_part_of_relationship():
    """Given the part of raw relationship it add to it the volume properties (initially contained in a book or proceeding). Finally it saves a new csv file
    representing the updated relationship. 
    """
    part_of_frame = pd.read_csv(PART_OF_PATH, sep=";")
    book_frame = pd.read_csv(BOOK_PATH, sep=";", header=None, low_memory=False)
    proceedings_frame = pd.read_csv(PROCEEDINGS_PATH, sep=";", header=None, low_memory=False)

    # Both the books and the proceedings contains the volume of the series they are part of. So this information is transposed into the relationship.
    reduced_book_frame = pd.DataFrame({":START_ID" : book_frame.iloc[:, 0], "volume" : book_frame.iloc[:, 32]})
    reduced_proceedings_frame = pd.DataFrame({":START_ID" : proceedings_frame.iloc[:, 0], "volume" : proceedings_frame.iloc[:, 30]})
    reduced_part_of_frame = pd.DataFrame({":START_ID" : part_of_frame.iloc[:, 0], ":END_ID" : part_of_frame.iloc[:, 1]})
    book_and_proceedings_frame = pd.concat([reduced_book_frame, reduced_proceedings_frame])

    merged_frame = pd.merge(reduced_part_of_frame, book_and_proceedings_frame, on=':START_ID')
    merged_frame.to_csv(PART_OF_PATH, sep = ";", index=False)

def update_cites_relationship():
    """Removes from the CITE relationship frame, the entries which contains an homepage ID or an incollection ID in either their START_ID or END_ID."""

    cites_frame = pd.read_csv(CITE_RELATIONSHIP, sep = ";", low_memory=False)
    homepage_frame = pd.read_csv(WWW_PATH, sep = ";", low_memory=False)
    homepage_frame_start = pd.DataFrame({':START_ID' : homepage_frame.iloc[:, 0]})
    homepage_frame_end = pd.DataFrame({':END_ID' : homepage_frame.iloc[:, 0]})
    incollection_frame = pd.read_csv(INCOLLECTION_PATH, sep = ";", low_memory=False)
    incollection_frame_start = pd.DataFrame({':START_ID' : incollection_frame.iloc[:, 0]})
    incollection_frame_end = pd.DataFrame({':END_ID' : incollection_frame.iloc[:, 0]})
    general_frame_start = pd.concat([homepage_frame_start, incollection_frame_start])
    general_frame_end = pd.concat([homepage_frame_end, incollection_frame_end])

    # Cutting off the homepages and incollections from the cites relationship
    reduced_frame_start = pd.merge(cites_frame, general_frame_start, on=[':START_ID'])
    cond_start = cites_frame[':START_ID'].isin(reduced_frame_start[':START_ID'])
    cites_frame.drop(cites_frame[cond_start].index, inplace = True)
    reduced_frame_end = pd.merge(cites_frame, general_frame_end, on=[':END_ID'])
    cond_end = cites_frame[':END_ID'].isin(reduced_frame_end[':END_ID'])
    cites_frame.drop(cites_frame[cond_end].index, inplace = True)

    # Saving the new cites
    cites_frame.to_csv(CITE_RELATIONSHIP, sep=";", index=False)

def update_authoredby_relationship():
    """Removes from the AUTHORED_BY relationship frame, the entries which contains an homepage ID or an incollection ID in their START_ID."""

    authoredby_frame = pd.read_csv(AUTHORED_BY_PATH, sep = ";", low_memory=False)
    homepage_frame = pd.read_csv(WWW_PATH, sep = ";", low_memory=False, header=None)
    homepage_frame = pd.DataFrame({':START_ID' : homepage_frame.iloc[:, 0]})
    incollection_frame = pd.read_csv(INCOLLECTION_PATH, sep = ";", low_memory=False, header=None)
    incollection_frame = pd.DataFrame({':START_ID' : incollection_frame.iloc[:, 0]})
    general_frame = pd.concat([homepage_frame, incollection_frame])

    # Cutting off the homepages and incollections from the authoredby relationship
    reduced_frame = pd.merge(authoredby_frame, general_frame, on=[':START_ID'])
    cond = authoredby_frame[':START_ID'].isin(reduced_frame[':START_ID'])
    authoredby_frame.drop(authoredby_frame[cond].index, inplace = True)

    # Saving the new authoredby rels.
    authoredby_frame.to_csv(AUTHORED_BY_FINAL_PATH, sep=";", index=False)

def update_publishedby_relationship():
    """Removes from the PUBLISHED_BY relationship frame, the entries which contains an incollection ID in their START_ID."""
    publishedby_frame = pd.read_csv(PUBLISHED_BY_PATH, sep = ";", low_memory=False)
    incollection_frame = pd.read_csv(INCOLLECTION_PATH, sep = ";", low_memory=False, header=None)
    incollection_frame = pd.DataFrame({':START_ID' : incollection_frame.iloc[:, 0]})

    # Cutting off the incollections from the publishedby relationship
    reduced_frame = pd.merge(publishedby_frame, incollection_frame, on=[':START_ID'])
    cond = publishedby_frame[':START_ID'].isin(reduced_frame[':START_ID'])
    publishedby_frame.drop(publishedby_frame[cond].index, inplace = True)

    # Saving the new publishedby rel.
    publishedby_frame.to_csv(PUBLISHED_BY_FINAL_PATH, sep=";", index=False)

def update_editedby_relationship():
    """Removes from the EDITED_BY relationship frame, the entries which contains an homepage ID in their START_ID.
    """
    editedby_frame = pd.read_csv(EDITED_BY_PATH, sep = ";", low_memory=False)
    homepage_frame = pd.read_csv(WWW_PATH, sep = ";", low_memory=False, header=None)
    homepage_frame = pd.DataFrame({':START_ID' : homepage_frame.iloc[:, 0]})

    # Cutting off the incollections from the publishedby relationship
    reduced_frame = pd.merge(editedby_frame, homepage_frame, on=[':START_ID'])
    cond = editedby_frame[':START_ID'].isin(reduced_frame[':START_ID'])
    editedby_frame.drop(editedby_frame[cond].index, inplace = True)

    # Saving the new publishedby rel.
    editedby_frame.to_csv(EDITED_BY_FINAL_PATH, sep=";", index=False)

def clean_fields(path: str, field_idx: list):
    """Given a path of a dataset csv file and a list of indexes of the corresponding dataframe to drop, then it returns a dataframe with all those columns removed.

    Args:
        path (str): path of the csv file corresponding to a dataset.
        field_idx (list): list of indexes of the columns of the corresponding dataframe to drop.
    """
    frame = pd.read_csv(path, sep = ";", low_memory=False, header=None)

    # Dropping all the columns corresponding to the list of indexes passed.
    frame.drop(frame.columns[field_idx], axis=1, inplace=True)
    #if(path == ARTICLE_PATH):
    #    frame.iloc[:,13] = frame.iloc[:, 13].astype('Int64')
    #frame.to_csv(path[:len(path)-4] + '_new.csv', sep = ";", index=False, header=False)
    if path == ARTICLE_PATH:
        frame.iloc[:,13] = frame.iloc[:, 13].astype('Int64')
        final_path = ARTICLE_FINAL_PATH
    elif path == BOOK_PATH:
        final_path = BOOK_FINAL_PATH
    elif path == PROCEEDINGS_PATH:
        final_path = PROCEEDINGS_FINAL_PATH
    elif path == PHD_THESIS_PATH:
        final_path = PHD_THESIS_FINAL_PATH
    elif path == INPROCEEDINGS_PATH:
        final_path = INPROCEEDINGS_FINAL_PATH
        
    frame.to_csv(final_path, sep = ";", index=False, header=False)
            




def clean_fields_header(path: str, field_idx: list):
    """Given a path of a dataset headers csv file and a list of indexes of the corresponding dataframe to drop, then it returns a dataframe with all those columns removed.

    Args:
        path (str): path of the csv file containing the headers of a dataframe.
        field_idx (list): list of indexes of the columns of the corresponding headers to drop.
    """
    frame = pd.read_csv(path, sep = ";", low_memory=False)
    frame.drop(frame.columns[field_idx], axis=1, inplace=True)
    # frame.to_csv(path[:len(path)-4] + '_new.csv', sep = ";", index=False)
    if path == ARTICLE_HEADER_PATH:
        final_path = ARTICLE_FINAL_HEADER_PATH
    elif path == BOOK_HEADER_PATH:
        final_path = BOOK_FINAL_HEADER_PATH
    elif path == PROCEEDINGS_HEADER_PATH:
        final_path = PROCEEDINGS_FINAL_HEADER_PATH
    elif path == PHD_THESIS_HEADER_PATH:
        final_path = PHD_THESIS_FINAL_HEADER_PATH
    elif path == INPROCEEDINGS_HEADER_PATH:
        final_path = INPROCEEDINGS_FINAL_HEADER_PATH

    frame.to_csv(final_path, sep = ";", index=False)


def neo4jSetup():
    """Perform all the actions to clean the data and get them ready to be imported in Neo4J."""

    progress_bar(5, 100, "FETCHING PARTICULAR COLUMNS OF ARTICLE DATAFRAME")
    article_frame = reduce_frame(ARTICLE_PATH, "ArticleId", 0, "key", 16)
    progress_bar(20, 100, "FETCHING PARTICULAR COLUMNS OF BOOK DATAFRAME")
    book_frame = reduce_frame(BOOK_PATH, "ArticleId", 0, "key", 16)
    progress_bar(25, 100, "FETCHING PARTICULAR COLUMNS OF INCOLLECTION DATAFRAME")
    incollection_frame = reduce_frame(INCOLLECTION_PATH, "ArticleId", 0, "key", 12)
    progress_bar(30, 100, "FETCHING PARTICULAR COLUMNS OF INPROCEEDINGS DATAFRAME")
    inproceedings_frame = reduce_frame(INPROCEEDINGS_PATH, "ArticleId", 0, "key", 14)
    progress_bar(35, 100, "FETCHING PARTICULAR COLUMNS OF PROCEEDINGS DATAFRAME")
    proceedings_frame = reduce_frame(PROCEEDINGS_PATH, "ArticleId", 0, "key", 14)

    progress_bar(45, 100, "CONCATENATING ALL PUBLICATION FRAMES TOGETHER")
    concatenated_publications_frame = concat_all_publication_frames(article_frame, book_frame, incollection_frame, inproceedings_frame, proceedings_frame)

    progress_bar(55, 100, "EXTENDING AUTHORS INFORMATION")
    authors_data = authors_additional_information()
    authors_data.to_csv(AUTHORS_NODE_EXTENDED_PATH, sep=";", index=False)

    progress_bar(70, 100, "COMPILING CITE RELATIONSHIP")
    cite_relationship = compile_relationship(merge_cite_nodes(), concatenated_publications_frame)
    progress_bar(73, 100, "COMPILING BELONG TO RELATIONSHIP")
    belong_relationship = make_belong_relationship()
    progress_bar(75, 100, "WRITING CITE & BELONG TO RELATIONSHIP TO FILE")
    cite_relationship.to_csv("output_cite_relationship.csv", sep=";", index=False)
    belong_relationship.to_csv(BELONG_TO_PATH, sep=";", index=False)

    # Cleaning up relationships
    progress_bar(80, 100, "UPDATING PUBLISHEDIN RELATIONSHIP")
    update_published_in_relationship()
    progress_bar(82, 100, "UPDATING PART OF RELATIONSHIP")
    update_part_of_relationship()
    progress_bar(84, 100, "UPDATING CITES RELATIONSHIP")
    update_cites_relationship()
    progress_bar(86, 100, "UPDATING AUTHOREDBY RELATIONSHIP")
    update_authoredby_relationship()
    progress_bar(88, 100, "UPDATING PUBLISHEDBY RELATIONSHIP")
    update_publishedby_relationship()
    progress_bar(90, 100, "UPDATING EDITEDBY RELATIONSHIP")
    update_editedby_relationship()
    
    # Delete useless phdthesis, inproceedings, articles, book, proceedings properties
    progress_bar(91, 100, "CLEANING PROPERTIES FROM PUBLICATIONS FILES")
    clean_fields(PHD_THESIS_PATH, [1, 2, 5, 13, 14, 15, 17, 18, 19, 20, 21])
    clean_fields(INPROCEEDINGS_PATH, [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 19, 20, 22, 23, 25, 26, 28])
    clean_fields(ARTICLE_PATH, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 14, 15, 22, 23, 24, 25, 27, 28, 30, 31, 33])
    clean_fields(BOOK_PATH, [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 22, 23, 25, 26, 27, 28, 29, 32])
    clean_fields(PROCEEDINGS_PATH, [2, 4, 5, 6, 7, 10, 20, 21, 23, 24, 25, 26, 27, 30])
    progress_bar(95, 100, "UPDATING HEADER FILES")
    clean_fields_header(PHD_THESIS_HEADER_PATH, [1, 2, 5, 13, 14, 15, 17, 18, 19, 20, 21])
    clean_fields_header(INPROCEEDINGS_HEADER_PATH, [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 19, 20, 22, 23, 25, 26, 28])
    clean_fields_header(ARTICLE_HEADER_PATH, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 14, 15, 22, 23, 24, 25, 27, 28, 30, 31, 33])
    clean_fields_header(BOOK_HEADER_PATH, [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 22, 23, 25, 26, 27, 28, 29, 32])
    clean_fields_header(PROCEEDINGS_HEADER_PATH, [2, 4, 5, 6, 7, 10, 20, 21, 23, 24, 25, 26, 27, 30])
    progress_bar(100, 100, "COMPLETED")

####################################################################################################
#                                            MongoDB                                               #
####################################################################################################

def reduce_jsonl_dataset(path: str, entries_to_keep: int):
    """Given the path of a jsonl file it stores a new one with only entries_to_keep entries.

    Args:
        path (str): path of the jsonl file.
        entries_to_keep (int): number of entries to keep.
    """
    with jsonlines.open(path, mode = "r") as complete_file, jsonlines.open('reduced_' + path, mode = "w") as reduced_file:
        i = 0
        for line in complete_file:
            # print(type(line))
            if(line.get("externalids").get("DBLP") is not None):
                i += 1
                reduced_file.write(line)
            if(i == entries_to_keep):
                break

def integrate_authors_info():
    """Given an article, it creates for each author an object with meaningful metadata about him/her."""
    authors_frame = pd.read_csv(AUTHORS_DATA_PATH, sep="\t", engine='python')
    paper_updated_info = []

    with jsonlines.open(REDUCED_PAPERS_PATH_JSONL, mode = "r") as papers:
        for paper in papers:
            authors_of_paper = paper.get('authors')
            updated_authors = []
            for author in authors_of_paper:
                author_name = author.get('name')
                author_id = author.get('authorId')
                authors_generalities = author_name.split()
                if(author_id is not None):
                    author_additional_data = authors_frame.loc[authors_frame['authorid'] == int(author_id)].copy()
                    author_additional_data.dropna(axis=1, inplace=True)
                    author_additional_data = author_additional_data.to_dict(orient='records')
                    if(len(author_additional_data) > 0):
                        author_additional_data = author_additional_data[0]
                        author_additional_data.pop('updated', None)
                        author_additional_data.pop('aliases', None)
                        try:
                            author_additional_data.update({
                                'email' : f"{authors_generalities[0]}{authors_generalities[1]}@gmail.com"
                            })
                        except:
                            author_additional_data.update({
                                'email' : f"SMBUD_{authors_generalities[0]}@gmail.com"
                            })
                        updated_authors.append(author_additional_data)
            paper.update({'authors' : updated_authors})
            paper_updated_info.append(paper)
    
    with jsonlines.open(REDUCED_PAPERS_PATH_JSONL, mode = "w") as papers:
        for paper in paper_updated_info:
            papers.write(paper)

def merge_jsons():
    """Merges all the json documents inside the ./json_docs folder into a single object."""
    documents = []
    for filename in glob.glob('./json_docs/*.json'):
        with open(filename, 'r') as file:
            document = json.load(file)
            content = document.get('body_text')

            # We don't care about citations in between articles. They'll be explicited in another property of the article.
            for sec in content:
                sec.pop('cite_spans', None)

                # Dropping invalid references to figures
                new_ref_spans = []
                for ref_span in sec.get('ref_spans'):
                    if(ref_span.get('ref_id') is not None):
                        new_ref_spans.append(ref_span)
                
                if(len(new_ref_spans) > 0):
                    sec.update({
                        "figures_in_paragraph" : new_ref_spans
                    })
                    sec.pop('ref_spans', None)
                else:
                    sec.pop('ref_spans', None)
            
            documents.append(
                {
                    "content" : content,
                    "list_of_figures" : document.get('ref_entries')
                }
            )
    
    with open(PAPERS_TEXT_MERGED_PATH, "w") as articles_text_file:
        json.dump(documents, articles_text_file, indent=4)

def integrate_text_info():
    """Inserts in each article object some content (paragraphs, text, figures, keywords...)"""
    new_articles = []
    possible_keywords = ["MACHINE LEARNING", "DEEP LEARNING", "BIG DATA", "GPU", "HTML", "PYTHON", "C++", "JAVA", "UI", "UX", "FLUTTER", "NEO4J", "MONGODB", "SPARK"]
    with open(PAPERS_TEXT_MERGED_PATH, 'r') as articles_text_file, jsonlines.open(REDUCED_PAPERS_PATH_JSONL, 'r') as articles_notext_file:
        articles_text = json.load(articles_text_file)

        for i, article in enumerate(articles_notext_file):
            new_article = article
            foss = [fos.get('category') for fos in article.get('s2fieldsofstudy')] if article.get('s2fieldsofstudy') != None else None
            already_picked_fos = []
            if foss != None:
                for fos in foss:
                    if fos not in already_picked_fos:
                        already_picked_fos.append(fos)
                        
            new_article.update({
                "s2fieldsofstudy" : already_picked_fos,
                "content" : {
                    "sections" : articles_text[i].get("content"),
                    "list_of_figures" : articles_text[i].get("list_of_figures"),
                    "keywords" : random.choices(possible_keywords, k = random.randint(1, 4))
                }
            })

            new_articles.append(new_article)

    with open(PAPERS_FINAL_PATH, "w") as papers_file:
        json.dump(new_articles, papers_file, indent=4)

def aggregate_sections():
    """Aggregates the paragraphs belonging to the same section in a section object."""
    aggregated_papers = ...
    with open(PAPERS_FINAL_PATH, "r") as papers_file:
        papers = json.load(papers_file)
        
        for paper in papers:
            alredy_picked_sections = []
            new_sections = []
            for section in paper.get('content').get('sections'):
                if section.get('section') not in alredy_picked_sections:
                    alredy_picked_sections.append(section.get('section'))
                    new_section = {
                        "section_title" : section.pop('section', None),
                        "paragraphs" : [section]
                    }
                    new_sections.append(new_section)
                else:
                    section.pop('section', None)
                    paragraph_to_update = new_sections[len(new_sections) - 1].get('paragraphs')
                    paragraph_to_update.append(section)
                    new_sections[len(new_sections) - 1].update({
                        'paragraphs' : paragraph_to_update
                    })
            
            paper.get('content').update({
                'sections' : new_sections
            })
        
        aggregated_papers = papers
    
    # Since null-value fields should not be added in MongoDB we pop them out from our data.
    cleaned_papers = []
    for paper in aggregated_papers:
        new_authors = []
        for author in paper.get('authors'):
            author = remove_null_values(author)
            new_authors.append(author)
        paper.update({'authors' : new_authors})
        paper.get('content').update({'list_of_figures' : remove_null_values(paper.get('content').get('list_of_figures'))})
        paper = remove_null_values(paper)
        cleaned_papers.append(paper)

    with open(PAPERS_FINAL_PATH, "w") as papers_file:
        json.dump(cleaned_papers, papers_file, indent = 4)  

def remove_null_values(d):
    """Given a dictionary, drop all the keys associated to null values.

    Args:
        d (dict): dictionary for which the keys associated to null values have to be dropped.

    Returns:
        dict: dictionary containing only those keys of the initial dictionary which are associated with non-null values.
    """
    return {
        key: remove_null_values(value) if isinstance(value, dict) else value
        for key, value in d.items()
        if value != None
    }   

def integrate_images_in_paragraphs():
    """Integrates images and their metadata in the paragraphs."""
    new_papers = ...
    with open(PAPERS_FINAL_PATH, "r") as papers_file:
        papers = json.load(papers_file)
        
        # Generating additional metadata for each figure
        url = f'https://picsum.photos/v2/list?limit=100'
        request = requests.get(url)
        request.raise_for_status()
        images = request.json()

        for paper in papers:
            list_of_figures_in_paper = paper.get('content').get('list_of_figures') # Dictionary containing all the figures entries of the paper).
            for key, figure in list_of_figures_in_paper.items():
                if(key.startswith("FIGREF")):
                    image_idx = random.randint(0, 99)
                    figure.update({
                        'author' : images[image_idx].get('author'),
                        'width' : images[image_idx].get('width'),
                        'height' : images[image_idx].get('height'),
                        'url' : images[image_idx].get('url'),
                        'download_url' : images[image_idx].get('download_url'),
                    })
            
            list_of_sections = paper.get('content').get('sections')            # Array of paragraphs objects
            for section in list_of_sections:
                list_of_paragraphs = section.get('paragraphs')
                for paragraph in list_of_paragraphs:
                    if((figures_in_paragraph := paragraph.get('figures_in_paragraph')) != None):
                        for fig in figures_in_paragraph:
                            for k, f in list_of_figures_in_paper.items():
                                if (fig.get('ref_id') == k):
                                    fig.update({
                                        'caption' : fig.pop('text', None)
                                    })
                                    fig.update(f)
            
            # In the overall section of figures of the article we want only few data.
            entries_to_remove = ('text', 'width', 'height', 'download_url')
            for key, figure in list_of_figures_in_paper.items():
                if(key.startswith("FIGREF")):
                    for k in entries_to_remove:
                        figure.pop(k, None)

            # Change list_of_figures paradigm, from single document to array
            list_of_figures_in_paper = paper.get('content').get('list_of_figures')
            new_list_of_figures = []
            for key, figure in list_of_figures_in_paper.items():
                figure.update({
                    'fig_id' : key
                })
                new_list_of_figures.append(figure)
            
            if(len(new_list_of_figures) > 0):
                paper.get('content').update({
                    'list_of_figures' : new_list_of_figures
                })
            else:
                paper.get('content').pop('list_of_figures')
                    
        
        new_papers = papers

    with open(PAPERS_FINAL_PATH, "w") as papers_file:
        json.dump(new_papers, papers_file, indent = 4)

def generate_subsection(probability: int, super_section_title : str, has_figures : bool) -> list:
    """Generate recursively subsections.

    Args:
        probability (int): the probability for which a new subsection will be generated.
        super_section_title (str): the title of the section for which a subsection is generated.
        has_figures (bool): true if the section for which a subsection is generated has figures.

    Returns:
        list: list of generated subsections.
    """
    paragraph_of_subsection = []
    
    # Generating random paragraphs for the subsection (max 3 paragraphs per subsection)
    for i in range(random.randint(1, 3)):
        if(has_figures and random.randint(0, 100) < 30):
            if(random.randint(0, 100) < 5):
                paragraph = {
                    'text' : lorem.paragraph(),
                    'figures_in_paragraph' : [{
                        "ref_id": "FIGREF0",
                        "caption": "We love SMBUD <3",
                        "text": "Best professors <3",
                        "type": "figure",
                        "author": "Marco Brambilla & Andrea Tocchetti",
                        "width": 2022,
                        "height": 2022,
                        "url": "https://drive.google.com/file/d/1OO-66JJlwrVSOXXAA96PiBmEKQ_jI_4X/view?usp=share_link",
                        "download_url": "https://i1.rgstatic.net/ii/profile.image/1123684051890181-1644918563576_Q128/Andrea-Tocchetti.jpg"
                    }],
                }
            else:
                paragraph = {
                    'text' : lorem.paragraph(),
                    'figures_in_paragraph' : [{
                        "ref_id": "FIGREF0",
                        "caption": "A funny easter egg.",
                        "text": "But there might be something funnier UwU",
                        "type": "figure",
                        "author": "SMBUD Group 2",
                        "width": 2022,
                        "height": 2022,
                        "url": "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.robinsonpetshop.it%2Fnews%2Fgatto%2Fadottare-un-cucciolo-di-gatto-7-cose-sapere%2F&psig=AOvVaw3KOs6kst4pmV_AbT79jz2S&ust=1668619974297000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCMDS7PnbsPsCFQAAAAAdAAAAABAD",
                        "download_url": "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.terranuova.it%2FNews%2FAmbiente%2FLa-mattanza-dei-cuccioli-di-foca&psig=AOvVaw1R-7W8F0iuaTUzSHdxxA-P&ust=1668619999647000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCOiMzYXcsPsCFQAAAAAdAAAAABAD"
                    }],
                }
        else:
            paragraph = {
                'text' : lorem.paragraph()
            }
        paragraph_of_subsection.append(paragraph)
    
    new_probability = probability // 2
    if(random.randint(0, 100) < new_probability):
        section = {
            'section_title' : 'subsection of ' + super_section_title,
            'paragraphs' : paragraph_of_subsection,
            'subsections' : generate_subsection(new_probability, 'subsection of ' + super_section_title, has_figures)
        }
    else:
        section = {
            'section_title' : 'subsection of ' + super_section_title,
            'paragraphs' : paragraph_of_subsection,
        }
    list_of_subsections = []
    for i in range(random.randint(1,3)):
        new_section = section.copy()
        olt_title = new_section.get('section_title')
        new_section.update({
            'section_title' : f'{i+1} ' + olt_title
        })
        list_of_subsections.append(new_section)
    
    return list_of_subsections

def generate_subsections():
    """For each section it random generates subsections."""
    new_papers = []
    with open(PAPERS_FINAL_PATH, "r") as papers_file:
        papers = json.load(papers_file)

        for paper in papers:
            paper_has_figures = (paper.get('content').get('list_of_figures') != None and len(paper.get('content').get('list_of_figures')) > 0)
            list_of_sections = paper.get('content').get('sections')
            for section in list_of_sections:
                probability_of_subsection = 50
                sub_section = random.randint(0, 100) < probability_of_subsection
                if(sub_section):
                    section.update({
                        'subsections' : generate_subsection(probability_of_subsection, section.get('section_title'), paper_has_figures)
                    })
            new_papers.append(paper)
    
    with open(PAPERS_FINAL_PATH, "w") as papers_file:
        json.dump(new_papers, papers_file, indent = 4)


def generate_citations():
    """Generates for each article a set of random citations to other articles in the set by looking at the citationcount property of the article."""
    papers_with_citations = []

    with open(PAPERS_FINAL_PATH, "r") as papers_file:
        papers = json.load(papers_file)

        # First of all we need to collect all the identifiers and useful metadata of each article and its number of citations.
        papers_metadata = []
        for paper in papers:
            papers_metadata.append({
                'corpusid' : paper.get('corpusid'),
                'citationcount' : paper.get('citationcount') if paper.get('citationcount') != None else 0,
                'title' : paper.get('title'),
                'authors' : [author.get('name') for author in paper.get('authors')],
                'ingoing_citations' : 0
            })
        
        # Now for each article we create an array of citations to other articles, following these rules:
        #   - each article cannot cite itself.
        #   - an article A can cite an article B only once.
        #   - there is consistency between citationcount property of the article.
        for i, paper in enumerate(papers):
            already_cited_papers = []
            citations_of_paper = []
            j = 0
            while j < papers_metadata[i].get('citationcount'):
                random_article_idx = random.randint(0, 4999)
                if random_article_idx not in already_cited_papers and random_article_idx != i:
                    citations_of_paper.append({
                        'reference_id' : papers_metadata[random_article_idx].get('corpusid'),
                        'title' : papers_metadata[random_article_idx].get('title'),
                        'authors' : papers_metadata[random_article_idx].get('authors')
                    })

                    # Then we save for each article which has been cited the number of times it has been randomly picked in order to reconstruct consistency 
                    # also over the referencecount field.
                    new_ingoing_citation_counter = papers_metadata[random_article_idx].get('ingoing_citations') + 1
                    papers_metadata[random_article_idx].update({
                        'ingoing_citations' : new_ingoing_citation_counter
                    })
                    j += 1
            paper.update({
                'citations' : citations_of_paper
            })
            paper.pop('influentialcitationcount', None)
            papers_with_citations.append(paper)
        
        # Reconstructing the referencecount field consistency.
        for i, paper in enumerate(papers_with_citations):
            paper.update({
                'referencecount' : papers_metadata[i].get('ingoing_citations')
            })
    
    with open(PAPERS_FINAL_PATH, "w") as papers_file:
        json.dump(papers_with_citations, papers_file, indent = 4)

def fix_dates():
    """Makes dates compatible with MongoDB ISODate datatype."""
    new_papers = []
    with open(PAPERS_FINAL_PATH, "r") as papers_file:
        papers = json.load(papers_file)

        # publicationdate
        #updated
        for paper in papers:
            publdate = paper.get('publicationdate')
            updated = paper.get('updated')
            paper.update({
                'publicationdate' : {
                    '$date' : publdate
                },
                'updated' : {
                    '$date' : updated
                }
            })
            new_papers.append(paper)
        
    with open(PAPERS_FINAL_PATH, "w") as papers_file:
        json.dump(new_papers, papers_file, indent=4)


####################################################################################################
#                                       MongoDB Random Setup                                       #
####################################################################################################

def mongo_db_setup_random(number_of_papers: int = 5000):
    skeleton_doc = '{"corpusid": 0, "abstract": "", "updated": {}, "externalids": {}, "url": "", "title": "", "authors": [], "venue": "", "year": 0, "referencecount": 0, "citationcount": 0, "influentialcitationcount": 0, "isopenaccess": true, "s2fieldsofstudy": [], "publicationtypes": [], "publicationdate": "", "journal": {}, "content": {"sections": [],"list_of_figures": []}, "citations": []}'
    skeleton_aut = '{"authorid": 0, "name": "", "papercount": 0, "citationcount": 0, "hindex": 0, "s2url": "", "externalids.DBLP": ""}'
    skeleton_par = '{"text": ""}'
    skeleton_fig = '{"start": 0, "end": 0, "ref_id": "", "caption": "", "text": "", "type": "figure", "author": "", "width": 0, "height": 0, "url": "", "download_url": ""}'
    skeleton_sec = '{"section_title": "", "paragraphs": []}'
    skeleton_fig_small = '{"type": "figure", "author": "", "url": "", "fig_id": ""}'
    skeleton_cit = '{"reference_id": 0, "title": "","authors": []}'

    docArray = []
    oid_title_authors_list = []
    authors_name_list = []
    for _ in range(number_of_papers):
        authors = [gen_aut(skeleton_aut) for _ in range(random.randint(1, 5))]
    oid_title_authors_list.append([generate(), lorem.sentence(), authors])
    authors_name_list.append([author["name"] for author in authors])

    for i in range(number_of_papers):
        doc = json.loads(skeleton_doc)
    doc["corpusid"] = i
    doc["abstract"] = lorem.paragraph()
    date = random.randint(2010, 2022)
    doc["update"] = {"$date": f'{date}-{str(random.randint(1, 12)).zfill(2)}'
                              f'-{str(random.randint(1, 28)).zfill(2)}T{str(random.randint(1, 24)).zfill(2)}'
                              f':{str(random.randint(1, 60)).zfill(2)}:{str(random.randint(1, 60)).zfill(2)}'
                              f'.000+00:00'}
    doc["exertenalids"] = {
        "DBLP": f"jornals/{i}/{oid_title_authors_list[i][1].replace(' ', '')}",                                    # PROBLEMA QUA
        "MAG": str(random.randint(1000000000, 9999999999)),
        "corpusid": str(i),
        "DOI": str(random.randint(10000000000, 99999999999))
    }
    doc["URL"] = f'https://www.semanticscholar.org/paper/{oid_title_authors_list[i][1].replace(" ", "")}'
    doc["title"] = oid_title_authors_list[i][1]
    doc["authors"] = oid_title_authors_list[i][2]
    doc["venue"] = lorem.sentence().split(' ')[0] + lorem.sentence().split(' ')[0]
    doc["year"] = random.randint(2009, date)
    numRef = random.randint(1, 100)
    doc["referencecount"] = numRef
    numCit = random.randint(1, 10)
    doc["citationcount"] = numCit
    doc["influentialcitationcount"] = random.randint(0, numCit)
    s2fieldsofstudy = ["Economics", "Computer Science", "Math", "Bio"]
    doc["s2fieldsofstudy"] = s2fieldsofstudy[:random.randint(1, len(s2fieldsofstudy) - 1)]
    doc["publicationtypes"] = ["JournalArticle"]
    doc["publicationdate"] = doc["update"]
    tmp = random.randint(1, 100)
    doc["journal"] = {
        "name": doc["venue"],
        "volume": str(random.randint(1, 20)),
        "pages": f'{tmp}-{random.randint(tmp, tmp + 10)} '
    }
    doc["content"]["section"] = [gen_sec(skeleton_sec, skeleton_par, skeleton_fig) for _ in
                                 range(random.randint(1, 10))]
    doc["content"]["list_of_figures"] = [gen_img_small(skeleton_fig_small) for _ in range(random.randint(1, 10))]
    doc["citations"] = [gen_cit(skeleton_cit, random.randint(1, number_of_papers)) for _ in
                        range(random.randint(1, 10))]
    docArray.append(doc)

    with open(f'randomdocs.json', 'w') as file:
        json.dump(docArray, file, indent=4)


def gen_aut(skeleton_aut: str) -> dict:
    aut = json.loads(skeleton_aut)
    aut["authorid"] = random.randint(1000000, 9999999)
    aut["name"] = names.get_full_name()
    aut["papercount"] = random.randint(1, 50)
    aut["citationcount"] = random.randint(1, 1000)
    aut["hindex"] = random.randint(1, 100)
    aut["s2url"] = f'https://www.semanticscholar.org/author/{aut["authorid"]}'
    aut["externalids.DBLP"] = f"['{aut['name']}']"

    return aut


def gen_sec(skeleton_sec: str, skeleton_par: str, skeleton_fig: str, leaf: bool = False) -> dict:
    sec = json.loads(skeleton_sec)
    sec["section_title"] = lorem.sentence()
    numPar = random.randint(1, 10)
    sec["paragraphs"] = [gen_par(skeleton_par, skeleton_fig) for _ in range(numPar)]
    if not leaf:
        sec.update(
            {"sub_paragraphs": [[gen_sec(skeleton_par, skeleton_par, skeleton_fig, 0 == random.choice((0, 1)))]]})

    return sec


def gen_par(skeleton_par: str, skeleton_fig: str) -> dict:
    par = json.loads(skeleton_par)
    par["text"] = lorem.paragraph()
    numImg = random.randint(0, 2)
    if numImg > 0:
        par.update({"figures_in_paragraph": [gen_img(skeleton_fig) for _ in range(numImg)]})

    return par


def gen_img(skeleton_fig: str) -> dict:
    img = json.loads(skeleton_fig)
    img["start"] = random.randint(1, 1000)
    img["end"] = random.randint(img["start"], 1001)
    img["ref_id"] = f'FIGREF{random.randint(0, 100)}'
    img["caption"] = lorem.sentence()
    img["width"] = random.randint(500, 2500)
    img["height"] = random.randint(500, 2500)
    img["url"] = f'https://unsplash.com/photos/{random.randint(100000000, 999999999)}'
    img["download_url"] = f'https://picsum.photos/id/{random.randint(100000000, 999999999)}'

    return img


def gen_img_small(skeleton_img_small: str) -> dict:
    img = json.loads(skeleton_img_small)
    img["author"] = names.get_full_name()
    img["url"] = f'https://unsplash.com/photos/{random.randint(100000000, 999999999)}'
    img["fig_id"] = f'FIGREF{1, 100}'

    return img


def gen_cit(skeleton_cit: str, id: int) -> dict:
    cit = json.loads(skeleton_cit)
    cit["reference_id"] = id
    cit["title"] = lorem.sentence()
    cit["author"] = names.get_full_name()

    return cit


def mongo_db_setup():
    """Handles all the preprocessing operations for importing data in MongoDB."""
    # First of all we need to read the articles csv file and convert it in a json like structure.
    progress_bar(10, 100)
    reduce_jsonl_dataset(PAPERS_PATH_JSONL, 5000)
    progress_bar(20, 100)

    # Then we integrate some metadata about the authors
    integrate_authors_info()
    progress_bar(35, 100)

    # Merging the 5000 jsons documents representing the extracted text and properties from the pdf.
    merge_jsons()
    progress_bar(50, 100)

    # Integrating the texts in the articles objects. 
    integrate_text_info()
    progress_bar(65, 100)

    # Aggregating the text by sections.
    aggregate_sections()
    progress_bar(80, 100)

    # Integrating images metadata
    integrate_images_in_paragraphs()
    progress_bar(85, 100)

    generate_subsections()
    progress_bar(90, 100)

    # Generating citations
    generate_citations()
    progress_bar(95, 100)

    fix_dates()
    progress_bar(100, 100)


####################################################################################################
#                                            Spark                                                 #
####################################################################################################
def spark_session_handler():
    """ Handles the textual menu and the user choices for the Spark Session."""
    spark = SparkSession.builder.master("local").appName("SMBUD2").getOrCreate()
    clearScreen()
    printLogo()
    
    while(spark_operation_chosen := int(input("What operation should be performed?\n\t1. Setup dataset (N.B. : if you have already setup the dataset for Neo4J skip this step)\n\t2. Perform example queries\n\t 10. Exit\nChoice: "))):
        if spark_operation_chosen == 1:
            # Setup dataset operation
            setup_spark_dataset(spark)
        elif spark_operation_chosen == 2:
            spark_perform_queries(spark)
        elif spark_operation_chosen == 10:
            print("quit")
            break
        
        input("Press <ENTER> to continue...")
        clearScreen()
        printLogo()

def check_spark_dependencies():
    """Checks whether all the required files for spark have already been generated. Returns true if all the required files are already present in the script directory, false otherwise.

    Returns:
        bool: true if all the required files already exist in the script directory, false otherwise.
    """
    return isfile(AUTHORS_NODE_EXTENDED_PATH) and isfile(ARTICLE_FINAL_PATH) and isfile(ARTICLE_FINAL_HEADER_PATH) and isfile(PUBLISHED_BY_FINAL_PATH) and isfile(PUBLISHER_PATH) and isfile(PUBLISHED_IN_PATH) and isfile(JOURNAL_PATH)

def setup_spark_dataset(spark: SparkSession):
    """Set up the dataset files for spark if they have not been generated previosly."""

    if(check_spark_dependencies()):
        print("The dataset has been already generated. Skipping...")
    else:
        neo4jSetup()                    # Our setup for spark is built upon the Neo4J one.
    
    if not isfile(PARQUET_ARTICLE):
        print("setup articles")
        setup_articles_collection()
    if not isfile(PARQUET_JOURNAL):
        print("setup jounrals")
        setup_journals_collection(spark)
    if not isfile(PARQUET_AUTHOR):
        print("setup authors")
        setup_authors_collection()

def spark_import_procedure(spark: SparkSession) -> tuple:
    """Imports the dataset into spark dataframes."""
    setup_spark_dataset(spark)

    print("importing articles")
    df_article = spark.read.parquet(PARQUET_ARTICLE)
    df_article = df_article.withColumn("citations",col("citations").cast(ArrayType(IntegerType())))
    df_article = df_article.withColumn("incoming_citations",col("incoming_citations").cast(ArrayType(IntegerType())))
    df_article = df_article.withColumnRenamed(":ID", "ID")
    print("importing journals")
    df_journal = spark.read.parquet(PARQUET_JOURNAL)
    df_journal = df_journal.withColumn("NumArticles",col("NumArticles").cast(IntegerType()))
    print("importing authors")
    df_author = spark.read.parquet(PARQUET_AUTHOR)
    df_author = df_author.withColumn("written_articles_ids",col("written_articles_ids").cast(ArrayType(IntegerType())))
    df_author = df_author.withColumn(":ID",col(":ID").cast(IntegerType()))

    # print("start printing")
    # df_article.printSchema()
    # df_journal.printSchema()
    # df_author.printSchema()

    return df_article, df_author, df_journal


def spark_perform_queries(spark: SparkSession):
    df_articles, df_authors, df_journals = spark_import_procedure(spark)

    clearScreen()
    printLogo()
    
    while(query_selection := int(input("What query should be performed?\n\t1. Query 1\n\t2. Query 2\n\t3. Query 3\n\t4. Query 4\n\t5. Query 5\n\t6. Query 6\n\t7.Query 7\n\t8. Query 8\n\t9. Query 9\n\t10. Query 10\n\t20. Exit\nChoice: "))):
        if query_selection == 1:
            # Query 1
            perform_query_1(spark, df_authors, df_articles)
        elif query_selection == 2:
            # Query 2
            perform_query_2(spark, df_articles)
        elif query_selection == 3:
            # Query 3
            perform_query_3(spark, df_articles, df_authors)
        elif query_selection == 4:
            # Query 4
            perform_query_4(spark)
        elif query_selection == 5:
            # Query 5
            perform_query_5(spark)
        elif query_selection == 6:
            # Query 6
            perform_query_6(spark)
        elif query_selection == 7:
            # Query 7
            perform_query_7(spark)
        elif query_selection == 8:
            # Query 8
            perform_query_8(spark)
        elif query_selection == 9:
            # Query 9
            perform_query_9(spark)
        elif query_selection == 10:
            # Query 10
            perform_query_10(spark)
        elif query_selection == 20:
            print("quit")
            break
        
        input("Press <ENTER> to continue...")
        clearScreen()
        printLogo()


def setup_authors_collection():
    authors = pd.read_csv(AUTHORS_NODE_EXTENDED_PATH, sep=";")
    articles = pd.read_csv(ARTICLE_FINAL_PATH, sep=";", low_memory=False, header=None)
    articles = pd.DataFrame({':START_ID' : articles.iloc[:, 0]})
    authored_by = pd.read_csv(AUTHORED_BY_FINAL_PATH, sep=";")

    authors['affiliations'] = authors['affiliations'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    authors['externalids.ORCID'] = authors['externalids.ORCID'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    authors['externalids.ORCID'] = authors['externalids.ORCID'].str[0]
    authors['hindex'] = authors['hindex'].fillna(-1)
    authors['hindex'] = authors['hindex'].astype('int32')
    authors['papercount'] = authors['papercount'].fillna(-1)
    authors['papercount'] = authors['papercount'].astype('int32')
    authors['citationcount'] = authors['citationcount'].fillna(-1)
    authors['citationcount'] = authors['citationcount'].astype('int32')

    joined_df = pd.merge(articles, authored_by, on = ":START_ID")
    joined_df = joined_df.groupby(':END_ID')[":START_ID"].apply(list).reset_index(name='written_articles_ids')
    joined_df.rename(columns={':END_ID': ':ID'}, inplace=True)

    authors = pd.merge(authors, joined_df, on=":ID", how="left")
    authors["written_articles_ids"] = authors['written_articles_ids'].apply(lambda x: x if isinstance(x, list) else [])

    authors.to_parquet(PARQUET_AUTHOR, compression=None)

def setup_articles_collection():
    # serve a inserire l'header in article.csv
    if not isfile(ARTICLE_FINAL_PATH_EXTENDED):
        with open(ARTICLE_FINAL_HEADER_PATH) as header_file:
            header_list = list(csv.reader(header_file, delimiter=";"))
            header_list = header_list[0]
        for i in range(len(header_list)):
            header_list[i] = header_list[i].split(":")
            header_list[i] = header_list[i][0]
        header_list.insert(0,":ID")
        del header_list[1]
        articles = pd.read_csv(ARTICLE_FINAL_PATH, sep=";", low_memory=False)
        articles.to_csv(ARTICLE_FINAL_PATH_EXTENDED, header=header_list, sep=";", index=False)
    
    articles = pd.read_csv(ARTICLE_FINAL_PATH_EXTENDED, sep=";", low_memory=False)
    articles.drop(['cdate', 'note-type'], inplace=True, axis=1)
    articles[":ID"] = articles[":ID"].astype("int32")
    articles['ee'] = articles['ee'].apply(lambda x: x.split("|") if pd.notna(x) else None)
    articles['ee-type'] = articles['ee-type'].apply(lambda x: x.split("|") if pd.notna(x) else None)
    articles['mdate'] = articles['mdate'].apply(lambda x: pd.to_datetime(x).date() if pd.notna(x) else None)
    articles['note'] = articles['note'].apply(lambda x: x.split("|") if pd.notna(x) else None)
    articles["url"] = articles["url"].apply(lambda x: x if pd.notna(x) else "")
    articles["url"] = articles["url"].apply(lambda x: x.split("|"))
    articles['year'] = articles['year'].fillna(-1)
    articles['year'] = articles['year'].astype('int32')

    authored_by = pd.read_csv(AUTHORED_BY_FINAL_PATH, sep=";")
    authored_by = authored_by.groupby(':START_ID')[":END_ID"].apply(list).reset_index(name='authors_ids')
    authored_by.rename(columns={':START_ID': ':ID'}, inplace=True)

    articles = pd.merge(articles, authored_by, on=":ID", how="left")
    articles["authors_ids"] = articles['authors_ids'].apply(lambda x: x if isinstance(x, list) else None)

    published_in = pd.read_csv(PUBLISHED_IN_PATH, sep=';', low_memory=False)
    published_in[':END_ID'] = published_in[':END_ID'].fillna(-1)
    published_in[':END_ID'] = published_in[':END_ID'].astype('int32')
    published_in.rename(columns={':START_ID': ':ID', ':END_ID': 'journal_id'}, inplace=True)
    articles = pd.merge(articles, published_in, on=":ID", how="left")
    articles['journal_id'] = articles['journal_id'].apply(lambda x: int(x) if pd.notna(x) else None)

    cite = pd.read_csv(CITE_RELATIONSHIP, sep=';', low_memory=False)
    cite['START_ID'] = cite[':START_ID'].fillna(-1)
    cite[':START_ID'] = cite[':START_ID'].astype('int32')
    cite[':END_ID'] = cite[':END_ID'].fillna(-1)
    cite[':END_ID'] = cite[':END_ID'].astype('int32')
    cite[':START_ID'] = cite[':START_ID'].apply(lambda x: int(x))
    cite[':END_ID'] = cite[':END_ID'].apply(lambda x: int(x))
    citations = cite.groupby(':START_ID')[":END_ID"].apply(list).reset_index(name='citations')
    citations.rename(columns={':START_ID': ':ID'}, inplace=True)
    articles = pd.merge(articles, citations, on=":ID", how="left")
    incoming_citations = cite.groupby(':END_ID')[":START_ID"].apply(list).reset_index(name='incoming_citations')
    incoming_citations.rename(columns={':END_ID': ':ID'}, inplace=True)
    articles = pd.merge(articles, incoming_citations, on=":ID", how="left")

    articles.to_parquet(PARQUET_ARTICLE, compression=None)
    

def setup_journals_collection(spark: SparkSession):
    df_journal = spark.read.csv(JOURNAL_PATH, sep=";", header=True, inferSchema=True)
    df_journal = df_journal.withColumnRenamed(":ID","ISSN").withColumnRenamed("journal:string","name")

    df_journal_published_in = spark.read.csv(PUBLISHED_IN_PATH, sep=";", header=True, inferSchema=True)
    df_journal_published_in = df_journal_published_in.withColumnRenamed(":START_ID","START_ID").withColumnRenamed(":END_ID","END_ID")
    
    set_journal = df_journal_published_in.select(collect_set("END_ID")).collect()[0][0]
    
    dataTmp = {}
    csv_file = pd.read_csv(PUBLISHED_IN_PATH, sep=";", low_memory=False)

    for i in range(1, len(csv_file.index)):
        if csv_file.loc[i, ':END_ID'] in dataTmp.keys():
            dataTmp[csv_file.loc[i, ':END_ID']].append(int(csv_file.loc[i, ':START_ID']))
        else:
            dataTmp[csv_file.loc[i, ':END_ID']] = [(int(csv_file.loc[i, ':START_ID']))]
    
    data = []
    columns = StructType([
                StructField("END_ID", IntegerType(), False),
                StructField("NumArticles", IntegerType(), True),
                StructField("Articles", ArrayType(IntegerType()), True)
            ])

    for el in set_journal:
        data.append([el, len(dataTmp[el]), dataTmp[el]])

    df_tmp = spark.createDataFrame(data = data, schema = columns)

    df_journal_final = df_journal.join(df_tmp, df_journal.ISSN == df_tmp.END_ID, "left").drop(df_tmp.END_ID)

    # https://learn.microsoft.com/en-us/azure/databricks/pandas/pyspark-pandas-conversion
    # https://stackoverflow.com/questions/70922875/how-to-convert-a-very-large-pyspark-dataframe-into-pandas
    # https://stackoverflow.com/questions/47536123/collect-or-topandas-on-a-large-dataframe-in-pyspark-emr
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")

    esult_pdf = df_journal_final.select("*").toPandas()
    esult_pdf.to_parquet(PARQUET_JOURNAL, compression=None)

####################################################################################################
#                                         SPARK QUERIES                                            #
####################################################################################################
def perform_query_1(spark: SparkSession, df_authors: ps.DataFrame, df_articles: ps.DataFrame):
    # QUERY 1 KO
    exploded_df_authors = df_authors.select(df_authors.name, explode(df_authors.written_articles_ids)).withColumnRenamed("col", "article_id")
    # exploded_df_authors.show()
    exploded_df_authors.filter(exploded_df_authors.name.like("%S. Ceri%")).join(df_articles, exploded_df_authors.article_id == df_articles.ID).show()

def perform_query_2(spark: SparkSession, df_articles: ps.DataFrame):
    """Fetch the articles which have Machine Learning in their title.

    Args:
        spark (SparkSession): current spark session.
        df_articles (ps.DataFrame): dataframe containing articles data.
    """
    # QUERY 2 OK
    df_articles.filter(df_articles.title.like("%Machine Learning%")).limit(5).show()
    
def perform_query_3(spark: SparkSession, df_articles: ps.DataFrame, df_authors: ps.DataFrame):
    # QUERY 3 KO

    """
    SELCT *
    FROM Article
    WHERE Article.authors IN {
    SELECT Authors
    FROM Authors
    WHERE Authos.citationcount > 10
    }    
    """
    average_citationcount = (df_authors.select(max("citationcount")) + df_authors.select(min("citationcount"))) / 2
    df_articles.isin(flatten(concat(df_authors.filter(df_authors.citationcount > average_citationcount).select("written_articles_ids"))))
    
    
def perform_query_4(spark: SparkSession, df_authors: ps.DataFrame):
    """Group the authors by affiliations and get the sum of all citations.

    Args:
        spark (SparkSession): current spark session.
        df_authors (ps.DataFrame): dataframe containing authors data.
    """
    # QUERY 4 OK

    exploded_df_authors = df_authors.select(df_authors.citationcount,explode(df_authors.affiliations))
    exploded_df_authors = exploded_df_authors.withColumnRenamed("col", "affiliation")
    exploded_df_authors.groupBy("affiliation").agg(sum("citationcount").alias("Number of citations")).show(truncate = False)

def perform_query_5(spark: SparkSession):
    """"
    df_articles.filter(size(df_articles.incoming_citation > 20)).groupBy(df_articles.year).agg(count("*"))
    """
    # QUERY 5
    print("ciao")
    
def perform_query_6(spark: SparkSession, df_articles: ps.DataFrame): 
    """ Return all the year in which there are at least 20 articles.
    Args:
        spark (SparkSession): current spark session.
        df_articles (ps.DataFrame): dataframe containing articles data.
    """
    # QUERY 6 OK
    df_articles.groupBy("year").count().filter("count >20").show(truncate = False)


def perform_query_7(spark: SparkSession):
    """
    QUERY 7
    df_articles.filter(size(df_articles.incoming_citation > 10)).groupBy(df_articles.year).agg(count("*").alias('count')).where(col("count") > 20)
    """
    print("ciao")
def perform_query_8(spark: SparkSession, df_authors: ps.DataFrame, df_articles: ps.DataFrame):
    """
    QUERY 8 NI
    i-10 index o g-index dovrebbero andare bene ???????????????????????????????????????
    
    import org.apache.spark.sql.functions.{size,col}
    
    """
    # filtro su un autore
    authors = [9474010, 12173054, 9647065]
    df_authors = df_authors.filter(df_authors.name=="F. Olken")
    # esplodo l'array con gli ID delle pubblicazioni di un array
    exploded_df_authors = df_authors.select(df_authors.name, explode(df_authors.written_articles_ids))
    # rinonino la colonna esplosa
    exploded_df_authors = exploded_df_authors.withColumnRenamed("col", "publication_ID")
    # join tra autore e articolo
    exploded_df_articles = df_articles.select(col("ID"), explode(df_articles.incoming_citations))
    exploded_df_articles = exploded_df_articles.groupby(col("ID")).count().withColumnRenamed("count", "num_of_incoming_citations")
    # filtro le riche con articoli che hanno meno di 10 citazioni
    exploded_df_articles = exploded_df_articles.filter("num_of_incoming_citations > 10")
    exploded_df_authors = exploded_df_authors.limit(4).join(exploded_df_articles, exploded_df_authors.publication_ID == df_articles.ID, "outer").distinct()

    # rappruppo per il nome dell'autore e conto le righe
    exploded_df_authors.groupBy(df_authors.name).count().withColumnRenamed( "count", "I-10 index").show()
    

def perform_query_9(spark: SparkSession):
    """
    QUERY 9
    authors = ["auth1", "auth2"]
    authors_with_pub_citation = df_authors.filter(df_authors.name.isin(authors)).join(df_articles, expr("array_contains(df_authors.publications, df_articles.ID)"))\
        .groupBy(df_authors.ID).collect().alias("citations")
    authors_with_added_col = authors_with_pub_citation.withColumn("iter", sequence(max(authors_with_pub_citation.citations))
    authors_with_added_col.select(authors_with_added_col.ID, authors_with_added_col.citations, explode(authors_with_added_col.iter))\
        .filter(size([x for x in authors_with_added_col.citations if x > authors_with_added_col.iter]) >= authors_with_added_col.iter)
    """
    print("ciao")
def perform_query_10(spark: SparkSession):
    """
    QUERY 10
    from pyspark.sql.functions import explode
//esplodo le pubblicaziondi di un autore e rinomino la colonna
exploded_df_authors = df_authors.select(df_authors.name,affiliations,explode(df_au
thors.publications))
exploded_df_authors = exploded_df_authors.withColumnRenamed("col", "publication_I
D")
//esplodo le affiliazioni di un autore e rinomino la colonna
exploded_df_authors = df_authors.select(df_authors.name,publication_ID,explode(aff
iliations))
exploded_df_authors = exploded_df_authors.withColumnRenamed("col", "affiliation")
//join tra articoli e journal
df_articles.join(df_journals, df_articles.Journal == df_journals.ID, "outer")
//join tra autori e articoli
exploded_df_authors.join(df_articles, exploded_df_authors.publication_ID == df_art
icles.ID, "outer")
exploded_df_authors.groupBy("affiliation","jounal").count()
    """
    print("ciao")


####################################################################################################
#                                          GRAPHICS                                                #
####################################################################################################

def printLogo():
    """Prints the logo onto the console."""

    print("""
        ####################################################################################################
        #                    ___                                                                           #
        #                  //    \                                                                         #
        #                 ||=====||                                                                        #
        #                  \ ___//                                                                         #
        #                   ./O           System and Methods for Big and Unstructured Data - Group 2       #
        #               ___/ //|\                                                                          #
        #              / o    /}                           Riccardo Inghilleri                             #
        #             (       /                               Manuela Merlo                                #
        #             \      /                                 Matteo Negro                                #
        #             |     (                                 Paolo Pertino                                #
        #             |      \                               Leonardo Pesce                                #
        #             )       \                                                                            #
        #            /         \                                                                           #
        #          /            )                                                                          #
        #        /              |                                                                          #
        #      //             / /                                                                          #
        #    /       ___(    ,| \                                                                          #
        #  /       /    \     |  \                                                                         #
        # (      /  /   /\     \  \                                                                        #
        # \   /___ _-_//'|     |  |                                                                        #
        #  \_______-/     \     \  \                                                                       #
        #                   \-_-_-_-_-                                                                     #
        ####################################################################################################
    """)

def clearScreen():
    """Clears the console"""

    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def progress_bar(progress : int, total : int, custom_string: str = ""):
    """Creates/update a console and its status.

    Args:
        progress (int): current progress of the progress bar
        total (int): the number corresponding at the 100% of completion
        custom_string (str, optional): optional log message. Defaults to "".
    """
    percent = int(100 * (progress / total))
    bar = '▮' * percent + '-' * (100 - percent)
    
    print(f"\r|{bar}| {percent :.2f}%", end = "\r")

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

def main():
    # Init
    clearScreen()
    printLogo()

    # Main menu
    while((operation_chosen := int(input("What operation do you want to perform (N.B.: db reduction cannot be performed if the data are not previously cleaned)?\n\t1. Neo4J Setup\n\t2. MongoDB Setup\n\t3. MongoDB Random Setup\n\t4. Spark\n\t10. Exit\n Choice: "))) != 10):
        clearScreen()

        # Operation chosen == 1 means that the user wants to clean his data in order to import them in Neo4J.
        if(operation_chosen == 1):
            printLogo()
            progress_bar(0, 100)
            neo4jSetup()
        # Operation chosen == 2 means that the user wants to structure his data in order to import them in MongoDB.
        elif(operation_chosen == 2):
            printLogo()
            progress_bar(0, 100)
            mongo_db_setup()
        elif(operation_chosen == 3):
            printLogo()
            mongo_db_setup_random(int(input('How many papers do you want to generate? (default 5000) : ')))
        elif(operation_chosen == 4):
            printLogo()
            spark_session_handler()
        elif(operation_chosen == 10):
            print("Ending...")
            break
        
        input("Press <ENTER> to continue...")
        clearScreen()

def test():
    spark = SparkSession.builder.master("local").appName("SMBUD2").getOrCreate()
    df_article = spark.read.parquet(PARQUET_ARTICLE)
    df_article = df_article.withColumn("citations",col("citations").cast(ArrayType(IntegerType())))
    df_article = df_article.withColumn("incoming_citations",col("incoming_citations").cast(ArrayType(IntegerType())))
    df_article = df_article.withColumnRenamed(":ID", "ID")
    print("importing journals")
    df_journal = spark.read.parquet(PARQUET_JOURNAL)
    df_journal = df_journal.withColumn("NumArticles",col("NumArticles").cast(IntegerType()))
    print("importing authors")
    df_author = spark.read.parquet(PARQUET_AUTHOR)
    df_author = df_author.withColumn("written_articles_ids",col("written_articles_ids").cast(ArrayType(IntegerType())))
    df_author = df_author.withColumn(":ID",col(":ID").cast(IntegerType()))
    # df_author.printSchema()
    perform_query_8(spark, df_author, df_article)

if __name__ == "__main__":
    # main()
    test()