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
import datetime
import glob
import random
import requests
import names
import lorem
import time

from random_object_id import generate
from os import system, name
from os.path import isfile
from ast import literal_eval
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType, IntegerType
from pyspark import pandas as ps

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
    clearScreen()
    printLogo()
    
    while(spark_operation_chosen := int(input("What operation should be performed?\n\t1. Setup dataset (N.B. : if you have already setup the dataset for Neo4J skip this step)\n\t2. Import dataset into Spark (if the dataset has not been created it will automatically perform the previous action)\n\t3. Perform example queries\n\t 10. Exit\nChoice: "))):
        if spark_operation_chosen == 1:
            # Setup dataset operation
            setup_spark_dataset()
        elif spark_operation_chosen == 2:
            # Import dataset into spark
            spark_import_procedure()
        elif spark_operation_chosen == 3:
            print("perform queries")
        elif spark_operation_chosen == 10:
            print("quit")
        
        time.sleep(2)
        clearScreen()
        printLogo()

def check_spark_dependencies():
    """Checks whether all the required files for spark have already been generated. Returns true if all the required files are already present in the script directory, false otherwise.

    Returns:
        bool: true if all the required files already exist in the script directory, false otherwise.
    """
    return isfile(AUTHORS_NODE_EXTENDED_PATH) and isfile(ARTICLE_FINAL_PATH) and isfile(ARTICLE_FINAL_HEADER_PATH) and isfile(PUBLISHED_BY_FINAL_PATH) and isfile(PUBLISHER_PATH) and isfile(PUBLISHED_IN_PATH) and isfile(JOURNAL_PATH)

def setup_spark_dataset():
    """Set up the dataset files for spark if they have not been generated previosly."""

    if(check_spark_dependencies()):
        print("The dataset has been already generated. Skipping...")
    else:
        neo4jSetup()                    # Our setup for spark is built upon the Neo4J one.
                                        # TODO: maybe other operations are necessary.

def spark_import_procedure():
    """Imports the dataset into spark."""

    # If the dataset hasn't already been setup, it cannot be imported. 
    if(not check_spark_dependencies()):
        print("The dataset has not been generated yet. The setup procedure will be called...")
        setup_spark_dataset()

def import_authors_collection():
    authors = pd.read_csv(AUTHORS_NODE_EXTENDED_PATH, sep=";")
    authors['affiliations'] = authors['affiliations'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    authors['externalids.ORCID'] = authors['externalids.ORCID'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    authors['externalids.ORCID'] = authors['externalids.ORCID'].str[0]
    authors['hindex'] = authors['hindex'].fillna(-1)
    authors['hindex'] = authors['hindex'].astype('int32')
    authors['papercount'] = authors['papercount'].fillna(-1)
    authors['papercount'] = authors['papercount'].astype('int32')
    authors['citationcount'] = authors['citationcount'].fillna(-1)
    authors['citationcount'] = authors['citationcount'].astype('int32')
    
    spark = SparkSession.builder.getOrCreate()
    schema = StructType([ \
        StructField(":ID", IntegerType(), False), \
        StructField("name", StringType(), False), \
        StructField("affiliations", ArrayType(StringType()), True), \
        StructField("homepage", StringType(), True), \
        StructField("papercount", IntegerType(), True), \
        StructField("citationcount", IntegerType(), True), \
        StructField("hindex", IntegerType(), True), \
        StructField("url", StringType(), True), \
        StructField("ORCID", StringType(), True), \
    ])

    # Creating a dataframe starting from a predefined schema.
    authors_df = spark.createDataFrame(data = authors, schema = schema)
    # authors_df.printSchema()
    # df.explain()
    authors_df.show()

def import_articles_collection():
    articles = pd.read_csv(ARTICLE_FINAL_PATH, sep=";")
    articles['authors'] = articles['authors'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    articles['ee'] = articles['ee'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    articles['ee-type'] = articles['ee-type'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    articles['note'] = articles['note'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    articles['note-type'] = articles['note-type'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)
    articles['url'] = articles['url'].apply(lambda x: literal_eval(x) if pd.notna(x) else None)

    spark = SparkSession.builder.getOrCreate()
    
    """
    schema = StructType([ 
        StructField("ID", IntegerType(), False), 
        StructField("authors", ArrayType(IntegerType()), True), 
        StructField("journal", IntegerType(), True), 
        StructField("cdate", DateType(), True), 
        StructField("ee", ArrayType(StringType()), True), 
        StructField("ee-type", ArrayType(StringType()), True), 
        StructField("key",StringType(), True), 
        StructField("mdate", DateType(), True), 
        StructField("month", StringType(), True), 
        StructField("note", ArrayType(StringType()), True), 
        StructField("note-label", StringType(), True), 
        StructField("note-type", ArrayType(StringType()), True), 
        StructField("publtype", StringType(), True), 
        StructField("title", StringType(), True), 
        StructField("url", ArrayType(StringType()), True), 
        StructField("year", IntegerType(), True), 
    ])
    """
    # df = spark.read.csv(ARTICLE_FINAL_PATH, sep=";", header=True, inferSchema=True)
    # df = spark.createDataFrame(data = articles, schema = schema)
    # df.printSchema()
    # df.show(truncate=False)
    df = spark.read.csv(ARTICLE_FINAL_PATH, sep=";", header=True, inferSchema=True)
    df.printSchema()
    df.explain()
    df.show(truncate=False)
    

def import_journals_collection():
    spark = SparkSession.builder.getOrCreate()
    """
    schema_journal = StructType([ 
        StructField(":ID", IntegerType(), False),
        StructField("journal", StringType(), True),
    ])
    """
    df_journal = spark.read.csv(JOURNAL_PATH, sep=";", header=True, inferSchema=True)

    """
    schema_journal_published_in = StructType([ 
        StructField(":START_ID", IntegerType(), False),
        StructField(":END_ID", IntegerType(), False),
        StructField("number", IntegerType(), True),
        StructField("pages", StringType(), True),
        StructField("volume", StringType(),True),
    ])
    """
    df_journal_published_in = spark.read.csv(PUBLISHED_IN_PATH, sep=";", header=True, inferSchema=True)

    """
    schema_journal_published_by = StructType([ 
        StructField(":START_ID", IntegerType(), False),
        StructField(":END_ID", IntegerType(), False),
    ])
    """
    df_journal_published_by = spark.read.csv(PUBLISHED_BY_FINAL_PATH, sep=";", header=True, inferSchema=True)

    """
    schema_publisher = StructType([ 
        StructField(":ID", IntegerType(), False),
        StructField("publisher", StringType(), True),
    ])
    """
    df_publisher = spark.read.csv(PUBLISHER_PATH, sep=";", header=True, inferSchema=True)

    # Print detected 
    # We can see info about datatypes we uploaded in the db.
    df_journal_published_in.printSchema()
    df_journal_published_in.explain()
    df_journal_published_in.show(truncate=False)


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
        
        time.sleep(2)
        clearScreen()


def test():
    # import_journals_collection()
    import_articles_collection()
    
        
if __name__ == "__main__":
    # main()
    test()