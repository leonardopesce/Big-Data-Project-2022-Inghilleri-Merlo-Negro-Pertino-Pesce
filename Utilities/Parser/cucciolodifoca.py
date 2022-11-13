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

from os import system, name

from ast import literal_eval

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
PAPERS_PATH_JSONL = "papers.jsonl"
REDUCED_PAPERS_PATH_JSONL = "reduced_papers.jsonl"
PAPERS_PATH_CSV = "papers.csv"
PAPERS_TEXT_MERGED_PATH = "papers_text.json"
PAPERS_FINAL_PATH = "papers_full_info.json"

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
    authoredby_frame.to_csv(AUTHORED_BY_PATH[:len(AUTHORED_BY_PATH)-4] + "_new.csv", sep=";", index=False)

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
    publishedby_frame.to_csv(PUBLISHED_BY_PATH[:len(PUBLISHED_BY_PATH)-4] + "_new.csv", sep=";", index=False)

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
    editedby_frame.to_csv(EDITED_BY_PATH[:len(EDITED_BY_PATH)-4] + "_new.csv", sep=";", index=False)

def clean_fields(path: str, field_idx: list):
    """Given a path of a dataset csv file and a list of indexes of the corresponding dataframe to drop, then it returns a dataframe with all those columns removed.

    Args:
        path (str): path of the csv file corresponding to a dataset.
        field_idx (list): list of indexes of the columns of the corresponding dataframe to drop.
    """
    frame = pd.read_csv(path, sep = ";", low_memory=False, header=None)

    # Dropping all the columns corresponding to the list of indexes passed.
    frame.drop(frame.columns[field_idx], axis=1, inplace=True)
    if(path == ARTICLE_PATH):
        frame.iloc[:,13] = frame.iloc[:, 13].astype('Int64')
    frame.to_csv(path[:len(path)-4] + '_new.csv', sep = ";", index=False, header=False)

def clean_fields_header(path: str, field_idx: list):
    """Given a path of a dataset headers csv file and a list of indexes of the corresponding dataframe to drop, then it returns a dataframe with all those columns removed.

    Args:
        path (str): path of the csv file containing the headers of a dataframe.
        field_idx (list): list of indexes of the columns of the corresponding headers to drop.
    """
    frame = pd.read_csv(path, sep = ";", low_memory=False)
    frame.drop(frame.columns[field_idx], axis=1, inplace=True)
    frame.to_csv(path[:len(path)-4] + '_new.csv', sep = ";", index=False)

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
    authors_data.to_csv("output_author_extended.csv", sep=";", index=False)

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
                author_id = author.get('authorId')
                if(author_id is not None):
                    author_additional_data = authors_frame.loc[authors_frame['authorid'] == int(author_id)]
                    author_additional_data.dropna(axis=1, inplace=True)
                    author_additional_data = author_additional_data.to_dict(orient='records')
                    if(len(author_additional_data) > 0):
                        author_additional_data = author_additional_data[0]
                        author_additional_data.pop('updated', None)
                        author_additional_data.pop('aliases', None)
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
    """Inserts in each article object some content (paragraphs, text, figures...)"""
    new_articles = []
    with open(PAPERS_TEXT_MERGED_PATH, 'r') as articles_text_file, jsonlines.open(REDUCED_PAPERS_PATH_JSONL, 'r') as articles_notext_file:
        articles_text = json.load(articles_text_file)

        for i, article in enumerate(articles_notext_file):
            new_article = article
            new_article.update({
                "content" : {
                    "sections" : articles_text[i].get("content"),
                    "list_of_figures" : articles_text[i].get("list_of_figures")
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

            # for
        
        aggregated_papers = papers

    with open(PAPERS_FINAL_PATH, "w") as papers_file:
        json.dump(aggregated_papers, papers_file, indent = 4)  

def remove_null_values(d: dict) -> dict:
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

def mongodbSetup():
    """Handles all the preprocessing operations for importing data in MongoDB."""
    # First of all we need to read the articles csv file and convert it in a json like structure.
    reduce_jsonl_dataset(PAPERS_PATH_JSONL, 5000)

    # Then we integrate some metadata about the authors
    integrate_authors_info()

    # Merging the 5000 jsons documents representing the extracted text and properties from the pdf.
    merge_jsons()

    # Integrating the texts in the articles objects. 
    integrate_text_info()

    # Aggregating the text by sections.
    aggregate_sections()

    #TODO: remove null values from all the objects.
    #TODO: make a function by using pymongo to upload the documents in the database.

    
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
    clearScreen()
    percent = int(100 * (progress / total))
    bar = '▮' * percent + '-' * (100 - percent)
    
    print(f"\r|{bar}| {percent :.2f}% - [LOG] {custom_string}", end = "\r")

####################################################################################################
#                                              MAIN                                                #
####################################################################################################

def main():
    # Init
    clearScreen()
    printLogo()

    # Main menu
    while((operation_chosen := int(input("What operation do you want to perform (N.B.: db reduction cannot be performed if the data are not previously cleaned)?\n\t1. Neo4J Setup\n\t2. MongoDB Setup\n\t10. Exit\n Choice: "))) != 10):
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
            mongodbSetup()
        
        clearScreen()

def test():
    mongodbSetup()
    

if __name__ == "__main__":
    main()
    # test()