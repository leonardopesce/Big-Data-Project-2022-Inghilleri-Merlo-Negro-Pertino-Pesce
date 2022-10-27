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

from ast import literal_eval

BOOK_PATH = "output_book.csv"
INCOLLECTION_PATH = "output_incollection.csv"
INPROCEEDINGS_PATH = "output_inproceedings.csv"
PROCEEDINGS_PATH = "output_proceedings.csv"
WWW_PATH = "output_www.csv"
ARTICLE_PATH = "output_article.csv"
HAS_CITATIONS_PATH = "output_cite_has_citation.csv"
CITE_PATH = "output_cite.csv"
AUTHORS_DATA_PATH = "authors.csv"
AUTHORS_NODE_PATH = "output_author.csv"

def log(log_message: str):
    print(f'--- {log_message} ---')

def authors_additional_information():
    log("STARTED LOADING DETAILED AUTHORS DATA")
    author_data = pd.read_csv(AUTHORS_DATA_PATH, sep="\t", engine='python')
    log("FINISHED LOADING DETAILED AUTHORS DATA")
    log("STARTED LOADING AUTHORS NODES")
    author_node = pd.read_csv(AUTHORS_NODE_PATH, sep=";")
    log("FINISHED LOADING AUTHORS NODES")

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
    merged_frame = pd.merge(author_node, n_author_data, on = 'key')
    merged_frame.drop(["key"], axis=1, inplace=True)
    merged_frame.drop_duplicates([':ID'], inplace=True)

    author_node.rename(columns={'key' : 'name'}, inplace=True)
    final_frame = pd.concat([merged_frame, author_node])
    final_frame.drop_duplicates([':ID'], keep='first', inplace=True)
    log("FINISHED INTEGRATING AUTHORS INFORMATION")

    return final_frame

def reduce_frame(path: str, first_column_label: str, first_column_index : int, second_column_label : str, second_column_index : int) -> pd.DataFrame:
    log("STARTING LOADING DATA")
    frame = pd.read_csv(path, sep = ";", low_memory=False)
    log("FINISHED LOADING DATA")
    reduced_frame = pd.DataFrame({f"{first_column_label}" : frame.iloc[:, first_column_index], f"{second_column_label}" : frame.iloc[:, second_column_index]})
    return reduced_frame

def merge_cite_nodes() -> pd.DataFrame:
    log("STARTING CITE NODE MERGE")
    log("STARTING LOADING REQUIRED FILES")
    citation_frame = pd.read_csv(CITE_PATH, sep=";")
    has_citation_frame = pd.read_csv(HAS_CITATIONS_PATH, sep=";")
    log("FINISHED LOADING REQUIRED FILES")

    citation_frame.rename(columns={':ID': ':END_ID'}, inplace=True)
    merged_frame = pd.merge(has_citation_frame, citation_frame)
    merged_frame.rename(columns={'cite:string': 'key'}, inplace=True)
    log("REMOVING USELESS CITATIONS")
    for index, row in merged_frame.iterrows():
        if(index % 1000 == 0):
            print(f'--- Iteration: {index} ---')
        if row['key'] == '...':
            merged_frame.drop([index], inplace=True)

    merged_frame = pd.DataFrame(
        {':START_ID': merged_frame.iloc[:, 0], 'key': merged_frame.iloc[:, 2]})

    merged_frame.to_csv("output_cite_merged.csv", sep=";", index=False)
    log("PARTIAL MERGE ENDED. FILE STORED AS output_cite_merged.csv")
    return merged_frame

def concat_all_publication_frames(reduced_article_frame : pd.DataFrame, reduced_book_frame : pd.DataFrame, reduced_incollection_frame : pd.DataFrame, reduced_inproceedings_frame : pd.DataFrame, reduced_proceedings_frame : pd.DataFrame, reduced_www_frame : pd.DataFrame) -> pd.DataFrame:
    log("STARTED CONCATENATING FRAMES")
    
    return pd.concat([reduced_article_frame, reduced_book_frame, reduced_incollection_frame, reduced_inproceedings_frame, reduced_proceedings_frame, reduced_www_frame])

def compile_relationship(frame_1 : pd.DataFrame, frame_2 : pd.DataFrame) -> pd.DataFrame:
    merged_frame = pd.merge(frame_1, frame_2)
    merged_frame = pd.DataFrame({':START_ID': merged_frame.iloc[:, 0], ':END_ID' : merged_frame.iloc[:, 2]})
    return merged_frame

def main():
    log("STARTING REDUCING ARTICLE FRAME")
    article_frame = reduce_frame(ARTICLE_PATH, "ArticleId", 0, "key", 16)
    log("FINISHED REDUCING ARTICLE FRAME")
    log("STARTING REDUCING BOOK FRAME")
    book_frame = reduce_frame(BOOK_PATH, "ArticleId", 0, "key", 16)
    log("FINISHED REDUCING BOOK FRAME")
    log("STARTING REDUCING INCOLLECTION FRAME")
    incollection_frame = reduce_frame(INCOLLECTION_PATH, "ArticleId", 0, "key", 12)
    log("FINISHED REDUCING INCOLLECTION FRAME")
    log("STARTING REDUCING INPROCEEDINGS FRAME")
    inproceedings_frame = reduce_frame(INPROCEEDINGS_PATH, "ArticleId", 0, "key", 14)
    log("FINISHED REDUCING INPROCEEDINGS FRAME")
    log("STARTING REDUCING PROCEEDINGS FRAME")
    proceedings_frame = reduce_frame(PROCEEDINGS_PATH, "ArticleId", 0, "key", 14)
    log("FINISHED REDUCING PROCEEDINGS FRAME")
    log("STARTING REDUCING HOMEPAGES FRAME")
    www_frame = reduce_frame(WWW_PATH, "ArticleId", 0, "key", 7)
    log("FINISHED REDUCING HOMEPAGES FRAME")

    concatenated_publications_frame = concat_all_publication_frames(article_frame, book_frame, incollection_frame, inproceedings_frame, proceedings_frame, www_frame)
    log("FINISHED CONCATENATING FRAMES")

    authors_data = authors_additional_information()
    authors_data.to_csv("output_author_extended.csv", sep=";", index=False)

    log("STARTED BUILDING CITE AND BELONG RELATIONSHIPS")
    cite_relationship = compile_relationship(merge_cite_nodes(), concatenated_publications_frame)
    belong_relationship = compile_relationship(reduce_frame(INPROCEEDINGS_PATH, "DocIdA", 0, "key", 8), reduce_frame(PROCEEDINGS_PATH, "DocIdB", 0, "key", 14))
    cite_relationship.to_csv("output_cite_relationship.csv", sep=";",index=False)
    belong_relationship.to_csv("output_belong_relationship.csv", sep=";", index=False)
    log("CITE RELATIONSHIP SAVED INTO output_cite_relationship.csv.")
    log("BELONG RELATIONSHIP SAVED INTO output_belong_relationship.csv.")
    log("ENDING")

if __name__ == "__main__":
    main()