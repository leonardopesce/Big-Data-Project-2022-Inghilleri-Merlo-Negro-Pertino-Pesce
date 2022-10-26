import pandas as pd

BOOK_PATH = "output_book.csv"
INCOLLECTION_PATH = "output_incollection.csv"
INPROCEEDINGS_PATH = "output_inproceedings.csv"
PROCEEDINGS_PATH = "output_proceedings.csv"
WWW_PATH = "output_www.csv"
ARTICLE_PATH = "output_article.csv"
HAS_CITATIONS_PATH = "output_cite_has_citation.csv"
CITE_PATH = "output_cite.csv"

def log(log_message: str):
    print(f'--- {log_message} ---')

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