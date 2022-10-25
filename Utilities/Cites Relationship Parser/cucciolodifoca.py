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

def reduce_article_frame() -> pd.DataFrame:
    log("STARTING REDUCING ARTICLE FRAME")
    log("STARTING LOADING ARTICLE DATA")
    article_frame = pd.read_csv(ARTICLE_PATH, sep=";", low_memory=False)
    log("FINISHED LOADING ARTICLE DATA")
    reduced_article_frame = pd.DataFrame({'ArticleId': article_frame.iloc[:, 0], 'key': article_frame.iloc[:, 16]})
    log("FINISHED REDUCING ARTICLE FRAME")
    return reduced_article_frame

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

def reduce_book_frame() -> pd.DataFrame:
    log("STARTING REDUCING BOOK FRAME")
    log("STARTING LOADING BOOK DATA")
    book_frame = pd.read_csv(BOOK_PATH, sep=";", low_memory=False)
    log("FINISHED LOADING BOOK DATA")
    reduced_book_frame = pd.DataFrame({'ArticleId': book_frame.iloc[:, 0], 'key': book_frame.iloc[:, 16]})
    log("FINISHED REDUCING BOOK FRAME")
    return reduced_book_frame

def reduce_incollection_frame() -> pd.DataFrame:
    log("STARTING REDUCING INCOLLECTION FRAME")
    log("STARTING LOADING INCOLLECTION DATA")
    incollection_frame = pd.read_csv(INCOLLECTION_PATH, sep=";", low_memory=False)
    log("FINISHED LOADING INCOLLECTION DATA")
    reduced_incollection_frame = pd.DataFrame({'ArticleId': incollection_frame.iloc[:, 0], 'key': incollection_frame.iloc[:, 12]})
    log("FINISHED REDUCING INCOLLECTION FRAME")
    return reduced_incollection_frame

def reduce_inproceedings_frame() -> pd.DataFrame:
    log("STARTING REDUCING INPROCEEDINGS FRAME")
    log("STARTING LOADING INPROCEEDINGS DATA")
    inproceedings_frame = pd.read_csv(INPROCEEDINGS_PATH, sep=";", low_memory=False)
    log("FINISHED LOADING INPROCEEDINGS DATA")
    reduced_inproceedings_frame = pd.DataFrame({'ArticleId': inproceedings_frame.iloc[:, 0], 'key': inproceedings_frame.iloc[:, 14]})
    log("FINISHED REDUCING INPROCEEDINGS FRAME")
    return reduced_inproceedings_frame

def reduce_proceedings_frame() -> pd.DataFrame:
    log("STARTING REDUCING PROCEEDINGS FRAME")
    log("STARTING LOADING PROCEEDINGS DATA")
    proceedings_frame = pd.read_csv(PROCEEDINGS_PATH, sep=";", low_memory=False)
    log("FINISHED LOADING PROCEEDINGS DATA")
    reduced_proceedings_frame = pd.DataFrame({'ArticleId': proceedings_frame.iloc[:, 0], 'key': proceedings_frame.iloc[:, 14]})
    log("FINISHED REDUCING PROCEEDINGS FRAME")
    return reduced_proceedings_frame

def reduce_www_frame() -> pd.DataFrame:
    log("STARTING REDUCING WWW FRAME")
    log("STARTING LOADING WWW DATA")
    www_frame = pd.read_csv(WWW_PATH, sep=";", low_memory=False)
    log("FINISHED LOADING WWW DATA")
    reduced_www_frame = pd.DataFrame({'ArticleId': www_frame.iloc[:, 0], 'key': www_frame.iloc[:, 7]})
    log("FINISHED REDUCING WWW FRAME")
    return reduced_www_frame

def concat_all_publication_frames(reduced_article_frame : pd.DataFrame, reduced_book_frame : pd.DataFrame, reduced_incollection_frame : pd.DataFrame, reduced_inproceedings_frame : pd.DataFrame, reduced_proceedings_frame : pd.DataFrame, reduced_www_frame : pd.DataFrame) -> pd.DataFrame:
    log("STARTED CONCATENATING FRAMES")
    
    return pd.concat([reduced_article_frame, reduced_book_frame, reduced_incollection_frame, reduced_inproceedings_frame, reduced_proceedings_frame, reduced_www_frame])

def compile_cite_relationship(concatenated_publications_frame, cite_notes_merged):
    log("CREATING CITE RELATIONSHIP")
    merged_frame = pd.merge(cite_notes_merged, concatenated_publications_frame)
    merged_frame = pd.DataFrame({':START_ID': merged_frame.iloc[:, 0], ':END_ID' : merged_frame.iloc[:, 2]})
    log("FINISHED CREATING CITE RELATIONSHIP")
    return merged_frame

def main():
    concatenated_publications_frame = concat_all_publication_frames(reduce_article_frame(), reduce_book_frame(), reduce_incollection_frame(), reduce_inproceedings_frame(), reduce_proceedings_frame(), reduce_www_frame())
    log("FINISHED CONCATENATING FRAMES")
    cite_relationship = compile_cite_relationship(concatenated_publications_frame, merge_cite_nodes())
    cite_relationship.to_csv("output_cite_relationship.csv", sep=";",index=False)
    log("CITE RELATIONSHIP SAVED INTO output_cite_relationship.csv. ENDING...")

if __name__ == "__main__":
    main()