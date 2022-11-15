# Data Parser
In order to provide a version of the DBLP dataset as much information-complete as possible, we build a parser which merges different data sources information together.

## Requirements
* [dblp.xml](https://dblp.org/xml/dblp.xml.gz) file;
* [dblp.dtd](https://dblp.org/xml/dblp.dtd) file;
* [XMLtoCSV](https://github.com/ThomHurks/dblp-to-csv/blob/master/XMLToCSV.py) parser in order to transform the dblp.xml file into a set of csv files to preprocess later on;
* [Semantic Scholar Authors dataset](https://zenodo.org/record/7069915/files/2022-08-30-authors.csv.gz?download=1) csv file;
* [Semantic Scholar Papers dataset](https://zenodo.org/record/7069915/files/2022-08-30-papers.jsonl.gz?download=1) jsonl file;
* [S2ORC Papers with text dataset](https://drive.google.com/drive/folders/1EMhI28DJoyjcw4XQvlzHxcxh5wcpHnHq?usp=share_link) 5000 json documents (after downloading the package, please unzip all the subcompressed archieves until you have a folder containing 5000 json files. Place those files in a new directory called `json_docs` in your workspace folder, i.e. where you have placed the scripts and other files);
* Python 3.*

## Steps
Before launching our preprocessing tool, please follow these steps:
1. Install all the python required libraries by using the command 
```
pip install -r requirements.txt
```
2. Put all the files described before in the same directory;
3. Open the terminal and run
```
python ./XMLToCSV.py --neo4j dblp.xml dblp.dtd output.csv --relations author:authored_by journal:published_in publisher:published_by school:submitted_at editor:edited_by cite:has_citation series:is_part_of 
```
4. Once this script has finished working you will have in the directory a bounch of csv files containing the nodes and the relationships. Since these data are not complete, they need some other preprocessing operations before being able to be imported into Neo4J.
5. Our tool will provide the final restructuring and cleaning operations, so run:
```
python cucciolodifoca.py
```

```
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
    
What operation do you want to perform (N.B.: db reduction cannot be performed if the data are not previously cleaned)?
        1. Neo4J Setup
        2. MongoDB Setup
        3. MongoDB Random Setup
        4. Spark
        10. Exit
 Choice: 
 ```
### Neo4J
6. Choose option 1.
7. Finally open neo4j and copy all the csv files just created in the import folder of your dbms (N.B.: your DBMS should be switched off before doing this operation).
8. Open the terminal of the DBMS (click on the 3 dots on the right of the DBMS name in Neo4J desktop)
9. *_!!! ONLY WINDOWS USER !!!_* Move to the bin folder by typing:
```
cd bin
```
10. Open the neo4j_import.sh file which is provided in this repository folder; copy and run the code related to your operating system on the terminal.
11. Run the DBMS and run neo4j browser
12. Switch the current db from _neo4j_ to _system_.
13. Type the following query on the query text area and run it.
```cypher
CREATE DATABASE dblp.db
```
14. You have successfully imported the whole dataset into neo4j. Switch to that database to start using it.

### MongoDB
6. Choose option 2.
7. Open MongoDB Compass.
8. Connect to the MongoDB server.
9. Create a new database by using the bottom-left `+ CREATE DATABASE` button, by specifing as collection name `publications`.
10. Then open the database and the `publications` collection.
11. Finally click on `Import data` and pick from your pc the `papers_full_info.json` file generated by our script.