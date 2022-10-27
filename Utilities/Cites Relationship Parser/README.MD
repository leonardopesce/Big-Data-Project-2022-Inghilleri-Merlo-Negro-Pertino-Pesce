# Data Parser
In order to provide a version of the DBLP dataset as much information-complete as possible, we build a parser which merges different data sources information together.

## Requirements
* [dblp.xml](https://dblp.org/xml/dblp.xml.gz) file;
* [dblp.dtd](https://dblp.org/xml/dblp.dtd) file;
* [XMLtoCSV](https://github.com/ThomHurks/dblp-to-csv/blob/master/XMLToCSV.py) parser in order to transform the dblp.xml file into a set of csv files to preprocess later on;
* [Semantic Scholar Authors dataset](https://zenodo.org/record/7069915) file;
* Python 3.*
* Pandas library
* NumPy library
* lxml library

## Steps
Before launching our preprocessing tool, please follow these steps:
1. Put all the files described before in the same directory;
2. Open the terminal and run
```
python ./XMLToCSV.py --neo4j dblp.xml dblp.dtd output.csv --relations author:authored_by journal:published_in publisher:published_by school:submitted_at editor:edited_by cite:has_citation series:is_part_of 
```
3. Once this script has finished working you will have in the directory a bounch of csv files containing the nodes and the relationships. Since these data are not complete, they need some other preprocessing operations before being able to be imported into Neo4J.
4. Our tool will provide the final restructuring and cleaning operations, so run:
```
python cucciolodifoca.py
```
5. Finally open neo4j and copy all the csv files just created in the import folder of your dbms (N.B.: your DBMS should be switched off before doing this operation).
6. Open the terminal of the DBMS (click on the 3 dots on the right of the DBMS name in Neo4J desktop)
7. *_!!! ONLY WINDOWS USER !!!_* Move to the bin folder by typing:
```
cd bin
```
8. Open the neo4j_import.sh file which is provided in this repository folder; copy and run the code related to your operating system on the terminal.
9. Run the DBMS and run neo4j browser
10. Switch the current db from _neo4j_ to _system_.
11. Type the following query on the query text area and run it.
```cypher
CREATE DATABASE dblp.db
```
12. You have successfully imported the whole dataset into neo4j. Switch to that database to start using it.