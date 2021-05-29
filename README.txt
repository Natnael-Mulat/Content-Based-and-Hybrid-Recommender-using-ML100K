--------------------------------------------------------------------------------------
Content-based recommenders: Feature Encoding, TF-IDF/CosineSim and Hybrid Recommenders 
for Ml-100k ratings
Programmers/Researchers: Paul Choi, Basel Elzatahry, Rida Shahid, Natnael Mulat
Institution: Davidson College
--------------------------------------------------------------------------------------
Data - 'ML-100k'
======================================================================================
This program is designed to run content-based and hybrid recommender systems for 
MovieLens 100K movie ratings with 943 users on 1682 movies, where each user has rated 
at least 20 movies.The data was collected through the MovieLens web site
(movielens.umn.edu)

Libraries Used
======================================================================================

   NumPy	       -- a library for the Python programming language, a large collection 
		of high-level mathematical functions to operate on these arrays		   
   Matplotlib   -- a plotting library for the Python programming language and its 
		numerical mathematics extension NumPy.
   Pickle       -- a library for serializing and de-serializing a Python object 
		structure. Any object in Python can be pickled so that it can be 
		saved on disk. 	
   Scikit-learn -- a machine learning library for Python.
   Scipy        -- a library for Python for scientific computing and technical computing.


DETAILED DESCRIPTIONS OF COMMANDS
======================================================================================
R   --  To read the critics data into the program, please note that this command
	    is a required command for other function calls

RML   --  To read the ml-100k data into the program, please note that this command
	    is a required command for other function calls

FE    --  To run (set up) feature encoding on the data and generate the feature preference 
          matrix.

TFIDF     --  To run (set up) TFIDF by generating the cosine similarity matrix, please note
              that the user will be required to choose the sim threshold.

CBR-FE    --  To run feature encoding for a user input user and calculate the top N 
              recommendations for them.

CBR-TFIDF     --  To run TFIDF for a user input user and calculate the top N 
                  and single recommendations for them.

SIM    --  reads/writes a Item-Item similarity matrix, please note that this has 
	   additional command when run that specifies the similarity method.

H    --  To set up and run hybrid recommenders for a user input user and calculate 
           the top N and single recommendations for them, please note
              that the user will be required to choose the hybrid threshold.
 
LCVSIM --  Leave one out cross validation that returns the error and error list,
	   please note that this command will ask for choice of recommender system
	   between all algorithms

TEST   -- Performs the test of hypothesis using error lists by calculating the p-value and
          using the t-test, please note that this command will ask for choice of recommender 
          systems to compare
 
-----------------------------------------------------------------------------------------
SET OF COMMANDS TO GET RESULTS 
-----------------------------------------------------------------------------------------
RML --> FE --> CBR-FE

This set of commands reads the ML-100K dataset, sets up feature encoding and prints the 
top N and single item recommendations calculated using FE for a given user. 


RML --> TFIDF --> CBR-TFIDF

This set of commands reads the ML-100K dataset, sets up TFIDF, prints the cosine sim matrix and histogram and prints the top N and single item recommendations calculated using TFIDF for a given user.  


RML --> TFIDF --> SIM --> H

This set of commands reads the ML-100K dataset, sets up TFIDF, prints the cosine sim matrix and histogram, calculates the item-item similarities and prints the top N and single item recommendations calculated using hybrid recommender for a given user. 


RML --> FE --> LCVSIM --> FE

This set of commands gives us the MSE, MSE_list, MAE, MAE_list, RMSE, RMSE_list for
a FE recommender. The leave-one-out-cross validation uses the matrix and recommender functions to find the prediction and actual value for calculation of errors. 


RML --> TFIDF --> LCVSIM --> TFIDF

This set of commands gives us the MSE, MSE_list, MAE, MAE_list, RMSE, RMSE_list for
a TFIDF recommender. This set of command uses the cosine-sim matrix to make predictions. 
The leave-one-out-cross validation uses the matrix and recommender functions to find the prediction and actual value for calculation of errors. 


RML --> TFIDF --> SIM --> LCVSIM --> H

This set of commands gives us the MSE, MSE_list, MAE, MAE_list, RMSE, RMSE_list for
a hybrid recommender. This set of command uses the cosine-sim and item-item matrices to make predictions. The leave-one-out-cross validation uses the matrix and recommender functions 
to find the prediction and actual value for calculation of errors. 


RML --> TEST 

This set of commands gives the p-values for two models chosen by the user, please note that 
the chosen model should have already been run with lcvsim.

ACKNOWLEDGEMENTS
============================================================================================

Thanks to DR. CARLOS E.SEMINARIO for all the guidance he provided through out this research.