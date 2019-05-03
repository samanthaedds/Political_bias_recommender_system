# Political_bias_recommender_system
Recommender system for political bias coded in Python based on data pulled from news APIs for SI650 Information Retrieval. 
We built a recommender system for news that incorporates a user’s prior known behavior and political leanings, and
takes into account a user’s feedback. Our system then recommends something the user
might not otherwise read. We create simulated users to demonstrate our results (SI650_Project_Simulations.py), and
additionally build an interactive interface to be used in real time, very similar to our simulations.
Any code written by my project partner is noted (lines 312-382 in SI650_Project_Simulations.py).

# Big idea
We simulate the click logs of 10 users with different political leanings, creating an average political leaning score for each
user (based on PEW Research and other accredited sources noted). That score and their initial broad query are used as inputs for our BM25, which
also includes weights for political leanings.Based on the user’s broad query BM25 takes online news articles tagged by category
and political leaning, determines the relevant subset by category, and ranks the articles.
As mentioned above the political leaning of the source is included as a weight. Results
are then filtered to a user based on Precision at k, where k = 20.

Our user then provides feedback, marking the articles relevant to them, which is then
incorporated as a weight in the next iteration of BM25. Since the user initially provides
a very broad category of interest, such as "Business", if they choose mostly "Finance"
articles, the next time they ask for "Business" we would rank "Finance" articles higher
than other articles.
