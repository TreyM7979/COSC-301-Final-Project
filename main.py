import pandas as pd
import sqlite3
from sklearn.cluster import KMeans

#loading the data into pandas
players = pd.read_csv("../data/raw/players.csv")
player_data = pd.read_csv("../data/raw/player_data.csv")
stats = pd.read_csv("../data/raw/seasons_stats.csv")

#clean the columns
players = players.loc[:, ~players.columns.str.contains("^Unnamed")]#remove unnamed columns
player_data = player_data.loc[:, ~player_data.columns.str.contains("^Unnamed")]
stats = stats.loc[:, ~stats.columns.str.contains("^Unnamed")]

#fix player names mix up
player_data.rename(columns={"name": "Player"}, inplace=True)#rename name to Player to match other datasets

#merge data
df = stats.merge(players, on="Player", how="left")#left join to keep all stats, even if player info is missing
df = df.merge(player_data, on="Player", how="left")

#remove duplicate columns
df = df.drop(columns=["height_y", "weight_y"], errors="ignore")#drop the duplicate height and weight columns from player_data
df.rename(columns={"height_x": "height", "weight_x": "weight"}, inplace=True)#rename for clarity

#clean columns names
df.columns = df.columns.str.lower().str.replace(" ", "_")#convert to lowercase and replace spaces with underscores

#handle all the missing values
df = df.dropna(subset=["pts", "ast", "trb"], how="all")#drop rows where all three key stats are missing

df.to_csv("../data/processed/nba_clean.csv", index=False)#save the cleaned data to a new CSV file

#clustering
features = df[["pts", "ast", "trb"]].dropna()#select the stats we want and drop rows with missing values for clustering

kmeans = KMeans(n_clusters=3, random_state=42)#fit the model and predict clusters
features["cluster"] = kmeans.fit_predict(features)#add labels

#attach cluster back
df = df.loc[features.index]
df["cluster"] = features["cluster"]#add the cluster labels back


conn = sqlite3.connect("../data/processed/nba.db")#save the cleaned and clustered data to SQLite
df.to_sql("nba_stats", conn, if_exists="replace", index=False)#save the data to a table in the database
conn.close()

#correlation between stats/heatmap
conn = sqlite3.connect("../data/processed/nba.db")#connecting to sqlite
query = """
SELECT pts, ast, trb, stl, blk
FROM nba_stats
"""
#selecting key stats from nba_stats(pts, ast, trb, stl, blks) to analyze there correlation
data = pd.read_sql(query, conn)#load the query results into data
conn.close()#close the connection
corr = data.corr()#set corr as the heatmap of the selected stats
print(corr)#print the heatmap to console

#points over time
conn = sqlite3.connect("../data/processed/nba.db")#connecting to sqlite
query = """
SELECT year, AVG(pts) as avg_pts
FROM nba_stats
GROUP BY year
ORDER BY year
"""
#selecting the year and average points, from nba_stats/the sqlite, grouping and ordering by year so we can see the trend of points over time
result = pd.read_sql(query, conn)#load the query results into result
conn.close()#close the connection
result.to_csv("../data/processed/points_over_time.csv", index=False)#save the results to a new CVS file for excel

#cluster
conn = sqlite3.connect("../data/processed/nba.db")#connecting to sqlite
query = """
SELECT cluster, AVG(pts) as avg_pts, AVG(ast) as avg_ast, AVG(trb) as avg_trb
FROM nba_stats
GROUP BY cluster
"""
#selecting cluster, avergae points, average assists, average boards, from nba_stats and grouping by cluster
cluster_result = pd.read_sql(query, conn)#load the query results into cluster_result
conn.close()#close the connection
print(cluster_result)#print the cluster summary to console
cluster_result.to_csv("../data/processed/cluster_summary.csv", index=False)#save the cluster summary to a new CSV file for excel