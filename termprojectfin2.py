# ## UI MUSIC REC APP ##
from cmu_112_graphics import *
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# IMPORTANT INSTALLS 
# pip install cmu-112-graphics pandas scikit-learn
# python.exe -m pip install --upgrade pip
# pip install scikit-learn
# pip install pandas scikit-learn


def appStarted(app):
    app.currentScreen = 'home'
    app.musicData = None
    app.recommendations = []
    app.kMeansResults = None
    app.currentRecIndex = 0
    app.recListLen = 0
    app.songList = []
    app.logoImage = app.loadImage('headphones.png')
    app.logoImage = app.scaleImage(app.logoImage, 0.3)
    app.final_recommendations = []
    app.data_incl_liked = []
    app.X = []




def loadMusicInput(filename, app):
    with open(filename, "r", encoding="utf-8") as f:
       fileString = f.read()




   # parse csv file
    app.songList = []
    for line in fileString.strip().splitlines()[1:]:
        songData = line.split(',')
        songDict = {
            'title': songData[0].strip(),
            'artist': songData[1].strip()
        }
        app.songList.append(songDict)
    return app.songList




def uploadMusicButtonAction(app):
    app.currentScreen = 'upload'
    # load data
    # app.musicData = loadMusicInput("hardcoded_songs_sample.csv", app)
    app.recommendations = kMeansClusterAlg(app)
    app.currentScreen = 'recommendations'
    return




def kMeansClusterAlg(app):
    # Maelle will implement this
    #curr. placeholder

    print("training in process")

    data = pd.read_csv("./archive/spotify_data.csv")

    # Example features for recommendation
    features = ['danceability', 'energy', 'tempo', 'valence', 'acousticness']
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    app.X = X
    
    # Define liked songs
    liked_songs = pd.read_csv("initial_liked_songs.csv")  # Load liked songs
    app.musicData = liked_songs
    liked_song_set = set(zip(liked_songs['artist_name'], liked_songs['track_name']))
    
    print("training in process2")
    all_songs_zipped = zip(data['artist_name'], data['track_name'])
    data['liked'] = [(1 if (artist, track) in liked_song_set else 0) for (artist, track) in all_songs_zipped]
    app.data_incl_liked = data
    
    print("training in process3")
    # Train the KNN Regressor
    knn_regressor = neighbors.KNeighborsRegressor(n_neighbors=15, metric='euclidean')
    knn_regressor.fit(X, data['liked'])
    
    # print("training in process4")
    # Predict similarity scores
    data['similarity_score'] = knn_regressor.predict(X)
    # recommendations = data.sort_values(by='similarity_score', ascending=False).head(25)
    
    # print("training in process5")
    # Return top 25 recommendations as a list of tuples
    # result =  [[row['track_name'], row['artist_name']] for _, row in recommendations.iterrows()]
    # app.recListLen = 25
    # print("result")
    # print(result)

    # hardcoded_recs = [["Firework", "Katy Perry"], ["This Love", "Maroon 5"], ["Payphone", "Maroon 5"], ["Pink Pony Club", "Chappell Roan"], ["Counting Stars", "OneRepublic"], ["One Foot In Front of the Other", "WALK THE MOON"]]
    # app.recListLen = 6


    print("training in process4")
    artist_info = pd.read_csv("Empirical Musicology Review Popular-music Artist Demographic Database - EMR (MBB, RS200, UG).csv")
    data_with_artist_info = data.merge(
        artist_info, 
        left_on='artist_name', 
        right_on='artist', 
        how='left'
    )

    # Fill missing values for demographic columns with 0 (assuming missing values mean no diversity attribute applies)
    diversity_columns = ['nonmale', 'non-cis', 'race', 'ethnicity']
    data_with_artist_info[diversity_columns] = data_with_artist_info[diversity_columns].fillna(0)

    # Filter diverse artists based on provided attributes
    diverse_artists = data_with_artist_info[
        (data_with_artist_info['nonmale'] == 1) | 
        (data_with_artist_info['non-cis'] == 1) | 
        (data_with_artist_info['race'] == 1) | 
        (data_with_artist_info['ethnicity'] == 1)
    ]

    print("training in process5")
    # Compute similarity scores for diverse artists
    diverse_artists['similarity_score'] = knn_regressor.predict(scaler.transform(diverse_artists[features]))

    # Select top similarity-based and diverse recommendations
    similarity_top = data_with_artist_info.sort_values(by='similarity_score', ascending=False).head(15)
    diversity_top = diverse_artists.sort_values(by='similarity_score', ascending=False).head(15)

    # print('similarity_top')
    # print( similarity_top[['artist_name', 'track_name', 'similarity_score']])

    # print('diversity_top')
    # print( diversity_top[['artist_name', 'track_name', 'similarity_score']])

    # Combine the two sets, ensuring no duplicates
    final_recommendations = pd.concat([similarity_top, diversity_top]).drop_duplicates()

    # Compute a combined score for similarity and diversity
    final_recommendations['combined_score'] = (
        0.7 * final_recommendations['similarity_score'] +
        0.3 * (final_recommendations[diversity_columns].sum(axis=1))
    )

    print("training in process6")

    # Sort by combined score and get the final top recommendations
    app.final_recommendations = final_recommendations.sort_values(by='combined_score', ascending=False).head(25)

    # Output hybrid recommendations
    print("Top hybrid recommendations (similarity + diversity):")
    print(final_recommendations[['artist_name', 'track_name', 'similarity_score', 'combined_score']])
    result = [[row['track_name'], row['artist_name']] for _, row in final_recommendations.iterrows()]
    app.recListLen = 25

    # Output the result
    # print("Top hybrid recommendations (similarity + diversity):")
    # print(result)

    print("done")
    kMeansClusterImage(app)
    return result






def kMeansClusterImage(app):
    # Maelle will also implement this

    
    # Create a 2D projection of the feature space for visualization using PCA
    



    if ( (len(app.data_incl_liked) == 0) or (len(app.final_recommendations) == 0)):
        print("cant make graph, no data yet")
        return 
    data = app.data_incl_liked
    final_recommendations = app.final_recommendations


    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(app.X)



    # colors = ['red' if liked else 'lightblue' for liked in data['liked']]  # Red for liked songs, light blue for others
    # sizes = [200 if liked else 50 for liked in data['liked']]  # Larger markers for liked songs, smaller for others
    # alphas = [1.0 if liked else 0 for liked in data['liked']]  # Higher opacity for liked songs

    # Create a scatter plot for all songs
    plt.figure(figsize=(10, 8))


    # Plot non-liked songs (light blue, transparent) first
    not_liked_indices = data[data['liked'] == 0].index

    plt.scatter(
        X_2d[not_liked_indices, 0], 
        X_2d[not_liked_indices, 1], 
        c='lightblue', 
        s=50, 
        alpha=0.3, 
        label='Non-Liked Songs'
    )

    # Plot liked songs (red, opaque) next
    liked_indices = data[data['liked'] == 1].index

    plt.scatter(
        X_2d[liked_indices, 0], 
        X_2d[liked_indices, 1], 
        c='red', 
        s=200, 
        alpha=1.0, 
        label='Liked Songs'
    )



    # Plot recommended songs (green) on top
    recommended_indices = final_recommendations.index

    plt.scatter(
        X_2d[recommended_indices, 0], 
        X_2d[recommended_indices, 1], 
        c='green', 
        s=150, 
        alpha=0.9, 
        label='Recommended Songs'

    )



    # Add annotations for liked songs
    for i in liked_indices:
        plt.annotate(
            data.loc[i, 'track_name'], 
            (X_2d[i, 0], X_2d[i, 1]), 
            fontsize=9, 
            alpha=0.7
        )



    # Labeling and legend
    plt.title("Song Recommendations Based on Liked Songs", fontsize=14)
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    return




def likeRecButtonAction(app):
    #recommend another song similar to this one
    if app.currentRecIndex + 1 < app.recListLen:
        app.currentRecIndex = app.currentRecIndex + 1
    else:
        app.currentScreen = 'final'
    return






def dislikeRecButtonAction(app):
    #delete the song from the list
    if app.currentRecIndex + 1 < app.recListLen:
        app.recommendations.pop(app.currentRecIndex)
        app.recListLen -= 1
    else:
        app.recommendations.pop(app.currentRecIndex)
        app.recListLen -= 1
        app.currentScreen = 'final'
    return








def redrawAll(app, canvas):
    canvas.create_rectangle(0, 0, app.width, app.height, fill="lightblue", outline="")
    canvas.create_rectangle(20, 20, app.width - 20, app.height - 20, fill="white", outline="")
    if app.currentScreen == 'home':
        canvas.create_text(app.width//2, app.height//3 - 100, text="Welcome to the Music Recommendation App!", fill="black", font="Script 22 bold")
        canvas.create_image(app.width//2, app.height//3 + 20, image=ImageTk.PhotoImage(app.logoImage))

        canvas.create_text(app.width//2, app.height//3 + 150, text="To upload data, make a spreadsheet with 3 columns", fill="black", font="Script 14")
        canvas.create_text(app.width//2, app.height//3 + 170, text="(Song Number, Song Name, Song Artist). Then, download it as a .csv file!", fill="black", font="Script 14")
        canvas.create_text(app.width//2, app.height//3 + 200, text="Once you press Upload Music Data, the model will begin training, and a K-Means", fill="black", font="Script 14")
        canvas.create_text(app.width//2, app.height//3 + 220, text="graph will pop up. The K-Means graph uses your input song data to create a visual", fill="black", font="Script 14")
        canvas.create_text(app.width//2, app.height//3 + 240, text="representation of the algorithm's clusters of the recommended, liked, and", fill="black", font="Script 14")
        canvas.create_text(app.width//2, app.height//3 + 260, text="non-liked songs", fill="black", font="Script 14")
        canvas.create_text(app.width//2, app.height//3 + 300, text="Make sure you close the graph to continue.", fill="black", font="Script 14")


    
        # draw upload button w/ adjusted y pos.
        buttonTopY = app.height//2 + 200
        buttonBottomY = buttonTopY + 40
        canvas.create_rectangle(app.width//2-70, buttonTopY, app.width//2+70, buttonBottomY, fill="black")
        canvas.create_text(app.width//2, buttonTopY + 20, text="Upload Music Data", fill="white")


        #draw transparency button
        tfButtonTopY = buttonBottomY + 20
        tfButtonBottomY = tfButtonTopY + 40
        canvas.create_rectangle(app.width//2-80, tfButtonTopY, app.width//2+80, tfButtonBottomY, fill="black")
        canvas.create_text(app.width//2, tfButtonTopY + 20, text="Transparency/Fairness", fill="white")


    elif app.currentScreen == 'transparency':
        canvas.create_text(app.width//2, app.height//4, text="Transparency & Fairness", font="Arial 20 bold", fill="black")
        canvas.create_text(app.width//2, app.height//3, text="This algorithm takes the song and artist data you upload,", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//3 + 20, text="and compares it on different metrics (genre, danceability,", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//3 + 40, text="energy, mood) to other songs in our dataset.", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//3 + 80, text="Then, we'll give you some recommendations of similar", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//3 + 100, text="songs from artists with different backgrounds!", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//2 + 140, text="Your uploaded song data and your recommendations", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//2 + 160, text="will not be saved by this app or our algorithm.", fill="black", font="Arial 16")


            #see how this works button
        seeHowButtonTopY = app.height//3 + 140
        seeHowButtonBottomY = seeHowButtonTopY + 40
        canvas.create_rectangle(app.width//2 - 70, seeHowButtonTopY, app.width//2 + 70, seeHowButtonBottomY, fill="black")
        canvas.create_text(app.width//2, seeHowButtonTopY + 20, text="See How This Works!", fill="white")
        
        #back button, go back to home screen
        backButtonTopY = app.height - 80
        backButtonBottomY = backButtonTopY + 40
        canvas.create_rectangle(app.width//2-50, backButtonTopY, app.width//2+50, backButtonBottomY, fill="black")
        canvas.create_text(app.width//2, backButtonTopY + 20, text="Back", fill="white")
    elif app.currentScreen == 'algorithmExplanation':
        canvas.create_text(app.width//2, app.height//4, text="Algorithm Explanation", font="Arial 20 bold", fill="black")
        canvas.create_text(app.width//2, app.height//3, text="Our recommendation system uses K-Means clustering", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//3 + 20, text="to group songs by various metrics. This allows us to find songs", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//3 + 40, text="that share similar characteristics with your song data.", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//3 + 80, text="Each cluster represents a distinct musical 'mood' or 'genre',", fill="black", font="Arial 16")
        canvas.create_text(app.width//2, app.height//3 + 100, text="which helps us recommend songs that you might like!", fill="black", font="Arial 16")
        
        #back to transparency screen
        backButtonTopY = app.height - 80
        backButtonBottomY = backButtonTopY + 40
        canvas.create_rectangle(app.width//2-50, backButtonTopY, app.width//2+50, backButtonBottomY, fill="black")
        canvas.create_text(app.width//2, backButtonTopY + 20, text="Back", fill="white")
    
    elif app.currentScreen == 'recommendations':
        if app.recommendations:
            song, artist = app.recommendations[app.currentRecIndex]
            canvas.create_text(app.width//2, app.height//3, text=f"Recommendation:", font="Script 18 bold underline")
            canvas.create_text(app.width//2, app.height//3 + 40, text=f"{song} by {artist}", fill="black", font="Script 20")
            
            canvas.create_rectangle(app.width//2-50, app.height//2, app.width//2+50, app.height//2+40, fill="black")
            canvas.create_text(app.width//2, app.height//2+20, text="Like", fill="white")
            canvas.create_rectangle(app.width//2-50, app.height//2+60, app.width//2+50, app.height//2+100, fill="black")
            canvas.create_text(app.width//2, app.height//2+80, text="Dislike", fill="white")
    
    
    elif app.currentScreen == 'final':
        canvas.create_text(app.width//2, app.height//9, text="Your Recommendations:", font="Arial 18 bold", fill="black")
        for i, (song, artist) in enumerate(app.recommendations):
            canvas.create_text(app.width//2, app.height//7 + i * 20, text=f"{song} by {artist}", fill="black")
        # image
        #canvas.create_image(app.width//2, app.height * 0.85, image=ImageTk.PhotoImage(app.logoImage)) # logoImage to be changed to kmeans graph image












def mousePressed(app, event):
    if app.currentScreen == 'home':
        #use same y-coords from redrawAll for button detection
        buttonTopY = app.height//2 + 200
        buttonBottomY = buttonTopY + 40
        if (app.width//2 - 70 < event.x < app.width//2 + 70) and (buttonTopY < event.y < buttonBottomY):
            uploadMusicButtonAction(app)


        tfButtonTopY = buttonBottomY + 20
        tfButtonBottomY = tfButtonTopY + 40
        if (app.width//2 - 70 < event.x < app.width//2 + 70) and (tfButtonTopY < event.y < tfButtonBottomY):
            app.currentScreen = 'transparency'
    elif app.currentScreen == 'transparency':


        seeHowButtonTopY = app.height//3 + 140
        seeHowButtonBottomY = seeHowButtonTopY + 40
        if (app.width//2 - 70 < event.x < app.width//2 + 70) and (seeHowButtonTopY < event.y < seeHowButtonBottomY):
            app.currentScreen = 'algorithmExplanation'


        backButtonTopY = app.height - 80
        backButtonBottomY = backButtonTopY + 40
        if (app.width//2 - 50 < event.x < app.width//2 + 50) and (backButtonTopY < event.y < backButtonBottomY):
            app.currentScreen = 'home'


    elif app.currentScreen == 'algorithmExplanation':
        #back button
        backButtonTopY = app.height - 80
        backButtonBottomY = backButtonTopY + 40
        if (app.width//2 - 50 < event.x < app.width//2 + 50) and (backButtonTopY < event.y < backButtonBottomY):
            app.currentScreen = 'transparency'


    elif app.currentScreen == 'recommendations':


        if (app.width//2-50 < event.x < app.width//2+50) and (app.height//2 < event.y < app.height//2+40):
            likeRecButtonAction(app)
        elif (app.width//2-50 < event.x < app.width//2+50) and (app.height//2+60 < event.y < app.height//2+100):
            dislikeRecButtonAction(app)












runApp(width=600, height=800)










