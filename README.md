# GUiIO_pt2
The aim of the course is to learn about methods of computational intelligence and the latest developments related to these methods, and to acquire the ability to use them appropriately to optimise problems occurring in automation and computer science

## CNN
cnn/cnn_inference_example.ipynb można w połowie uruchomić od razu tj. część do wpisywania własnego komentarza i sprawdzenia oceny. Do drugiej części tj. macierz konfuzji wymagany jest plik test_embeddings.feather. Można go wygenerować przy użyciu cnn/cnn_data_embedding.ipynb. 

cnn/cnn_network_training.ipynb służy do trenowania sieci. Posiada zcachowane informacje w jaki sposób była trenowana sieć. Wymaga plików z embeddingami które można stworzyć za pomocą cnn/cnn_data_embedding.ipynb. 

cnn/cnn_optional_embedding_to_df.ipynb jest dodatkowym plikiem służącym do konwersji plików .npy na .feather

UWAGA - bardzo możliwe, że trzeba będzie lekko zmodyfikować ścieżki plików