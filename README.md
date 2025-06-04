# GUiIO_pt2
The aim of the course is to learn about methods of computational intelligence and the latest developments related to these methods, and to acquire the ability to use them appropriately to optimise problems occurring in automation and computer science

## CNN
cnn/cnn_inference_example.ipynb można w połowie uruchomić od razu tj. część do wpisywania własnego komentarza i sprawdzenia oceny. Do drugiej części tj. macierz konfuzji wymagany jest plik test_embeddings.feather. Można go wygenerować przy użyciu cnn/cnn_data_embedding.ipynb. 

cnn/cnn_network_training.ipynb służy do trenowania sieci. Posiada zcachowane informacje w jaki sposób była trenowana sieć. Wymaga plików z embeddingami które można stworzyć za pomocą cnn/cnn_data_embedding.ipynb. 

cnn/cnn_optional_embedding_to_df.ipynb jest dodatkowym plikiem służącym do konwersji plików .npy na .feather

UWAGA - bardzo możliwe, że trzeba będzie lekko zmodyfikować ścieżki plików

## RNN - LSTM
Wszystkie trzy pliki można uruchomić by zbadać wyniki uczenia danej sieci. W plikach zawarte są: prosty model sieci LSTM - rnn_lstm.ipynb oraz modele wykorzystujące dwukierunkowe bloki LSTM: z mniejszą ilością warstw - rnn_mala_bi_lstm.ipynb oraz bardziej rozbudowany - rnn_bi_lstm.ipynb.


## Transformer
Żeby uruchomienie było możliwe najpierw trzeba pobrac i wypakować pilk .7z z dysku google (https://drive.google.com/file/d/1XNNPDQOZLEPU-xKVmAaiA11QtR37XtlA/view?usp=sharing). Plik ten zawiera potrzebne pliki oraz modele potrzebne do uruchomienia notatników Jupyter, opórcz notatnika zwiazanego z klasyfikatorem wytrenowanym na modelu gpt2-small. Możliwe będzie zmodyfikowanie scieżek do plików, w szczególności do wczytywania modeli. Same modele należy wczytywać korzystając z funkcji load_model wraz z argumentem "with_optimizer=True", funkcja zwraca wczytany model oraz optymalizator, w większości przypadków należy zignorować. 
