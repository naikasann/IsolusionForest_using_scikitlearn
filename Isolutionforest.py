import os
import datetime
import yaml
import pprint
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import pickle

# Global variables required at runtime
execution_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
save_folder_name = "./result/kfold/" + execution_time
# Create a folder of results
os.makedirs(save_folder_name, exist_ok=True)
# Keep the log file open. There's a better way to write this... Well, forgive me ...
logfile = open(save_folder_name + "/result.log", mode="w")

def logprint(string):
    """
    func : Output to log and output to terminal.
           I know about logging and stuff, but I don't use it ... . I really don't! ;(
    """
    print(string)
    print(string, file=logfile)

def save_model(forest, save_folder_name):
    """ func : Save the model trained by randomforest. """
    # Create a folder of model.
    save_folder_name = save_folder_name + "/model/"
    os.makedirs(save_folder_name, exist_ok=True)
    # Save in joblib and pickle.
    joblib.dump(forest, save_folder_name + "model.joblib")
    with open(save_folder_name + 'model.pkl', 'wb') as model_file:
        pickle.dump(forest, model_file)

def make_dataset(x_train, y_train, x_test, y_test, save_folder_name):
    """
    func :  Create the data set used for each Fold.
            (so that you can conduct replicated experiments)
    """
    # Creating training data.
    with open(save_folder_name + "/traindata.csv", mode="w") as traindata_file:
        for (x, y) in zip(x_train, y_train):
            traindata_file.write(",".join(map(str, x.tolist())) + "," + str(y) + "\n")

    # Creating test data.
    with open(save_folder_name + "/testdata.csv", mode="w") as traindata_file:
        for (x, y) in zip(x_test, y_test):
            traindata_file.write(",".join(map(str, x.tolist())) + "," + str(y) + "\n")

def make_cm(matrix, columns):
    cm = pd.DataFrame(matrix, columns=[['Predicted Results'] * len(columns), columns], index=[['Correct answer data'] * len(columns), columns])
    logprint(cm)
    return cm

def output_result(forest , y_pred, X_test, Y_test, config, save_folder_name):
    """
    func : Test the training results and output the accuracy.
            (In confusion matrix and classification report)
    """
    # Output and save the confusion matrix.
    matrix = confusion_matrix(Y_test, y_pred, labels=[0,1])
    matrix_df = make_cm(matrix, ["normal", "outlier"])
    matrix_df.to_csv(save_folder_name + "/confusion_matrix.csv")
    # Save the image of the confusion matrix.
    plt.figure(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, square=True, cmap='Blues', fmt='g')
    plt.savefig(save_folder_name + "/confusion_matrix.png")

    # Output and save the classification report.
    csv_name = save_folder_name + "/classification_report.csv"
    classifycation_repo_df = pd.DataFrame(classification_report(Y_test, y_pred, target_names=["normal", "outlier"], output_dict=True))
    classifycation_repo_df.T.to_csv(csv_name)
    score_df = pd.DataFrame([[accuracy_score(Y_test, y_pred),
                         precision_score(Y_test, y_pred, average="micro"),
                         recall_score(Y_test, y_pred, average="micro"),
                         f1_score(Y_test, y_pred, average="micro"),
                         matthews_corrcoef(Y_test, y_pred)]],
                         columns=["accuracy","precision","recall","f1 score", "mcc"])
    score_df.to_csv(csv_name, header=False, index=False, mode="a")

    pprint.pprint(classifycation_repo_df)
    pprint.pprint(classifycation_repo_df, stream=logfile)
    logprint(score_df)

    return score_df

def main():
    # Load the configuration file.
    with open("kfold_train_config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Set the number of data types to be used for training.
    data_number = config["data_number"]

    # Create a normal data set.
    normal_df = pd.read_csv(config["normal"])
    # Extracting data
    X_normal = normal_df[normal_df.columns[normal_df.columns != normal_df.columns[data_number]]].values
    target = normal_df[normal_df.columns[data_number]].values
    Y_normal = [0] * len(X_normal)
    # Create a outlier data set.
    outlier_df = pd.read_csv(config["outlier"])
    X_outlier = outlier_df[outlier_df.columns[outlier_df.columns != outlier_df.columns[data_number]]].values
    Y_outlier = [1] * len(X_outlier)
    target = np.concatenate([target, outlier_df[outlier_df.columns[data_number]].values])

    logprint("all normal data : {}".format(len(X_normal)))
    logprint("all outlier data : {}".format(len(X_outlier)))

    # くっつけて一つのデータセットにする
    X_data = np.concatenate([X_normal, X_outlier], axis=0)
    Y_data = np.concatenate([Y_normal, Y_outlier], axis=0)

    # Split the data for K-cross validation()
    kfold = StratifiedKFold(n_splits=config["K_fold"], shuffle=True)
    result_df = pd.DataFrame(columns=["accuracy","precision","recall","f1 score", "mcc"])
    # Start each fold experiment.
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_data, Y_data)):
        logprint("=== Fold {} ===".format(fold + 1))
        # Retrieve real data from the index
        X_train, Y_train, X_test, Y_test  = X_data[train_idx], Y_data[train_idx], X_data[test_idx], Y_data[test_idx]
        target_for_tsne = target[test_idx]

        logprint("train data : {}".format(X_train.shape[0]))
        logprint("test  data : {}".format(X_test.shape[0]))

        # IsolationForest
        forest = IsolationForest(n_estimators=100,
                            max_features = 6,
                            random_state=1234)
        forest.fit(X_train)

        # test data predict
        y_pred = forest.predict(X_test)

        # IsolationForest returns the result as normal=1 abnormal=-1
        # => normal = 0, outlier = 1
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        # THRETHOLD setting.
        THRETHOLD = 0.06
        outlier_score = forest.decision_function(X_test)
        # Get indexes below the outlier score.
        predicted_outlier_index = np.where(outlier_score < THRETHOLD)
        y_pred[predicted_outlier_index] = 1

        save_folder_name = "./result/kfold/" + execution_time
        # Create a folder to store the results of each fold experiment.
        save_folder_name = save_folder_name + "/fold{}".format(fold+1)
        os.makedirs(save_folder_name, exist_ok=True)

        # Create a data set.
        make_dataset(X_train, Y_train, X_test, Y_test,save_folder_name)
        # Output and save each learning result.
        save_model(forest, save_folder_name)
        score_df = output_result(forest, y_pred, X_test, Y_test, config, save_folder_name)
        # k-cross-validation method concatenate data frames to display all training results
        result_df = pd.concat([result_df, score_df])

        # Using t-SNE for dimension reduction(2D)
        tsne = TSNE(n_components=2)
        X_embedded = tsne.fit_transform(X_test)

        # Create a two-dimensional scatter plot.
        fig = plt.figure(figsize=(16,8))
        # for plot
        # test data predict
        y_pred = forest.predict(X_test)
        # IsolationForest returns the result as normal=1 abnormal=-1
        # => normal = 1, outlier = 0
        y_pred[y_pred == 1] = 1
        y_pred[y_pred == -1] = 0

        plt.subplot(1,2,1)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred, cmap=cm.tab10)
        plt.title("Isolution forest result")
        plt.colorbar()
        plt.legend(fontsize = 10)
        # Create a two-dimensional scatter plot.
        plt.subplot(1,2,2)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target_for_tsne, cmap=cm.tab10)
        plt.title("Isolution forest result")
        plt.colorbar()
        plt.legend(fontsize = 10)
        # save figure.
        fig.savefig(save_folder_name + "/TSNE_result_2D.png")

        logprint("==============")

    # Output and save all training results of k-cross-validation method.
    logprint("=== result ===")
    result_df.index = result_df.index + 1
    logprint(result_df)
    logprint("==============")
    result_df.to_csv("./result/kfold/" + execution_time + "/experiment_result.csv")


if __name__ == "__main__":
    main()