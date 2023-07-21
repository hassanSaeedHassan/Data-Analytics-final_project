from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import ConfusionMatrixDisplay
# imports for neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class nn_modeling:
    
    def __init__(self,model, X_train,X_test,y_train,y_test,group_train,group_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.group_train=group_train
        self.group_test=group_test
        self.model=model
        self.y_pred = None

    # Combine the compilation and training
    def compile_and_train(self, lr=1e-2):
        self.compile_model(lr)
        return self.train_model() 
    
    # compile a model using these specific metrics
    def compile_model(self, lr):
        metrics = [
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            self.f1,
        ]

        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=metrics
        )

    def train_model(self):
        # Use EarlyStopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            mode='max'
        )

        # Calculate the class wheights for the model, improves predictive equality
        class_weights = {0: 1., 1: np.sum(self.y_train == 0) / np.sum(self.y_train == 1)}

        hist = self.model.fit(
                        self.X_train, self.y_train,
                        class_weight=class_weights,batch_size=512,
                        epochs=100, # set lower if you only want to train for short period to get approximat results
                        callbacks=[early_stopping],
                        verbose=1,
                        validation_split=0.1 # Use 10% of training set as validation for EarlyStopping
        )
        # return the training history for possible visualization
        return hist

    # Evaluate a model by passing its output into the evaluate-function
    def score_keras_model(self):
        # Score the test set
        predictions = self.model.predict(self.X_test).flatten()
        self.y_pred = predictions
        self.plot_auc_accuracy_nn()

    def plot_auc_accuracy_nn(self, FIXED_FPR=0.05):
        # Calculating AUC and plotting the ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, color='purple', label='AUC = %0.2f' % roc_auc)
        axes[0].plot([0, 1], [0, 1], color='magenta', linestyle='--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')

        # Calculating Accuracy and plotting it
        tpr_fixed_fpr = tpr[fpr < FIXED_FPR][-1]
        fpr_fixed_fpr = fpr[fpr < FIXED_FPR][-1]
        threshold_fixed_fpr = thresholds[fpr < FIXED_FPR][-1]

        print("AUC:", roc_auc)
        to_pct = lambda x: str(round(x, 4) * 100) + "%"
        print("TPR: ", to_pct(tpr_fixed_fpr), "\nFPR: ", to_pct(fpr_fixed_fpr), "\nThreshold: ", round(threshold_fixed_fpr, 2))

        # Calling get_fairness_metrics to calculate predictive equality
        predictive_equality, disparities_df = self.get_fairness_metrics_nn(self.group_test,self.y_test,self.y_pred, FIXED_FPR)
        print("Predictive Equality: ", to_pct(predictive_equality))

        # extract the predicted class labels
        thresh = round(threshold_fixed_fpr, 2)
        predictions = np.where(self.y_pred > thresh, 1, 0) # or use the threshold instead of 0.5?

        accuracy = accuracy_score(y_test, predictions)
        axes[1].bar(['Accuracy'], [accuracy], color=['purple'])
        axes[1].set_ylim([0, 1.5])
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_xticks([])
        axes[1].tick_params(axis='y', labelsize=10)

        # Add accuracy value as text label on the accuracy bar plot
        axes[1].text(0, accuracy + 0.05, '{:.2f}'.format(accuracy), horizontalalignment='center', fontsize=15)
        # Adjusting the layout of the subplots and displaying the plot
        fig.tight_layout()
        plt.show()

        #ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        self.plot_confmat_classreport_nn(predictions)
        print(classification_report(self.y_test, predictions))        


    def get_fairness_metrics_nn(self,groups,y_true, y_pred,FIXED_FPR=0.05):
        labels = self.y_test
        groups = self.group_test
        g = Group()
        aequitas_df = pd.DataFrame(
            {"score": y_pred,
             "label_value": y_true,
             "group": groups}
        )

        # Use aequitas to compute confusion matrix metrics for every group.
        disparities_df = g.get_crosstabs(aequitas_df, score_thresholds={"score_val": [FIXED_FPR]})[0]

        # Predictive equality is the differences in FPR (we use ratios in the paper)
        predictive_equality = disparities_df["fpr"].min() / disparities_df["fpr"].max()
        return predictive_equality, disparities_df
        
    
    def plot_confmat_classreport_nn(self, predictions):
        
        # Calculating Confusion Matrix and plotting it
        CM = confusion_matrix(self.y_test, predictions)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        sns.heatmap(CM, annot=True, fmt='g', cmap='Pastel1', ax=axes)
        axes.set_xlabel('Predicted')
        axes.set_ylabel('Truth')
        axes.set_title('Confusion Matrix')

        # Adjusting the layout of the subplots and displaying the plot
        fig.tight_layout()
        plt.show()

    def f1(self, y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    # --- Two currently unused metrics ---
    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def focal_loss(self, gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        return focal_loss_fixed

    def save_model(self,filename):
        # Save the model to disk using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Return the path to the saved model file
        return filename
