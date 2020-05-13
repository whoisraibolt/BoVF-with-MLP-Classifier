#!/usr/bin/env python
# coding: utf-8

# Imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from timer import Timer
import globals

# Call function train
def train(solver,
          momentum,
          hidden_layer_sizes,
          alpha,
          learning_rate,
          learning_rate_init,
          random_state,
          max_iter,
          descriptor_name,
          model_name):

    with Timer() as timer:
        # MLP Model
        MLP = MLPClassifier(solver = solver,
                            momentum = momentum,
                            hidden_layer_sizes = hidden_layer_sizes,
                            alpha = alpha,
                            learning_rate = learning_rate,
                            learning_rate_init = learning_rate_init,
                            random_state = random_state,
                            max_iter = max_iter)
        
        MLP.fit(globals.train_feature_vec[0],
                globals.train_feature_vec[1])
        
        print('%s %s\n' % (descriptor_name, model_name), file = globals.file)
        print('Training Set Evaluation\n', file = globals.file)
        print('%s + M = %s + %s - train score: %.2f\n' % (solver,
                                                          momentum,
                                                          learning_rate,
                                                          MLP.score(globals.train_feature_vec[0],
                                                                    globals.train_feature_vec[1])),
                                                                    file = globals.file)

        return MLP

    print('Time:', timer, '\n', file = globals.file)

# Call function test
def test(model,
         solver,
         momentum,
         learning_rate,
         descriptor_name,
         model_name):

    with Timer() as timer:
        # MLP Model
        print('%s %s\n' % (descriptor_name, model_name), file = globals.file)
        print('Testing Set Evaluation\n', file = globals.file)
        MLP_predict = model.predict(globals.test_feature_vec[0])
        
        print('%s + M = %s + %s - test score: %.2f\n' % (solver,
                                                         momentum,
                                                         learning_rate,
                                                         accuracy_score(globals.test_feature_vec[1],
                                                                        MLP_predict)),
                                                                        file = globals.file)

        return MLP_predict
        
    print('Time:', timer, '\n', file = globals.file)

# Call function classificationReport
def classificationReport(model,
                         predict,
                         descriptor_name,
                         model_name):
    #  MLP Model 
    print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

    # Classification report for Classifier
    print("Classification report for Classifier: \n\n%s: \n\n %s" % (model,
                                                                     classification_report(globals.test_feature_vec[1],
                                                                                           predict)),
                                                                                           file = globals.file)